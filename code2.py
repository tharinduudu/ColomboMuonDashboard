# rowcount.py — GitHub auto-load, 2σ clip, signed deviations,
# colored segmented selector for 4 views, stacked charts.

from __future__ import annotations
from datetime import timedelta
import io
from typing import List, Dict, Any

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Muon Log Dashboard", layout="wide")
st.title("Colombo Detector Log Dashboard")

# ---------------- GitHub source (EDIT THESE) ----------------
GH_OWNER    = "nithyabiyanwila"      # e.g., "nithyabiyanwila"
GH_REPO     = "colombo_detector"          # repo that contains the data folder
GH_BRANCH   = "main"
GH_DATA_DIR = "colombo_data"               # folder in repo containing .log/.txt/.csv

# Private repo? put GH_TOKEN in Streamlit Secrets
GH_TOKEN = st.secrets.get("GH_TOKEN", None)

# ---------------- Fixed parsing settings ----------------
SEP = r"[ ]+"          # spaces (1+)
ENGINE = "python"      # CSV engine for regex sep
HAS_HEADER = False
SKIP_ROWS = 0
TIME_COL_1B = 6
C1_1B, C3_1B, C4_1B, C5_1B = 1, 3, 4, 5
TIMESTAMP_SPLIT_5 = True
CUSTOM_FMT = "%a %b %d %H:%M:%S %Y"
MAX_POINTS = 10_000    # minute-level plotting cap

# ---------------- Helpers: parsing ----------------
def clean_numeric(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.strip()
    s2 = s2.str.replace(",", "", regex=False)
    s2 = s2.str.replace(r"\b(None|nan|NaN)\b", "", regex=True)
    s2 = s2.str.replace(r"[^0-9.\-eE]", "", regex=True)
    return pd.to_numeric(s2, errors="coerce")

def to_datetime_joined(df: pd.DataFrame, start_idx: int, fmt: str, split5: bool) -> pd.Series:
    if split5:
        parts = [df.iloc[:, start_idx + k].astype(str).str.strip() for k in range(5)]
        txt = parts[0] + " " + parts[1] + " " + parts[2] + " " + parts[3] + " " + parts[4]
    else:
        txt = df.iloc[:, start_idx].astype(str).str.strip()
    return pd.to_datetime(txt, format=fmt, errors="coerce")

def sigma_clip_series(s: pd.Series, sigma: float = 2.0) -> pd.Series:
    m = s.mean(skipna=True); sd = s.std(skipna=True)
    if pd.isna(m) or pd.isna(sd) or sd == 0:
        return s
    return s.where((s - m).abs() <= sigma * sd)

def align_end_to_full_hours(start_dt: pd.Timestamp, end_dt: pd.Timestamp,
                            min_t: pd.Timestamp, max_t: pd.Timestamp) -> pd.Timestamp:
    # Clamp both, then set end = start + floor((end-start)/1h)*1h
    start_dt = max(min_t, min(start_dt, max_t))
    end_dt   = max(min_t, min(end_dt,   max_t))
    if end_dt < start_dt:
        end_dt = start_dt
    hours = int((end_dt - start_dt).total_seconds() // 3600)
    hours = max(0, min(hours, int((max_t - start_dt).total_seconds() // 3600)))
    return start_dt + timedelta(hours=hours)

# ---------------- Helpers: GitHub fetching ----------------
def _gh_headers() -> Dict[str, str]:
    h = {"Accept": "application/vnd.github+json"}
    if GH_TOKEN:
        h["Authorization"] = f"Bearer {GH_TOKEN}"
    return h

@st.cache_data(show_spinner=False)
def gh_list_dir(owner: str, repo: str, path: str, ref: str) -> List[Dict[str, Any]]:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    params = {"ref": ref}
    r = requests.get(url, headers=_gh_headers(), params=params, timeout=30)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    data = r.json()
    return [data] if isinstance(data, dict) else data

def gh_walk(owner: str, repo: str, root_path: str, ref: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    stack = [root_path]
    while stack:
        p = stack.pop()
        items = gh_list_dir(owner, repo, p, ref)
        for it in items:
            if it.get("type") == "dir":
                stack.append(it["path"])
            elif it.get("type") == "file":
                out.append(it)
    return out

def is_data_file(name: str) -> bool:
    n = name.lower()
    return n.endswith(".log") or n.endswith(".txt") or n.endswith(".csv")

def raw_url_for(path: str) -> str:
    return f"https://raw.githubusercontent.com/{GH_OWNER}/{GH_REPO}/{GH_BRANCH}/{path}"

@st.cache_data(show_spinner=True, ttl=300)
def fetch_and_combine_files() -> pd.DataFrame:
    files = [it for it in gh_walk(GH_OWNER, GH_REPO, GH_DATA_DIR, GH_BRANCH) if it.get("type") == "file"]
    files = [f for f in files if is_data_file(f.get("name", ""))]
    if not files:
        return pd.DataFrame(columns=["time","Total (per min)","Top + Middle","Top + Bottom","Middle + Bottom"])

    dfs: List[pd.DataFrame] = []
    errs: List[str] = []

    for f in sorted(files, key=lambda x: x.get("name","")):
        try:
            r = requests.get(raw_url_for(f["path"]), headers=_gh_headers(), timeout=60)
            r.raise_for_status()
            txt = r.text
            df = pd.read_csv(io.StringIO(txt),
                             sep=SEP, engine=ENGINE,
                             header=(0 if HAS_HEADER else None),
                             dtype=str, skiprows=SKIP_ROWS,
                             on_bad_lines="skip")
            t0 = TIME_COL_1B - 1
            i1, i3, i4, i5 = C1_1B - 1, C3_1B - 1, C4_1B - 1, C5_1B - 1
            need = max(i1,i3,i4,i5, t0 if not TIMESTAMP_SPLIT_5 else t0+4)
            if df.shape[1] <= need:
                errs.append(f"{f['name']}: not enough columns (has {df.shape[1]}, needs ≥ {need+1})"); continue
            time = to_datetime_joined(df, t0, CUSTOM_FMT, TIMESTAMP_SPLIT_5)
            if time.isna().all():
                errs.append(f"{f['name']}: time parse failed"); continue
            out = pd.DataFrame({
                "time": time,
                "Total (per min)": clean_numeric(df.iloc[:, i1]),
                "Top + Middle":    clean_numeric(df.iloc[:, i3]),
                "Top + Bottom":    clean_numeric(df.iloc[:, i4]),
                "Middle + Bottom": clean_numeric(df.iloc[:, i5]),
            }).dropna(subset=["time"])
            dfs.append(out)
        except Exception as e:
            errs.append(f"{f.get('name','?')}: {e}")

    if not dfs:
        return pd.DataFrame(columns=["time","Total (per min)","Top + Middle","Top + Bottom","Middle + Bottom"])

    data = pd.concat(dfs, ignore_index=True).sort_values("time").reset_index(drop=True)
    if errs:
        st.warning("Some files were skipped:\n- " + "\n- ".join(errs[:10]) + ("" if len(errs)<=10 else f"\n...and {len(errs)-10} more"))
    return data

# ---------------- Load & combine data ----------------
work_all = fetch_and_combine_files()
if work_all.empty:
    st.error("No data files found. Check GH_OWNER/GH_REPO/GH_BRANCH/GH_DATA_DIR.")
    st.stop()

# ---------------- Time range (typed only) ----------------
min_t, max_t = work_all["time"].min(), work_all["time"].max()
st.subheader("Time range")
c1, c2 = st.columns(2)
start_txt = c1.text_input("Start (YYYY-MM-DD HH:MM:SS)", value=min_t.strftime("%Y-%m-%d %H:%M:%S"))
end_txt   = c2.text_input("End (YYYY-MM-DD HH:MM:SS)",   value=max_t.strftime("%Y-%m-%d %H:%M:%S"))
st.caption(f"File time range: **{min_t}** → **{max_t}**")

typed_start = pd.to_datetime(start_txt, errors="coerce")
typed_end   = pd.to_datetime(end_txt,   errors="coerce")
if pd.isna(typed_start):
    st.error("Invalid start time. Use YYYY-MM-DD HH:MM:SS."); st.stop()
if pd.isna(typed_end):
    st.error("Invalid end time. Use YYYY-MM-DD HH:MM:SS."); st.stop()

if typed_start < min_t or typed_start > max_t:
    st.error("Start time was outside the file's time range and has been aligned to the nearest bound.")
if typed_end < min_t or typed_end > max_t:
    st.error("End time was outside the file's time range and has been aligned to the nearest bound.")

start_dt = max(min_t, min(typed_start, max_t))
orig_end_clamped = max(min_t, min(typed_end,   max_t))
end_dt = align_end_to_full_hours(start_dt, orig_end_clamped, min_t, max_t)
if end_dt != orig_end_clamped:
    st.info(f"End time was aligned to **{end_dt}** so the window covers a whole number of hours from the start.")

mask = (work_all["time"] >= start_dt) & (work_all["time"] <= end_dt)
work_sel = work_all.loc[mask].copy()

# ---------------- 2σ clipping per series ----------------
coinc_cols = ["Top + Middle", "Top + Bottom", "Middle + Bottom"]
work_clip = work_sel.copy()
for c in coinc_cols + ["Total (per min)"]:
    work_clip[c] = sigma_clip_series(work_clip[c], sigma=2.0)

st.caption("2σ clipping applied to counts in the selected window")

# ---------------- Deviations (from CLIPPED data) ----------------
# Minute-level signed % change vs overall mean (selected window)
minute_means = {c: work_clip[c].mean(skipna=True) for c in coinc_cols}
for c in coinc_cols:
    m = minute_means[c]
    work_clip[f"{c} Δ% (min vs overall)"] = ((work_clip[c] - m) / m * 100.0) if (pd.notna(m) and m != 0) else pd.NA

# Hourly mean (from clipped minutes)
hourly = (work_clip.set_index("time").resample("H").mean(numeric_only=True).dropna(how="all"))
# Hourly signed % change vs overall hourly mean
hourly_dev = pd.DataFrame(index=hourly.index)
for c in coinc_cols:
    overall = hourly[c].mean(skipna=True)
    hourly_dev[f"{c} Δ% (hour mean vs overall)"] = (
        (hourly[c] - overall) / overall * 100.0 if (pd.notna(overall) and overall != 0) else pd.NA
    )

# Downsample minute charts only
n = len(work_clip)
stride = max(1, n // MAX_POINTS)
plot_min = work_clip.iloc[::stride].copy()

# ---------------- Colored segmented selector (button-like) ----------------
# CSS to make the radio horizontal, pill-shaped, and colored per option.
st.markdown("""
<style>
/* horizontal layout */
div.stRadio > div {flex-direction: row;}
/* pill look for each label */
div.stRadio label { 
  border: 1px solid #d1d5db; 
  border-radius: 9999px; 
  padding: 0.35rem 0.9rem; 
  margin-right: 0.5rem; 
  cursor: pointer; 
  user-select: none;
  font-weight: 600;
}
/* colors per option (order matters) */
div.stRadio label:nth-of-type(1){ background:#e6f0ff; color:#1d4ed8; border-color:#1d4ed8;}
div.stRadio label:nth-of-type(2){ background:#e9fff5; color:#059669; border-color:#059669;}
div.stRadio label:nth-of-type(3){ background:#fff3e6; color:#d97706; border-color:#d97706;}
div.stRadio label:nth-of-type(4){ background:#ffecec; color:#b91c1c; border-color:#b91c1c;}
/* subtle pressed effect when selected */
div.stRadio label:has(input:checked){
  box-shadow: inset 0 0 0 2px rgba(0,0,0,0.06);
  filter: saturate(1.2);
}
</style>
""", unsafe_allow_html=True)

VIEW_OPTIONS = [
    "counts per minute",
    "counts per hour",
    "percentage change per minute",
    "percentage change per hour",
]

# Default to "percentage change per hour" on first load, then remember the user's choice
if "view" not in st.session_state:
    st.session_state["view"] = "percentage change per hour"

default_idx = VIEW_OPTIONS.index(st.session_state["view"])
view = st.radio(
    "View",
    VIEW_OPTIONS,
    index=default_idx,            # sets the initial selection
    horizontal=True,
    label_visibility="collapsed",
    key="view"                    # Streamlit will remember the selection on reruns
)

st.write("")  # small spacer

# ---------------- Render selected view (stacked charts) ----------------
if view == "counts per minute":
    st.subheader("top and middle layers")
    st.line_chart(plot_min.set_index("time")[["Top + Middle"]])
    st.subheader("top and bottom layers")
    st.line_chart(plot_min.set_index("time")[["Top + Bottom"]])
    st.subheader("middle and bottom layers")
    st.line_chart(plot_min.set_index("time")[["Middle + Bottom"]])

elif view == "counts per hour":
    st.subheader("top and middle layers")
    st.line_chart(hourly[["Top + Middle"]])
    st.subheader("top and bottom layers")
    st.line_chart(hourly[["Top + Bottom"]])
    st.subheader("middle and bottom layers")
    st.line_chart(hourly[["Middle + Bottom"]])

elif view == "percentage change per minute":
    st.subheader("top and middle layers")
    st.line_chart(plot_min.set_index("time")[["Top + Middle Δ% (min vs overall)"]])
    st.subheader("top and bottom layers")
    st.line_chart(plot_min.set_index("time")[["Top + Bottom Δ% (min vs overall)"]])
    st.subheader("middle and bottom layers")
    st.line_chart(plot_min.set_index("time")[["Middle + Bottom Δ% (min vs overall)"]])

else:  # "percentage change per hour"
    st.subheader("top and middle layers")
    st.line_chart(hourly_dev[[c for c in hourly_dev.columns if c.startswith("Top + Middle")]])
    st.subheader("top and bottom layers")
    st.line_chart(hourly_dev[[c for c in hourly_dev.columns if c.startswith("Top + Bottom")]])
    st.subheader("middle and bottom layers")
    st.line_chart(hourly_dev[[c for c in hourly_dev.columns if c.startswith("Middle + Bottom")]])
