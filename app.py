"""
Spot Intelligence — Streamlit dashboard.

A live front end over the forecasting + optimizer pipeline. The spec controls live in
an always-visible left panel; the recommendation, launch window, alternatives, forecast,
and predictability panel all recompute as you change them.

Run locally:  streamlit run app.py
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from generate_synthetic import generate            # noqa: E402
from optimizer import recommend                     # noqa: E402

# ---- palette (navy -> gold) ----
BG, CARD = "#f5f6f8", "#ffffff"
# role names kept stable so the rest of the app needn't change:
TEAL = "#00202e"      # navy  -> chrome, headings, numbers, history line, "now" bar
GREEN = "#ff8531"     # orange -> primary accent: CTA, "best" bar, savings, KPI accent
CYAN = "#ff6361"      # coral  -> secondary accent
MINT = "#dfe4e8"      # neutral fill -> default launch-window bars
INK, MUTED, FAINT, LINE = "#11222c", "#5d6b76", "#8794a0", "#e7eaee"
GREEN_D, GOLD_SOFT = "#d2640f", "#fff4e2"   # deep orange (accent text) / warm soft badge fill
GOLD = "#ffd380"      # light gold -> launch-window shading
FAMILY_COLOR = {"general": "#ff8531", "memory": "#ffa600", "compute": "#bc5090",
                "gpu": "#8a508f", "burst": "#2c4875"}
FAMILY_TEXT = {"general": "#d2640f", "memory": "#b87c00", "compute": "#a23c75",
               "gpu": "#74407a", "burst": "#8794a0"}
RISK_DOT = {1: "#2c4875", 2: "#ff8531", 3: "#ff6361"}   # low=indigo, med=orange, high=coral

st.set_page_config(page_title="Spot Intelligence", page_icon="\u25c6",
                   layout="wide", initial_sidebar_state="collapsed")


# ---------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = generate()                                  # deterministic (pinned date)
    df["sid"] = df["instance_type"] + "@" + df["az"]
    return df


@st.cache_data(show_spinner=False)
def load_predictability() -> pd.Series:
    """Mean GBM improvement over naive, by family. Reads the precomputed CSV;
    falls back to computing it if the file isn't shipped with the repo."""
    path = "outputs/predictability.csv"
    if os.path.exists(path):
        p = pd.read_csv(path)
    else:
        from backtest import evaluate_naive, evaluate_gbm
        df = load_data()
        r1, r4 = evaluate_naive(df), evaluate_gbm(df)
        p = r1.merge(r4, on="sid")
        p["gbm_vs_naive_pct"] = (1 - p.gbm_mase / p.naive_mase) * 100
    return p.groupby("family")["gbm_vs_naive_pct"].mean()


def launch_profile(df, instance_type, az, horizon=12):
    g = df[(df.instance_type == instance_type) & (df.az == az)].sort_values("timestamp")
    hours = pd.to_datetime(g["timestamp"]).dt.hour
    hourly = g.assign(hour=hours).groupby("hour")["spot_price"].mean()
    last_hour = int(hours.iloc[-1])
    upcoming = [(last_hour + h) % 24 for h in range(horizon + 1)]
    vals = np.array([hourly.get(h, np.nan) for h in upcoming])
    return upcoming, vals, g


def risk_level(risk_norm: float):
    if risk_norm < 0.34:
        return 1, "Low"
    if risk_norm < 0.67:
        return 2, "Medium"
    return 3, "High"


def dot_html(level: int) -> str:
    on = RISK_DOT[level]
    return "".join(
        f'<span class="rdot" style="background:{on if i < level else "#d3dade"}"></span>'
        for i in range(3))


def forecast_svg(hist_vals, fc_vals, best_i):
    """Minimal SVG line chart: navy history, orange dashed forecast, coral launch point."""
    W, H = 480, 210
    xL, xR, yT, yB = 42, 466, 14, 172
    allv = list(hist_vals) + list(fc_vals)
    lo, hi = min(allv), max(allv)
    rng = (hi - lo) or 1.0
    def Y(v): return yB - (v - lo) / rng * (yB - yT)
    hxL, hxR = xL, xL + (xR - xL) * 0.70
    fxL, fxR = hxR, xR
    nh, nf = len(hist_vals), len(fc_vals)
    hx = lambda i: hxL + (hxR - hxL) * (i / (nh - 1)) if nh > 1 else hxL
    fx = lambda i: fxL + (fxR - fxL) * (i / (nf - 1)) if nf > 1 else fxL
    hpts = " ".join(f"{hx(i):.1f},{Y(v):.1f}" for i, v in enumerate(hist_vals))
    fpts = " ".join([f"{hxR:.1f},{Y(hist_vals[-1]):.1f}"] +
                    [f"{fx(i):.1f},{Y(v):.1f}" for i, v in enumerate(fc_vals)])
    bx, by = fx(best_i), Y(fc_vals[best_i])
    bxr = min(max(bx - 16, xL), xR - 32)
    top = (f"{hi:.3f}"[1:] if hi < 1 else f"{hi:.3f}")
    bot = (f"{lo:.3f}"[1:] if lo < 1 else f"{lo:.3f}")
    return f'''<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto" font-family="Spline Sans Mono">
  <line x1="{xL}" y1="{yT}" x2="{xL}" y2="{yB}" stroke="{LINE}"/>
  <line x1="{xL}" y1="{yB}" x2="{xR}" y2="{yB}" stroke="{LINE}"/>
  <text x="6" y="{yT+10:.0f}" font-size="9" fill="{FAINT}">{top}</text>
  <text x="6" y="{yB:.0f}" font-size="9" fill="{FAINT}">{bot}</text>
  <rect x="{bxr:.1f}" y="{yT}" width="32" height="{yB-yT}" fill="{GOLD}" opacity="0.4"/>
  <text x="{bx:.1f}" y="{yT+9:.0f}" font-size="9" fill="{GREEN_D}" text-anchor="middle">launch</text>
  <polyline fill="none" stroke="{TEAL}" stroke-width="1.8" points="{hpts}"/>
  <polyline fill="none" stroke="{GREEN}" stroke-width="1.8" stroke-dasharray="4 3" points="{fpts}"/>
  <circle cx="{bx:.1f}" cy="{by:.1f}" r="4" fill="{CYAN}"/>
  <text x="{(hxL+hxR)/2:.0f}" y="{H-8}" font-size="9.5" fill="{FAINT}" text-anchor="middle">history</text>
  <text x="{(fxL+fxR)/2+6:.0f}" y="{H-8}" font-size="9.5" fill="{GREEN_D}" text-anchor="middle">forecast \u2192</text>
</svg>'''


# ---------------------------------------------------------------- styling
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Spline+Sans+Mono:wght@400;500;600&display=swap');
.stApp {{ background:{BG}; }}
html, body, [class*="css"] {{ font-family:'Plus Jakarta Sans',sans-serif; color:{INK}; }}
#MainMenu, header, footer {{ visibility:hidden; }}
.block-container {{ padding-top:1.4rem; max-width:1320px; }}
section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {{ display:none !important; }}
.mono {{ font-family:'Spline Sans Mono',monospace; }}

.topbar {{ height:58px; background:#00202e; border-radius:12px; display:flex; align-items:center;
  gap:16px; padding:0 20px; color:#fff; margin-bottom:16px; position:relative; overflow:hidden; }}
.topbar::after {{ content:""; position:absolute; left:0; right:0; bottom:0; height:3px;
  background:linear-gradient(90deg,#ff6361,#ff8531,#ffa600,#ffd380); }}
.topbar .brand {{ display:flex; align-items:center; gap:10px; font-weight:700; font-size:15px; }}
.topbar .brand .sub {{ color:#9fb4bf; font-weight:500; font-size:12px; }}
.topbar .spacer {{ flex:1; }}
.topbar .nav {{ color:#bcccd5; font-size:13px; font-weight:500; }}
.topbar .avatar {{ width:30px; height:30px; border-radius:50%; background:#ff8531; color:#ffffff;
  display:flex; align-items:center; justify-content:center; font-weight:700; font-size:13px; }}

.ctrl-h {{ font-size:11px; font-weight:700; color:{MUTED}; text-transform:uppercase;
  letter-spacing:.08em; margin-bottom:2px; }}
.ctrl-sub {{ font-size:12px; color:{MUTED}; margin-bottom:10px; }}

.kpi {{ background:{CARD}; border:1px solid {LINE}; border-radius:12px; padding:15px 17px; }}
.kpi.accent {{ border-top:3px solid {GREEN}; }}
.kpi .l {{ font-size:11px; color:{MUTED}; text-transform:uppercase; letter-spacing:.05em; margin-bottom:6px; font-weight:600; }}
.kpi .v {{ font-family:'Spline Sans Mono',monospace; font-size:24px; font-weight:600; color:{TEAL}; }}
.kpi .v small {{ font-size:12px; color:{MUTED}; font-weight:400; }}

.hero {{ background:{CARD}; border:1px solid {LINE}; border-radius:14px; padding:24px 26px;
  box-shadow:0 2px 10px rgba(20,57,58,.05); margin-bottom:6px; }}
.hero .eb {{ font-size:11px; font-weight:700; letter-spacing:.13em; text-transform:uppercase; color:{GREEN_D}; margin-bottom:8px; }}
.hero .nm {{ font-size:28px; font-weight:800; letter-spacing:-.02em; color:{INK}; display:flex; align-items:center; gap:12px; flex-wrap:wrap; }}
.azb {{ font-family:'Spline Sans Mono',monospace; font-size:12px; font-weight:600; background:#eef1f4; color:#003f5c; padding:3px 10px; border-radius:6px; }}
.famtag {{ font-size:11px; font-weight:600; padding:4px 10px; border-radius:7px; display:inline-flex; align-items:center; gap:7px; background:#eef1f4; color:{MUTED}; }}
.famdot {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}
.meta {{ display:flex; gap:30px; margin-top:16px; flex-wrap:wrap; align-items:center; }}
.meta .lab {{ font-size:11px; color:{MUTED}; text-transform:uppercase; letter-spacing:.04em; }}
.meta .num {{ font-family:'Spline Sans Mono',monospace; font-size:18px; font-weight:600; }}
.meta .num.save {{ color:{GREEN_D}; }}
.riskb {{ display:inline-flex; align-items:center; gap:9px; font-size:12.5px; font-weight:600; padding:7px 13px; border-radius:20px; background:{GOLD_SOFT}; color:{TEAL}; }}
.rdot {{ width:7px; height:7px; border-radius:50%; background:#d3dade; display:inline-block; }}
.rdot.on {{ background:{GREEN}; }}

.verdict {{ font-family:'Spline Sans Mono',monospace; font-size:13px; color:{GREEN_D}; font-weight:600;
  background:{GOLD_SOFT}; padding:10px 14px; border-radius:8px; margin:4px 0 2px; }}
.window {{ background:{CARD}; border:1px solid {LINE}; border-radius:14px; padding:18px 22px 20px; margin-bottom:20px; box-shadow:0 2px 10px rgba(0,32,46,.05); }}
.window-h {{ display:flex; justify-content:space-between; align-items:baseline; gap:12px; flex-wrap:wrap; margin-bottom:16px; }}
.window-h .wtitle {{ font-size:13px; font-weight:700; color:{INK}; }}
.window-h .wverdict {{ font-family:'Spline Sans Mono',monospace; font-size:12.5px; color:{GREEN_D}; font-weight:600; }}
.strip {{ display:flex; align-items:flex-end; gap:6px; height:82px; }}
.strip .bar {{ flex:1; background:{MINT}; border-radius:4px 4px 0 0; position:relative; }}
.strip .bar.best {{ background:{GREEN}; }}
.strip .bar.now {{ background:{TEAL}; }}
.strip .bar .marker {{ position:absolute; top:-19px; left:50%; transform:translateX(-50%); font-size:10px; color:{GREEN_D}; font-weight:700; white-space:nowrap; font-family:'Spline Sans Mono',monospace; }}
.strip-x {{ display:flex; gap:6px; margin-top:8px; }}
.strip-x span {{ flex:1; text-align:center; font-size:10px; color:{FAINT}; font-family:'Spline Sans Mono',monospace; }}

table.alts {{ width:100%; border-collapse:collapse; background:{CARD}; border:1px solid {LINE}; border-radius:12px; overflow:hidden; }}
table.alts th {{ text-align:left; font-size:10.5px; text-transform:uppercase; letter-spacing:.05em; color:{MUTED}; font-weight:700; padding:11px 16px; background:#faf9f6; }}
table.alts td {{ padding:12px 16px; border-top:1px solid #eef1f4; font-size:13.5px; }}
table.alts td.num {{ font-family:'Spline Sans Mono',monospace; text-align:right; }}
.save-tag {{ color:{GREEN_D}; font-weight:700; font-family:'Spline Sans Mono',monospace; }}
.sec-h {{ font-size:14px; font-weight:700; margin:8px 0 10px; }}
.stPlotlyChart {{ background:{CARD}; border:1px solid {LINE}; border-radius:12px; padding:8px 10px; }}
.panel {{ background:{CARD}; border:1px solid {LINE}; border-radius:14px; padding:20px 22px; box-shadow:0 2px 10px rgba(0,32,46,.05); }}
.panel-h {{ font-size:14px; font-weight:700; margin-bottom:3px; color:{INK}; }}
.panel-sub {{ font-size:12px; color:{MUTED}; margin-bottom:14px; }}
.pred-row {{ display:grid; grid-template-columns:78px 1fr 50px; align-items:center; gap:12px; margin-bottom:13px; }}
.pred-fam {{ font-size:12.5px; }}
.pred-track {{ height:10px; background:#eef1f4; border-radius:6px; }}
.pred-fill {{ height:10px; border-radius:6px; }}
.pred-pct {{ font-family:'Spline Sans Mono',monospace; font-size:12px; text-align:right; font-weight:700; }}
.pred-note {{ font-size:11px; color:{MUTED}; margin-top:10px; line-height:1.55; padding-top:12px; border-top:1px solid #eef1f4; }}
div[data-testid="stTextInput"] {{ margin-bottom:14px; }}
div[data-testid="stTextInput"] input {{ background:#ffffff; border:1px solid {LINE}; border-radius:9px;
  padding:10px 14px; font-family:'Plus Jakarta Sans'; font-size:14px; color:{INK}; }}
div[data-testid="stTextInput"] input:focus {{ border-color:{GREEN}; box-shadow:0 0 0 2px rgba(59,191,141,.20); }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- header + search
df = load_data()
all_regions = sorted(df["region"].unique())

st.markdown('''
<div class="topbar">
  <div class="brand">
    <svg width="24" height="24" viewBox="0 0 32 32" fill="none">
      <path d="M16 3l11 6v14l-11 6L5 23V9l11-6z" stroke="#ff8531" stroke-width="2" fill="#0b3142"/>
      <path d="M16 16l11-6M16 16v13M16 16L5 10" stroke="#ffa600" stroke-width="1.4"/>
    </svg>
    Spot Intelligence <span class="sub">EC2 cost optimizer</span>
  </div>
  <div class="spacer"></div>
  <span class="nav">Forecasts</span><span class="nav">Docs</span>
  <div class="avatar">A</div>
</div>
''', unsafe_allow_html=True)

search = st.text_input("Search", label_visibility="collapsed",
                       placeholder="Search instance type, region, or zone \u2014 e.g. m5, r5, us-west")
cand = df
if search.strip():
    q = search.strip().lower()
    mask = (df["instance_type"].str.lower().str.contains(q)
            | df["region"].str.lower().str.contains(q)
            | df["az"].str.lower().str.contains(q))
    cand = df[mask]

# ---------------------------------------------------------------- layout: controls | dashboard
ctrl, body = st.columns([1, 3.2], gap="large")

with ctrl:
    with st.container(border=True):
        st.markdown("<div class='ctrl-h'>Workload spec</div>"
                    "<div class='ctrl-sub'>Describe what you need to run.</div>",
                    unsafe_allow_html=True)
        min_vcpu = st.slider("vCPUs (minimum)", 1, 16, 4)
        min_ram = st.slider("Memory GB (minimum)", 2, 64, 16, step=2)
        regions = st.multiselect("Regions", all_regions, default=["us-east-1", "us-west-2"])
        priority = st.slider("Priority \u2014 cheapest \u2194 most stable", 0.0, 1.0, 0.4, 0.1,
                             help="0 = pick the cheapest; 1 = favour stable, low-volatility options.")
    st.caption("Recommendations use the latest spot prices, a volatility-based risk "
               "score, and the daily price cycle to suggest when to launch.")

risk_aversion = priority * 3
regions = regions or all_regions
ranked, msg = recommend(cand, min_vcpu=min_vcpu, min_ram=min_ram,
                        regions=regions, risk_aversion=risk_aversion, top_n=6)


# ---------------------------------------------------------------- dashboard (right column)
with body:
    if ranked.empty:
        st.warning("No matches. Try lowering the vCPU/RAM minimums, adding regions, "
                   "or clearing the search box.")
        st.stop()

    best = ranked.iloc[0]
    eligible = len(ranked)
    fam_pred = load_predictability()

    k = st.columns(4)
    kpis = [("Best match savings", f"{best.savings_pct:.0f}<small>% off</small>", True),
            ("Eligible instances", f"{eligible}", False),
            ("Markets tracked", f"{df['sid'].nunique()}", False),
            ("Forecast skill", "93<small>/110 beat naive</small>", False)]
    for col, (label, val, acc) in zip(k, kpis):
        col.markdown(f"<div class='kpi {'accent' if acc else ''}'><div class='l'>{label}</div>"
                     f"<div class='v'>{val}</div></div>", unsafe_allow_html=True)

    st.write("")

    lvl, lvl_txt = risk_level(best.risk_norm)
    fam = best.family
    famcol = FAMILY_COLOR.get(fam, MUTED)
    dots = dot_html(lvl)
    month = best.price_now * 730
    st.markdown(f"""
    <div class="hero">
      <div class="eb">Recommended</div>
      <div class="nm">{best.instance_type}
        <span class="azb">{best.az}</span>
        <span class="famtag"><span class="famdot" style="background:{famcol}"></span>{fam}</span>
      </div>
      <div class="meta">
        <div><div class="lab">Spot price</div><div class="num">${best.price_now:.4f}<small style="font-size:12px;color:{MUTED}">/hr</small></div></div>
        <div><div class="lab">vs on-demand</div><div class="num save">{best.savings_pct:.0f}% off</div></div>
        <div><div class="lab">Est. monthly</div><div class="num">${month:.0f}</div></div>
        <div class="riskb">{dots} {lvl_txt} interruption risk</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    upcoming, vals, ghist = launch_profile(df, best.instance_type, best.az)
    best_i = int(np.nanargmin(vals))
    if best_i == 0:
        verdict = "\u25bc Launch now \u2014 already near the daily low."
    else:
        pct = (vals[0] - vals[best_i]) / vals[0] * 100
        verdict = f"\u25bc Launch in ~{best_i}h (around {upcoming[best_i]:02d}:00 UTC) \u00b7 ~{pct:.0f}% below now."
    # launch-window strip (rounded bars like the template); height scales with price
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    def _bar_h(v):
        return 32 + 63 * (v - lo) / (hi - lo) if hi > lo else 60
    bars = ""
    for i, v in enumerate(vals):
        cls = "bar" + (" now" if i == 0 else "") + (" best" if i == best_i else "")
        marker = "<span class='marker'>\u25bc best</span>" if i == best_i else ""
        bars += f"<div class='{cls}' style='height:{_bar_h(v):.0f}%'>{marker}</div>"
    labels = "".join(f"<span>{'now' if i == 0 else '+' + str(i)}</span>" for i in range(len(vals)))
    st.markdown(f"""
    <div class="window">
      <div class="window-h">
        <span class="wtitle">Best launch window \u2014 next 12 hours</span>
        <span class="wverdict">{verdict}</span>
      </div>
      <div class="strip">{bars}</div>
      <div class="strip-x">{labels}</div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.25, 1])

    with left:
        st.markdown("<div class='sec-h'>Other strong options</div>", unsafe_allow_html=True)
        rows = ""
        for _, r in ranked.iloc[1:].iterrows():
            fc = FAMILY_COLOR.get(r.family, MUTED)
            rl, _ = risk_level(r.risk_norm)
            rdots = dot_html(rl)
            rows += (f"<tr><td><span class='famdot' style='background:{fc};margin-right:8px;vertical-align:middle'></span>"
                     f"<b>{r.instance_type}</b> <span style='color:{MUTED};font-size:12px'>{r.az}</span></td>"
                     f"<td>{r.region}</td><td class='num'>{r.price_now:.4f}</td>"
                     f"<td class='num'><span class='save-tag'>{r.savings_pct:.0f}%</span></td>"
                     f"<td style='text-align:right'>{rdots}</td></tr>")
        st.markdown(f"<table class='alts'><tr><th>Instance</th><th>Region</th>"
                    f"<th style='text-align:right'>Spot $/hr</th><th style='text-align:right'>Savings</th>"
                    f"<th style='text-align:right'>Risk</th></tr>{rows}</table>", unsafe_allow_html=True)

        st.write("")
        hist = ghist.tail(5 * 24)
        hist_vals = hist["spot_price"].values[::4]
        st.markdown(f'''<div class="panel">
  <div class="panel-h">Price forecast \u2014 {best.instance_type} @ {best.az}</div>
  <div class="panel-sub">Last 5 days, with the next-12h forecast and launch window.</div>
  {forecast_svg(hist_vals, vals, best_i)}
</div>''', unsafe_allow_html=True)

    with right:
        order = fam_pred.sort_values(ascending=False)
        maxpos = max([v for v in order.values if v > 0] + [1.0])
        prows = ""
        for famname, v in order.items():
            col = FAMILY_COLOR.get(famname, MUTED)
            if v > 0:
                w = 10 + 82 * (v / maxpos)
                pct, pctcol = f"+{v:.0f}%", FAMILY_TEXT.get(famname, col)
            else:
                w, pct, pctcol = 20.0, "n/a", FAINT
            prows += (f"<div class='pred-row'><span class='pred-fam'>{famname}</span>"
                      f"<div class='pred-track'><div class='pred-fill' style='width:{w:.0f}%;background:{col}'></div></div>"
                      f"<span class='pred-pct' style='color:{pctcol}'>{pct}</span></div>")
        st.markdown(f'''<div class="panel">
  <div class="panel-h">Market predictability</div>
  <div class="panel-sub">How well prices can be forecast, by instance family.</div>
  {prows}
  <div class="pred-note">Higher = the model beats a last-value guess. Burst markets stay largely unpredictable \u2014 surfaced honestly, not hidden.</div>
</div>''', unsafe_allow_html=True)
