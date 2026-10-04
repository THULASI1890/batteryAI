import hashlib, hmac, json, os, random, subprocess, sys, time
from html import escape
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="BatteryAI", page_icon="🔋", layout="wide")

FEATURES = ["cycle", "ambient_temp", "voltage_mean", "voltage_min",
            "current_mean", "temp_mean", "temp_max", "duration"]
QUICK = ["cycle", "ambient_temp", "current_mean"]
APP_DIR = os.path.dirname(os.path.abspath(__file__))  # look next to app.py, not where the terminal is open
MODEL_FILES = {"soh": "soh_model.pkl", "rul": "rul_model.pkl", "soh_q": "soh_quick.pkl", "rul_q": "rul_quick.pkl"}
USERS_FILE = os.path.join(APP_DIR, "users.json")
# (cycles, ambient C, current A, duration s, mean V, min V, mean temp C, max temp C)
PRESETS = {"Fresh cell": (10, 24.0, 2.0, 3350, 3.56, 2.65, 31.0, 35.0),
           "Mid-life cell": (80, 24.0, 2.0, 3000, 3.50, 2.70, 33.0, 38.0),
           "Aged cell": (140, 24.0, 2.0, 2450, 3.42, 2.78, 37.0, 43.0)}

# ======================= Styling =======================
st.markdown("""
<style>
#MainMenu, footer, header {visibility:hidden}
.block-container {padding-top:1.2rem; max-width:1400px}

/* ---------- Background: glow blobs + circuit grid ---------- */
.stApp {
  background:
    radial-gradient(circle at 15% 10%, rgba(52,211,153,.18), transparent 40%),
    radial-gradient(circle at 85% 85%, rgba(59,130,246,.18), transparent 40%),
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='60' height='60'%3E%3Cpath d='M60 0H0v60' fill='none' stroke='%2334d399' stroke-opacity='.07'/%3E%3Ccircle cx='0' cy='0' r='1.6' fill='%2334d399' fill-opacity='.25'/%3E%3C/svg%3E"),
    #0b1220;
  background-attachment: fixed;
  animation: drift 18s ease-in-out infinite alternate;
}
@keyframes drift {from {background-position: 0 0, 0 0, 0 0, 0 0} to {background-position: 40px 30px, -40px -30px, 0 0, 0 0}}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #0f172a, #0b1220); border-right: 1px solid #1f2937}

/* ---------- Hero ---------- */
.hero h1 {font-size:3rem; line-height:1.1; margin:0 0 .6rem 0}
.hero h1 span {background:linear-gradient(90deg,#34d399,#60a5fa); -webkit-background-clip:text; color:transparent}
.hero p {color:#9ca3af; font-size:1.1rem}
.pill {display:inline-block; background:rgba(6,78,59,.7); color:#a7f3d0; border-radius:999px;
       padding:4px 12px; font-size:.8rem; margin:0 6px 6px 0}
.battery {position:relative; width:120px; height:56px; border:3px solid #34d399; border-radius:10px;
          padding:4px; margin-bottom:22px; box-shadow:0 0 28px rgba(52,211,153,.4)}
.battery::after {content:""; position:absolute; right:-12px; top:16px; width:8px; height:20px;
                 background:#34d399; border-radius:0 4px 4px 0}
.battery .level {height:100%; border-radius:4px; background:linear-gradient(90deg,#10b981,#34d399);
                 animation:charge 3s ease-in-out infinite}
@keyframes charge {0% {width:12%} 70%, 100% {width:100%}}

/* ---------- Cards (glass) ---------- */
.card {background:rgba(17,24,39,.65); backdrop-filter:blur(8px); border:1px solid #1f2937;
       border-radius:14px; padding:18px; text-align:center; transition:transform .2s, border-color .2s}
.card:hover {transform:translateY(-4px); border-color:#34d399}
.card h4 {color:#9ca3af; font-size:.85rem; font-weight:500; margin:0 0 6px 0}
.card h1 {color:#34d399; font-size:2rem; margin:0}
.card small {color:#6b7280}
.verdict {border-radius:12px; padding:14px 18px; font-weight:600; margin:14px 0}

/* ---------- Big login ---------- */
.topbar {display:flex; justify-content:space-between; align-items:center; padding:6px 4px 26px}
.brand {font-size:1.5rem; font-weight:800; letter-spacing:.5px} .brand span {color:#34d399}
.big-hero h1 {font-size:4.3rem; line-height:1.02; font-weight:800; margin:.6rem 0 1.1rem}
.big-hero h1 span {background:linear-gradient(90deg,#34d399,#60a5fa); -webkit-background-clip:text; color:transparent}
.big-hero p {font-size:1.25rem; color:#9ca3af; max-width:580px}
.stats {display:flex; gap:34px; margin-top:30px}
.stats b {display:block; font-size:2rem; color:#34d399} .stats small {color:#6b7280}
.hbat {position:relative; display:flex; gap:10px; width:360px; height:150px; padding:14px; margin-bottom:26px;
       border:5px solid #34d399; border-radius:24px; box-shadow:0 0 70px rgba(52,211,153,.35)}
.hbat::after {content:""; position:absolute; right:-20px; top:44px; width:14px; height:50px; background:#34d399; border-radius:0 8px 8px 0}
.hbat i {flex:1; border-radius:9px; background:#10b981; opacity:.15; animation:cell 3.5s infinite}
.hbat i:nth-child(2) {animation-delay:.3s} .hbat i:nth-child(3) {animation-delay:.6s}
.hbat i:nth-child(4) {animation-delay:.9s} .hbat i:nth-child(5) {animation-delay:1.2s}
.hbat b {position:absolute; inset:0; display:grid; place-items:center; font-size:3.6rem; filter:drop-shadow(0 0 12px #fff)}
@keyframes cell {0%,10% {opacity:.15} 30%,85% {opacity:1} 100% {opacity:.15}}
[data-testid="stVerticalBlockBorderWrapper"]:has(.login-title) {background:rgba(17,24,39,.78); backdrop-filter:blur(14px);
   border:1px solid rgba(52,211,153,.4); border-radius:24px; padding:30px 26px; box-shadow:0 0 70px rgba(52,211,153,.15)}
.login-title h2 {margin:0; font-size:2rem} .login-title p {color:#9ca3af; margin:4px 0 14px}
.stTextInput input {height:3.1rem; font-size:1.05rem}
[data-testid^="stBaseButton"] {min-height:3rem; font-weight:700}

/* ---------- Dashboard ---------- */
.banner {border-radius:20px; padding:26px 32px; margin-bottom:18px; border:1px solid #1f2937;
         background:linear-gradient(120deg,rgba(6,78,59,.85),rgba(30,58,138,.6))}
.banner h1 {margin:0; font-size:2.1rem} .banner p {margin:4px 0 0; color:#a7f3d0}
.sec {font-size:1.15rem; font-weight:700; margin:26px 0 10px}
.bigbat {position:relative; display:flex; gap:6px; height:92px; padding:8px; margin:6px 16px 10px 0;
         border:4px solid var(--c); border-radius:16px}
.bigbat::after {content:""; position:absolute; right:-14px; top:26px; width:10px; height:28px; background:var(--c); border-radius:0 6px 6px 0}
.bigbat i {flex:1; border-radius:5px; background:rgba(255,255,255,.08)}
.bigbat i.on {background:var(--c); box-shadow:0 0 14px var(--c)}
.lifewrap {position:relative; margin:22px 0 8px}
.life {display:flex; height:16px; border-radius:99px; overflow:hidden} .life span {flex:1}
.mark {position:absolute; top:-8px; width:4px; height:32px; background:#fff; border-radius:2px; transform:translateX(-50%); box-shadow:0 0 12px #fff}
.lifelabels {display:flex; justify-content:space-between; color:#9ca3af; font-size:.8rem}
.routes {display:grid; grid-template-columns:repeat(3,1fr); gap:14px}
.route {border:1px solid #1f2937; border-radius:16px; padding:18px; background:rgba(17,24,39,.5); opacity:.4}
.route.active {opacity:1; border-color:#34d399; box-shadow:0 0 30px rgba(52,211,153,.25)}
.route h3 {margin:0 0 4px} .route p {margin:0; color:#9ca3af; font-size:.9rem}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"][aria-expanded="true"] {min-width:310px; max-width:310px}
[data-testid="stSidebar"] {background: radial-gradient(circle at 50% 100%, rgba(52,211,153,.2), transparent 55%),
                           linear-gradient(180deg,#0f172a,#0b1220); border-right:1px solid #1f2937}
.sb-brand {display:flex; align-items:center; gap:12px}
.sb-brand b {font-size:1.45rem} .sb-brand b span {color:#34d399}
.sb-logo {width:48px; height:48px; border-radius:14px; display:grid; place-items:center; font-size:1.5rem;
          background:linear-gradient(135deg,#10b981,#3b82f6); animation:pulse 2.6s ease-in-out infinite}
@keyframes pulse {0%,100% {box-shadow:0 0 18px rgba(52,211,153,.4)} 50% {box-shadow:0 0 40px rgba(52,211,153,.85)}}
[data-testid="stSidebar"] [role="radiogroup"] {gap:6px}
[data-testid="stSidebar"] [role="radiogroup"] label {width:100%; padding:12px 14px; border-radius:12px;
                                                   border:1px solid transparent; transition:.2s}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {background:rgba(52,211,153,.08)}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {border-color:rgba(52,211,153,.55);
   background:linear-gradient(90deg,rgba(16,185,129,.3),rgba(59,130,246,.18))}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {display:none}
[data-testid="stSidebar"] [role="radiogroup"] label p {font-size:1.05rem; font-weight:600}
.sb-panel {background:rgba(17,24,39,.65); border:1px solid #1f2937; border-radius:16px; padding:14px 16px; margin-top:14px}
.sb-panel h5 {margin:0 0 10px; color:#9ca3af; font-size:.72rem; letter-spacing:1.2px; text-transform:uppercase}
.sb-row {display:flex; justify-content:space-between; margin-top:6px} .sb-row span {color:#9ca3af}
.sb-user {display:flex; align-items:center; gap:12px; margin:22px 0 10px}
.sb-avatar {width:44px; height:44px; border-radius:50%; display:grid; place-items:center; font-weight:800;
            font-size:1.1rem; color:#06281f; background:linear-gradient(135deg,#34d399,#60a5fa)}
.side .bigbat {height:40px; padding:5px; gap:4px; border-width:3px; margin:2px 12px 10px 0}
.side .bigbat::after {top:9px; height:14px; width:7px; right:-10px}
.tip {font-size:.9rem; color:#d1d5db; line-height:1.55}
iframe[height="1"] {position:absolute; opacity:0; pointer-events:none}
</style>
""", unsafe_allow_html=True)

# ---------- JavaScript: floating particles + count-up numbers ----------
# st.markdown cannot run scripts, so a tiny component injects them into the page.
EFFECTS_JS = r"""
<script>
function parentCode() {
  if (window.__bai) return;
  window.__bai = true;

  // 1) Floating particle network behind the app
  const c = document.createElement('canvas');
  c.style.cssText = 'position:fixed;inset:0;width:100%;height:100%;pointer-events:none;z-index:0;opacity:.5';
  document.body.appendChild(c);
  const ctx = c.getContext('2d');
  let w, h;
  const resize = () => { w = c.width = window.innerWidth; h = c.height = window.innerHeight; };
  resize();
  window.addEventListener('resize', resize);
  const pts = Array.from({length: 55}, () => ({
    x: Math.random() * w, y: Math.random() * h,
    vx: (Math.random() - .5) * .35, vy: (Math.random() - .5) * .35}));
  (function loop() {
    ctx.clearRect(0, 0, w, h);
    pts.forEach((p, i) => {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0 || p.x > w) p.vx *= -1;
      if (p.y < 0 || p.y > h) p.vy *= -1;
      ctx.fillStyle = 'rgba(52,211,153,.7)';
      ctx.beginPath(); ctx.arc(p.x, p.y, 1.6, 0, 6.283); ctx.fill();
      for (let j = i + 1; j < pts.length; j++) {
        const q = pts[j], d = Math.hypot(p.x - q.x, p.y - q.y);
        if (d < 130) {
          ctx.strokeStyle = 'rgba(52,211,153,' + (0.18 * (1 - d / 130)) + ')';
          ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke();
        }
      }
    });
    requestAnimationFrame(loop);
  })();

  // 2) Count-up animation for the metric cards
  window.baiCount = function () {
    document.querySelectorAll('.card h1').forEach(el => {
      if (el.dataset.done) return;
      el.dataset.done = '1';
      const text = el.textContent, m = text.match(/[\d.]+/);
      if (!m) return;
      const target = parseFloat(m[0]), dec = (m[0].split('.')[1] || '').length, t0 = performance.now();
      (function step(now) {
        const k = Math.min((now - t0) / 900, 1), eased = 1 - Math.pow(1 - k, 3);
        el.textContent = text.replace(m[0], (target * eased).toFixed(dec));
        if (k < 1) requestAnimationFrame(step);
      })(t0);
    });
  };
}
const d = window.parent.document;
if (!window.parent.__bai) {
  const s = d.createElement('script');
  s.textContent = '(' + parentCode.toString() + ')()';
  d.body.appendChild(s);
}
setTimeout(() => window.parent.baiCount && window.parent.baiCount(), 200);
</script>
"""


def effects():
    html = EFFECTS_JS + f"<!-- {time.time()} -->"
    if hasattr(st, "iframe"):          # newer Streamlit
        st.iframe(html, height=1)
    else:                              # older Streamlit
        import streamlit.components.v1 as components
        components.html(html, height=1)


# ======================= Authentication =======================
def hash_pw(password, salt=None):
    salt = salt or os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 100_000).hex()
    return salt, digest


def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    salt, digest = hash_pw("demo123")  # demo account so visitors can try the app
    users = {"demo": {"salt": salt, "hash": digest}}
    save_users(users)
    return users


def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)


def check_login(username, password):
    user = load_users().get(username.strip().lower())
    if not user:
        return False
    _, digest = hash_pw(password, user["salt"])
    return hmac.compare_digest(digest, user["hash"])


def create_user(username, password):
    username = username.strip().lower()
    users = load_users()
    if len(username) < 3 or len(password) < 6:
        return "Username needs 3+ characters and password 6+."
    if username in users:
        return "That username is taken."
    salt, digest = hash_pw(password)
    users[username] = {"salt": salt, "hash": digest}
    save_users(users)
    return None


def login_page():
    st.markdown('<div class="topbar"><div class="brand">🔋 Battery<span>AI</span></div>'
                '<div style="color:#6b7280">Second-life battery intelligence</div></div>', unsafe_allow_html=True)
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.markdown("""
        <div class="big-hero">
          <div class="hbat"><i></i><i></i><i></i><i></i><i></i><b>⚡</b></div>
          <h1>Give every battery a <span>second life</span>.</h1>
          <p>Predict the State of Health, Remaining Useful Life and reuse potential of used
          lithium-ion batteries, using machine learning trained on NASA aging data.</p>
          <div class="stats">
            <div><b>SOH</b><small>health today</small></div>
            <div><b>RUL</b><small>cycles left</small></div>
            <div><b>Score</b><small>reuse potential</small></div>
          </div>
        </div>""", unsafe_allow_html=True)
    with right:
        with st.container(border=True):
            st.markdown('<div class="login-title"><h2>Welcome back 👋</h2>'
                        '<p>Log in to open your battery console.</p></div>', unsafe_allow_html=True)
            tab_in, tab_up = st.tabs(["Log in", "Sign up"])
            with tab_in:
                with st.form("login"):
                    u = st.text_input("Username")
                    p = st.text_input("Password", type="password")
                    if st.form_submit_button("Log in", type="primary", width="stretch"):
                        if check_login(u, p):
                            st.session_state.user = u.strip().lower()
                            st.rerun()
                        st.error("Wrong username or password.")
                st.caption("Try the demo account: **demo** / **demo123**")
            with tab_up:
                with st.form("signup"):
                    u = st.text_input("Choose a username")
                    p = st.text_input("Choose a password", type="password")
                    if st.form_submit_button("Create account", width="stretch"):
                        error = create_user(u, p)
                        if error:
                            st.error(error)
                        else:
                            st.success("Account created. Switch to the Log in tab.")


# ======================= Models & prediction =======================
@st.cache_resource
def load_models():
    return {key: joblib.load(os.path.join(APP_DIR, name)) for key, name in MODEL_FILES.items()}


def second_life_score(soh, rul, temp):
    score = 0.6 * soh + 0.3 * min(rul / 100, 1) * 100 + 0.1 * max(0, 100 - 4 * max(temp - 30, 0))
    return float(np.clip(score, 0, 100))


def health_color(soh):
    return "#34d399" if soh >= 80 else "#f59e0b" if soh >= 60 else "#ef4444"


def cells_html(soh):
    return "".join(f'<i class="{"on" if k < round(soh / 10) else ""}"></i>' for k in range(10))


def card(title, value, sub=""):
    return f'<div class="card"><h4>{title}</h4><h1>{value}</h1><small>{sub}</small></div>'


def show_result(r):
    soh, rul, score = r["SOH"], r["RUL"], r["Score"]
    col, cells = health_color(soh), cells_html(soh)

    left, right = st.columns([1.1, 2], gap="large")
    left.markdown(f'<div class="sec">Charge state</div><div class="bigbat" style="--c:{col}">{cells}</div>'
                  f'<b style="color:{col}">{soh:.0f}% of original capacity</b>', unsafe_allow_html=True)
    with right:
        c1, c2, c3 = st.columns(3)
        c1.markdown(card("⚡ State of Health", f"{soh:.1f}%", "capacity retained"), unsafe_allow_html=True)
        c2.markdown(card("⏳ Remaining Life", f"{rul:.0f}", "cycles to 70% SOH"), unsafe_allow_html=True)
        c3.markdown(card("♻️ Second-Life Score", f"{score:.0f}/100", "reuse potential"), unsafe_allow_html=True)

    pos = float(np.clip((100 - soh) / 60, 0, 1)) * 100
    st.markdown(f"""<div class="sec">Lifecycle position</div>
      <div class="lifewrap"><div class="life"><span style="background:#10b981"></span>
      <span style="background:#f59e0b"></span><span style="background:#ef4444"></span></div>
      <div class="mark" style="left:{pos}%"></div></div>
      <div class="lifelabels"><span>EV use · 100-80%</span><span>Second life · 80-60%</span>
      <span>Recycle · below 60%</span></div>""", unsafe_allow_html=True)

    active = 0 if soh >= 80 else 1 if soh >= 60 else 2
    routes = [("🚗", "Primary EV use", "Strong enough to keep powering vehicles."),
              ("🏠", "Home & solar storage", "Ideal for stationary backup, e-bikes and light loads."),
              ("♻️", "Recycle", "Recover materials safely and responsibly.")]
    tiles = "".join(f'<div class="route {"active" if i == active else ""}"><h3>{ic} {t}</h3><p>{d}</p></div>'
                    for i, (ic, t, d) in enumerate(routes))
    st.markdown(f'<div class="sec">Recommended next life</div><div class="routes">{tiles}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec">Health & forecast</div>', unsafe_allow_html=True)
    g, d = st.columns([1, 2])
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=soh, number={"suffix": "%"}, title={"text": "Battery Health"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": col},
               "steps": [{"range": [0, 60], "color": "#7f1d1d"},
                         {"range": [60, 80], "color": "#78350f"},
                         {"range": [80, 100], "color": "#064e3b"}]}))
    gauge.update_layout(height=320, margin=dict(t=60, b=10, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)")
    g.plotly_chart(gauge, width="stretch")

    horizon = int(max(rul, 10)) + 20
    future = np.arange(horizon + 1)
    rate = max(soh - 70, 0) / max(rul, 1)
    fig = go.Figure(go.Scatter(x=r["cycle"] + future, y=np.clip(soh - rate * future, 0, 100), mode="lines",
                               fill="tozeroy", fillcolor="rgba(52,211,153,.12)",
                               line=dict(color="#34d399", width=3), name="Projected SOH"))
    fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", annotation_text="End of life (70%)")
    fig.add_hline(y=80, line_dash="dot", line_color="#f59e0b", annotation_text="EV threshold (80%)")
    fig.update_layout(title="Future degradation (linear projection)", xaxis_title="Cycle", yaxis_title="SOH (%)",
                      height=320, margin=dict(t=60, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    d.plotly_chart(fig, width="stretch")
    effects()


def train_quick(csv_path):
    """Train the Quick-mode models (cycles, ambient temp, current) directly from the CSV."""
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    df = pd.read_csv(csv_path)
    soh = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1).fit(df[QUICK], df["SOH"])
    rul = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.9,
                       random_state=42).fit(df[QUICK], df["RUL"])
    joblib.dump(soh, os.path.join(APP_DIR, MODEL_FILES["soh_q"]))
    joblib.dump(rul, os.path.join(APP_DIR, MODEL_FILES["rul_q"]))


def train_now():
    """Build whatever is missing (data, main models, quick models). Returns an error text or None."""
    csv = os.path.join(APP_DIR, "nasa_battery_data.csv")
    have = lambda *keys: all(os.path.exists(os.path.join(APP_DIR, MODEL_FILES[k])) for k in keys)
    steps = []
    if not os.path.exists(csv):
        steps.append("make_demo_data.py")  # synthetic data, used only when no real CSV exists
    if not have("soh", "rul"):
        steps.append("train_models.py")
    try:
        for script in steps:
            path = os.path.join(APP_DIR, script)
            if not os.path.exists(path):
                return f"{script} was not found in {APP_DIR}. Put it next to app.py."
            r = subprocess.run([sys.executable, path], cwd=APP_DIR, capture_output=True, text=True)
            if r.returncode != 0:
                return (r.stderr or r.stdout)[-900:]
        if not have("soh_q", "rul_q"):
            train_quick(csv)
    except Exception as e:  # show the real reason on the page
        return f"{type(e).__name__}: {e}"
    return None


# ======================= Pages =======================
def dashboard_page():
    st.markdown(f'<div class="banner"><h1>🔋 Battery Health Console</h1>'
                f'<p>Hello {escape(st.session_state.user)}. Load an example or enter measurements, then run the analysis.</p></div>',
                unsafe_allow_html=True)
    missing = [n for n in MODEL_FILES.values() if not os.path.exists(os.path.join(APP_DIR, n))]
    if missing:
        st.error(f"Missing model files: {', '.join(missing)}\n\nLooked in: {APP_DIR}")
        if st.button("🛠 Train models now", type="primary"):
            with st.spinner("Training models... this takes under a minute."):
                problem = train_now()
            if problem:
                st.error("Training failed:")
                st.code(problem)
            else:
                st.rerun()
        st.caption("Uses nasa_battery_data.csv if present. Otherwise it creates synthetic demo data first.")
        return
    models = load_models()

    mode = st.radio("Analysis mode", ["Quick (cycles only)", "Detailed (sensor readings)"], horizontal=True)
    detailed = mode.startswith("Detailed")
    preset = st.selectbox("Quick-load an example battery", list(PRESETS))
    cy, am, cu, du, vm, vn, tm, tx = PRESETS[preset]
    with st.form("predict"):
        a, b, c, d = st.columns(4)
        cycle = a.number_input("Cycles completed", 1, 200, cy, key=f"cy{preset}")
        ambient = b.number_input("Ambient temp (°C)", 0.0, 60.0, am, key=f"am{preset}")
        i_mean = c.number_input("Mean current (A)", 0.1, 4.0, cu, 0.1, key=f"cu{preset}")
        if detailed:
            duration = d.number_input("Discharge time (s)", 500, 5000, du, 50, key=f"du{preset}")
            e, f, g, h = st.columns(4)
            v_mean = e.number_input("Mean voltage (V)", 3.0, 4.0, vm, 0.01, key=f"vm{preset}")
            v_min = f.number_input("Min voltage (V)", 2.0, 3.2, vn, 0.01, key=f"vn{preset}")
            t_mean = g.number_input("Mean cell temp (°C)", 15.0, 60.0, tm, key=f"tm{preset}")
            t_max = h.number_input("Max cell temp (°C)", 15.0, 70.0, tx, key=f"tx{preset}")
        else:
            d.caption("Quick mode estimates health from the cycle count alone, so changing cycles directly moves the result.")
        run = st.form_submit_button("⚡ Analyse battery", type="primary", width="stretch")

    if run:
        if detailed:
            x = pd.DataFrame([[cycle, ambient, v_mean, v_min, i_mean, t_mean, t_max, duration]], columns=FEATURES)
            soh_m, rul_m, temp = models["soh"], models["rul"], t_max
        else:
            x = pd.DataFrame([[cycle, ambient, i_mean]], columns=QUICK)
            soh_m, rul_m, temp = models["soh_q"], models["rul_q"], ambient
        soh = float(np.clip(soh_m.predict(x)[0], 0, 100))
        rul = float(max(rul_m.predict(x)[0], 0))
        result = {"cycle": cycle, "SOH": soh, "RUL": rul, "Score": second_life_score(soh, rul, temp)}
        st.session_state.last = result
        st.session_state.setdefault("history", []).append(
            {"Time": datetime.now().strftime("%H:%M:%S"), "Mode": "Detailed" if detailed else "Quick",
             **{k: round(v, 1) for k, v in result.items()}})
        st.rerun()  # refresh so the sidebar shows the newest result

    if "last" in st.session_state:
        show_result(st.session_state.last)
    else:
        st.info("No analysis yet. Pick an example above and click **Analyse battery**.")


def history_page():
    st.title("History")
    history = st.session_state.get("history", [])
    if not history:
        st.info("Your predictions from this session will appear here.")
        return
    df = pd.DataFrame(history)
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button("Download CSV", df.to_csv(index=False), "batteryai_history.csv", "text/csv")


def about_page():
    st.title("About BatteryAI")
    st.markdown("""
**What it does:** estimates a used lithium-ion battery's State of Health (SOH), Remaining Useful Life (RUL)
and a second-life suitability score.

**How:** a Random Forest predicts SOH and XGBoost predicts RUL, trained on the NASA PCoE Li-ion Battery Aging Dataset.

**Score guide:** SOH ≥ 80% suits primary EV use, 60-80% suits second-life storage, below 60% suggests recycling.

**Limits:** predictions come from single measurements, so treat RUL as indicative, not a guarantee.
Accounts here are for demonstration only and are not production-grade security.
""")


# ======================= Sidebar =======================
NAV = {"Dashboard": "🏠", "History": "🕘", "About": "ℹ️"}
TIPS = ["Lithium-ion cells are usually retired from vehicles at around 70-80% of their original capacity.",
        "Heat is one of the biggest drivers of battery ageing. Keeping cells cool helps them last longer.",
        "Retired EV batteries are often reused for home and grid storage, where weight matters less.",
        "Frequent deep discharges and fast charging can speed up capacity loss.",
        "Recycling recovers valuable metals such as lithium, nickel and cobalt."]


def sidebar():
    user = st.session_state.user
    history = st.session_state.get("history", [])
    last = st.session_state.get("last")
    tip = TIPS[st.session_state.setdefault("tip", random.randrange(len(TIPS)))]
    with st.sidebar:
        st.markdown('<div class="sb-brand"><div class="sb-logo">🔋</div><div><b>Battery<span>AI</span></b><br>'
                    '<small style="color:#6b7280">Second-life intelligence</small></div></div>', unsafe_allow_html=True)
        st.write("")
        page = st.radio("Navigate", list(NAV), format_func=lambda k: f"{NAV[k]}  {k}", label_visibility="collapsed")

        if last:
            body = (f'<div class="side"><div class="bigbat" style="--c:{health_color(last["SOH"])}">{cells_html(last["SOH"])}</div></div>'
                    f'<div class="sb-row"><span>Health</span><b>{last["SOH"]:.0f}%</b></div>'
                    f'<div class="sb-row"><span>Life left</span><b>{last["RUL"]:.0f} cycles</b></div>'
                    f'<div class="sb-row"><span>Reuse score</span><b>{last["Score"]:.0f}/100</b></div>')
        else:
            body = ('<div class="side"><div class="bigbat" style="--c:#374151">' + cells_html(0) + '</div></div>'
                    '<div class="tip">Run an analysis to see your battery here.</div>')
        st.markdown(f'<div class="sb-panel"><h5>Latest battery</h5>{body}</div>', unsafe_allow_html=True)

        avg = f'{np.mean([h["SOH"] for h in history]):.0f}%' if history else "-"
        st.markdown(f'<div class="sb-panel"><h5>This session</h5>'
                    f'<div class="sb-row"><span>Analyses run</span><b>{len(history)}</b></div>'
                    f'<div class="sb-row"><span>Average health</span><b>{avg}</b></div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sb-panel"><h5>💡 Did you know?</h5><div class="tip">{tip}</div></div>', unsafe_allow_html=True)

        st.markdown(f'<div class="sb-user"><div class="sb-avatar">{escape(user[:1].upper())}</div>'
                    f'<div><b>{escape(user)}</b><br><small style="color:#6b7280">Signed in</small></div></div>', unsafe_allow_html=True)
        if st.button("Log out", width="stretch"):
            st.session_state.clear()
            st.rerun()
    return page


# ======================= Router =======================
if "user" not in st.session_state:
    login_page()
    effects()
    st.stop()

page = sidebar()

{"Dashboard": dashboard_page, "History": history_page, "About": about_page}[page]()