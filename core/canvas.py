"""A tiny bidirectional drawing canvas built with Streamlit Components v2 (inline HTML/CSS/JS)."""
import streamlit as st

HTML = """
<div class="wrap">
  <canvas id="pad" width="280" height="280"></canvas>
  <div class="bar">
    <label>Brush <input id="size" type="range" min="10" max="40" value="22"></label>
    <button id="clear">🧹 Clear</button>
  </div>
</div>
"""
CSS = """
.wrap { display:flex; flex-direction:column; align-items:center; gap:10px; }
canvas { background:#000; border-radius:18px; cursor:crosshair; touch-action:none; box-shadow:0 0 0 3px var(--st-primary-color), 0 18px 40px -18px var(--st-primary-color); }
.bar { display:flex; gap:14px; align-items:center; font-family:sans-serif; color:var(--st-text-color); }
button { background:var(--st-primary-color); color:#fff; border:none; padding:8px 16px; border-radius:10px; font-weight:700; cursor:pointer; }
"""
JS = """
export default function(component) {
  const { parentElement, setStateValue } = component;
  const c = parentElement.querySelector('#pad');
  const ctx = c.getContext('2d');
  const size = parentElement.querySelector('#size');
  let drawing = false, last = null, timer = null;
  const pos = (e) => { const r = c.getBoundingClientRect(); const p = e.touches ? e.touches[0] : e; return [(p.clientX - r.left) * c.width / r.width, (p.clientY - r.top) * c.height / r.height]; };
  const send = () => { clearTimeout(timer); timer = setTimeout(() => setStateValue('image', c.toDataURL('image/png')), 150); };
  const start = (e) => { drawing = true; last = pos(e); e.preventDefault(); };
  const move = (e) => {
    if (!drawing) return; const p = pos(e);
    ctx.strokeStyle = '#fff'; ctx.lineWidth = +size.value; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.beginPath(); ctx.moveTo(...last); ctx.lineTo(...p); ctx.stroke(); last = p; e.preventDefault();
  };
  const end = () => { if (drawing) { drawing = false; send(); } };
  c.addEventListener('mousedown', start); c.addEventListener('mousemove', move); window.addEventListener('mouseup', end);
  c.addEventListener('touchstart', start); c.addEventListener('touchmove', move); c.addEventListener('touchend', end);
  parentElement.querySelector('#clear').onclick = () => { ctx.clearRect(0, 0, c.width, c.height); setStateValue('image', null); };
  return () => window.removeEventListener('mouseup', end);
}
"""

_component = None


def available():
    return hasattr(st, "components") and hasattr(st.components, "v2")


def draw_canvas(key="canvas"):
    """Returns a PNG data-URL of the drawing (or None)."""
    global _component
    if _component is None:
        _component = st.components.v2.component("digit_canvas", html=HTML, css=CSS, js=JS)
    res = _component(key=key, on_image_change=lambda: None)
    return getattr(res, "image", None)
