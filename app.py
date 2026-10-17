"""DigitVision — handwritten digit recognition with a PyTorch CNN (Streamlit)."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from core import auth, ui
from core.canvas import available as canvas_available, draw_canvas
from core.engine import (SIZE, baselines, confusion, decode_data_url, embedding_2d, feature_maps, load_data, predict, preprocess, train_cnn)

st.set_page_config(page_title="DigitVision · CNN Digit Recognition", page_icon="✍️", layout="wide")
ui.inject_css()
APP, ICON = "DigitVision", "✍️"
if not auth.auth_page(APP, "Draw a digit and watch a convolutional neural network read your handwriting in real time.",
                      ["🧠 3-block PyTorch CNN with BatchNorm, Dropout & augmentation", "✍️ Live drawing pad + photo upload",
                       "🏋️ Interactive training studio with live curves", "🔬 Feature maps, embeddings & confusion matrix"], ICON,
                      demo={"name": "Vision Explorer", "email": "demo@digitvision.ai", "password": "Demo@1234"}):
    st.stop()


@st.cache_data
def data():
    return load_data()


@st.cache_resource(show_spinner="Training the CNN (≈20 s, once)…")
def base_model():
    Xtr, Xte, ytr, yte = data()
    return train_cnn(Xtr, ytr, Xte, yte, epochs=25)


Xtr, Xte, ytr, yte = data()
model, hist = st.session_state.get("custom_model") or base_model()
page = ui.sidebar(APP, ICON, ["✍️ Draw & Predict", "🖼️ Upload Photo", "🧪 Test Gallery", "🏋️ Training Studio", "🔬 Model Insights", "👤 Account"])
st.session_state.setdefault("feedback", [])


def show_prediction(img16, source):
    probs = predict(model, img16)[0]
    order = probs.argsort()[::-1]
    top, conf = int(order[0]), float(probs[order[0]])
    c1, c2 = st.columns([1, 1.6])
    with c1:
        st.markdown(f"""<div class='card' style='text-align:center'><div style='color:var(--muted);font-weight:700'>PREDICTION</div>
            <div style='font-size:6rem;font-weight:800;line-height:1;background:linear-gradient(135deg,var(--a),var(--b));-webkit-background-clip:text;color:transparent;animation:pageIn .5s'>{top}</div>
            <div class='badge {"green" if conf > .8 else "amber" if conf > .5 else "red"}'>{conf * 100:.1f}% confident</div>
            <p style='margin-top:8px'>Runner-up: <b>{order[1]}</b> ({probs[order[1]] * 100:.1f}%) · <b>{order[2]}</b> ({probs[order[2]] * 100:.1f}%)</p></div>""", unsafe_allow_html=True)
        st.image(Image.fromarray((img16 * 255).astype(np.uint8)).resize((160, 160), Image.NEAREST), caption="What the CNN sees (16×16)")
    with c2:
        fig = go.Figure(go.Bar(x=list(range(10)), y=probs, marker_color=["#10b981" if i == top else "rgba(148,163,184,.45)" for i in range(10)], text=[f"{p * 100:.0f}%" for p in probs], textposition="outside"))
        fig.update_layout(title="Class probabilities (softmax)", xaxis=dict(tickmode="linear"), yaxis_range=[0, 1.1], height=330)
        st.plotly_chart(fig, width="stretch")
        f1, f2, f3 = st.columns(3)
        if f1.button("✅ Correct", width="stretch"):
            st.session_state.feedback.append({"pred": top, "true": top, "conf": conf, "src": source}); auth.log("feedback", f"correct {top}"); st.toast("Thanks for the feedback!")
        true = f2.selectbox("Actual digit", list(range(10)), index=top, label_visibility="collapsed")
        if f3.button("❌ Fix label", width="stretch"):
            st.session_state.feedback.append({"pred": top, "true": true, "conf": conf, "src": source}); auth.log("feedback", f"pred {top} → true {true}"); st.toast("Correction saved")
    return top, conf


if page == "✍️ Draw & Predict":
    ui.hero("Draw & Predict", "Draw any digit 0–9 in the black box — the CNN predicts as soon as you lift the pen", "✍️")
    l, r = st.columns([1, 1.8], gap="large")
    with l:
        url = draw_canvas() if canvas_available() else None
        if not canvas_available():
            st.warning("Your Streamlit version lacks Components v2 — upgrade to ≥1.51 for the drawing pad. Use **Upload Photo** meanwhile.")
        st.caption("Tip: draw big and centred, like on paper.")
    with r:
        if url:
            img, _ = preprocess(decode_data_url(url), invert=False)
            if img is None:
                st.info("Canvas is empty — draw a digit!")
            else:
                top, conf = show_prediction(img, "canvas")
                auth.log("predict", f"canvas → {top} ({conf:.0%})")
        else:
            st.markdown("<div class='card' style='text-align:center;padding:60px'><div style='font-size:4rem;animation:float 3s ease-in-out infinite'>✍️</div><h3>Waiting for your handwriting…</h3><p style='color:var(--muted)'>Your prediction, probabilities and the model’s-eye view appear here.</p></div>", unsafe_allow_html=True)
    fb = st.session_state.feedback
    if fb:
        acc = np.mean([f["pred"] == f["true"] for f in fb])
        k = st.columns(3)
        ui.kpi(k[0], "Your predictions rated", len(fb), "🗳️")
        ui.kpi(k[1], "Live accuracy on your handwriting", f"{acc * 100:.0f}%", "🎯")
        ui.kpi(k[2], "Avg confidence", f"{np.mean([f['conf'] for f in fb]) * 100:.0f}%", "📈")

elif page == "🖼️ Upload Photo":
    ui.hero("Upload a Photo", "Snap a handwritten digit on paper — see every preprocessing step", "🖼️")
    up = st.file_uploader("Image of a single handwritten digit", type=["png", "jpg", "jpeg", "webp"])
    inv = st.radio("Ink colour", ["Auto-detect", "Dark ink on light paper", "Light ink on dark background"], horizontal=True)
    if up:
        img, steps = preprocess(Image.open(up), invert={"Auto-detect": "auto", "Dark ink on light paper": True, "Light ink on dark background": False}[inv])
        cols = st.columns(len(steps) + 1)
        cols[0].image(Image.open(up), caption="original", width="stretch")
        for c, (name, im) in zip(cols[1:], steps.items()):
            c.image(im, caption=name, width="stretch")
        if img is not None:
            top, _ = show_prediction(img, "upload")
            auth.log("predict", f"upload → {top}")
        else:
            st.error("No digit found in the image.")
    else:
        st.info("Upload an image to begin. Clear, high-contrast photos work best.")

elif page == "🧪 Test Gallery":
    ui.hero("Test Gallery", f"Predictions on {len(Xte)} unseen test digits", "🧪")
    cm, pred = confusion(model, Xte, yte)
    probs = predict(model, Xte)
    c = st.columns([1, 1, 1, 2])
    only_wrong = c[0].toggle("Only mistakes")
    digit = c[1].selectbox("Digit", ["All"] + list(range(10)))
    n = c[2].slider("Samples", 12, 60, 24, 6)
    c[3].metric("Test accuracy", f"{(pred == yte).mean() * 100:.2f}%", f"{(pred != yte).sum()} errors")
    idx = np.arange(len(Xte))
    if only_wrong: idx = idx[pred != yte]
    if digit != "All": idx = idx[yte[idx] == digit]
    if not len(idx):
        st.success("No samples match — the model got them all right! 🎉")
    rng = np.random.default_rng(st.session_state.get("seed", 0))
    if st.button("🔀 Shuffle"):
        st.session_state.seed = int(rng.integers(1e6)); st.rerun()
    pick = rng.permutation(idx)[:n]
    cols = st.columns(6)
    for i, j in enumerate(pick):
        ok = pred[j] == yte[j]
        with cols[i % 6]:
            st.image(Image.fromarray((Xte[j] * 255).astype(np.uint8)).resize((96, 96), Image.NEAREST), width="stretch")
            st.markdown(f"<div style='text-align:center;margin-top:-8px'><span class='badge {'green' if ok else 'red'}'>pred {pred[j]} · true {yte[j]}</span><br><small style='color:var(--muted)'>{probs[j].max() * 100:.0f}%</small></div>", unsafe_allow_html=True)

elif page == "🏋️ Training Studio":
    ui.hero("Training Studio", "Tune hyper-parameters and train your own CNN with live learning curves", "🏋️")
    with st.form("hp"):
        c = st.columns(3)
        epochs = c[0].slider("Epochs", 5, 60, 20)
        lr = c[1].select_slider("Max learning rate", [5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2], 2e-3)
        batch = c[2].select_slider("Batch size", [16, 32, 64, 128], 64)
        c = st.columns(3)
        dropout = c[0].slider("Dropout", 0.0, 0.7, 0.3, 0.05)
        aug = c[1].slider("Augmentation strength", 0.0, 2.0, 1.0, 0.1)
        width = c[2].select_slider("Network width (filters)", [8, 16, 32, 48], 32)
        go_ = st.form_submit_button("🚀 Train model", type="primary", width="stretch")
    if go_:
        prog, chart, stat = st.progress(0.0), st.empty(), st.empty()

        def cb(h):
            prog.progress(len(h) / epochs, f"Epoch {len(h)}/{epochs}")
            df = pd.DataFrame(h)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.epoch, y=df.train_loss, name="train loss", line=dict(width=3)))
            fig.add_trace(go.Scatter(x=df.epoch, y=df.val_loss, name="val loss", line=dict(width=3)))
            fig.add_trace(go.Scatter(x=df.epoch, y=df.val_acc, name="val accuracy", yaxis="y2", line=dict(width=3, dash="dot")))
            fig.update_layout(height=360, yaxis2=dict(overlaying="y", side="right", range=[0, 1]), title="Live learning curves", legend=dict(orientation="h"))
            chart.plotly_chart(fig, width="stretch")
            stat.markdown(f"<div class='card'>📉 val loss <b>{h[-1]['val_loss']:.4f}</b> · 🎯 val accuracy <b>{h[-1]['val_acc'] * 100:.2f}%</b> · 🏋️ train accuracy <b>{h[-1]['train_acc'] * 100:.2f}%</b></div>", unsafe_allow_html=True)

        m, h = train_cnn(Xtr, ytr, Xte, yte, epochs, lr, batch, dropout, aug, width, callback=cb)
        st.session_state.custom_model = (m, h)
        auth.log("train", f"{epochs} ep, lr {lr}, width {width} → {h[-1]['val_acc']:.3f}")
        st.success(f"Model trained — {h[-1]['val_acc'] * 100:.2f}% validation accuracy. It is now used across the app.")
        st.balloons()
    if st.session_state.get("custom_model") and st.button("↩️ Revert to the default model"):
        del st.session_state["custom_model"]; st.rerun()
    params = sum(p.numel() for p in model.parameters())
    st.markdown(f"<div class='card'>🧩 Current model: <b>{params:,}</b> trainable parameters · Conv(3×3)→BN→ReLU→MaxPool ×3 → Dense(128) → Dropout → Softmax(10) · AdamW + OneCycle LR · label smoothing 0.05</div>", unsafe_allow_html=True)

elif page == "🔬 Model Insights":
    ui.hero("Model Insights", "Look inside the network", "🔬")
    cm, pred = confusion(model, Xte, yte)
    k = st.columns(4)
    ui.kpi(k[0], "Test accuracy", f"{(pred == yte).mean() * 100:.2f}%", "🎯")
    ui.kpi(k[1], "Training images", f"{len(Xtr):,}", "🖼️")
    ui.kpi(k[2], "Epochs", len(hist), "🔁")
    ui.kpi(k[3], "Parameters", f"{sum(p.numel() for p in model.parameters()):,}", "🧩")
    tabs = st.tabs(["📈 Training", "🔢 Confusion", "⚔️ CNN vs classic ML", "🗺️ Embedding", "🧬 Feature maps"])
    with tabs[0]:
        h = pd.DataFrame(hist)
        st.plotly_chart(px.line(h, x="epoch", y=["train_acc", "val_acc"], markers=True, title="Accuracy per epoch"), width="stretch")
    with tabs[1]:
        c1, c2 = st.columns([1.3, 1])
        c1.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Purples", labels=dict(x="Predicted", y="Actual"), title="Confusion matrix"), width="stretch")
        per = cm.diagonal() / cm.sum(1)
        c2.plotly_chart(px.bar(x=list(range(10)), y=per, title="Per-digit accuracy", range_y=[min(per) - 0.05, 1.0]), width="stretch")
    with tabs[2]:
        b = baselines(Xtr, ytr, Xte, yte)
        b["DigitVision CNN"] = {"acc": (pred == yte).mean(), "time": None}
        dfb = pd.DataFrame(b).T.reset_index().rename(columns={"index": "model"})
        st.plotly_chart(px.bar(dfb, x="model", y="acc", color="model", text=dfb.acc.map(lambda v: f"{v * 100:.2f}%"), range_y=[0.9, 1.005], title="Test accuracy comparison"), width="stretch")
    with tabs[3]:
        e = embedding_2d(model, Xte)
        st.plotly_chart(px.scatter(x=e[:, 0], y=e[:, 1], color=yte.astype(str), title="PCA of the CNN’s 128-d embedding — digits form clean clusters", labels={"color": "digit"}, height=480), width="stretch")
    with tabs[4]:
        j = st.slider("Test sample", 0, len(Xte) - 1, 7)
        a1, a2 = feature_maps(model, Xte[j])
        st.image(Image.fromarray((Xte[j] * 255).astype(np.uint8)).resize((120, 120), Image.NEAREST), caption=f"input · label {yte[j]}")
        st.markdown("**Block 1 activations (8×8)**")
        cols = st.columns(8)
        for i in range(16):
            m_ = a1[i] / (a1[i].max() + 1e-6)
            cols[i % 8].image(Image.fromarray((m_ * 255).astype(np.uint8)).resize((72, 72), Image.NEAREST), width="stretch")
        st.markdown("**Block 2 activations (4×4)**")
        cols = st.columns(8)
        for i in range(16):
            m_ = a2[i] / (a2[i].max() + 1e-6)
            cols[i % 8].image(Image.fromarray((m_ * 255).astype(np.uint8)).resize((72, 72), Image.NEAREST), width="stretch")
else:
    ui.account_page()
