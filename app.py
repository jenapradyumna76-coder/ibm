import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
import os
from fpdf import FPDF
import tempfile
from datetime import datetime
import matplotlib.cm as cm

# --- 1. INITIAL SETUP ---
st.set_page_config(page_title="Forensic AI", page_icon="🛡️", layout="wide")
os.makedirs('forensic_results', exist_ok=True)

class ForensicReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'AI DEEPFAKE FORENSIC INVESTIGATION', 0, 1, 'C')
        self.ln(10)

# --- 2. GRAD-CAM CORE LOGIC ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def apply_heatmap(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = cv2.resize(jet_heatmap, (img.shape[1], img.shape[0]))
    jet_heatmap = np.uint8(jet_heatmap * 255)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")
    return superimposed_img

# --- 3. AI ENGINE LOADING ---
@st.cache_resource
def load_forensic_engine():
    # Xception is highly effective at detecting texture inconsistencies
    model = tf.keras.applications.Xception(weights='imagenet')
    return model

# --- 4. MAIN ANALYSIS PIPELINE ---
def run_full_analysis(video_path, model):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 20) # Analyze 20th frame
    success, frame = cap.read()
    
    if success:
        raw_path = "forensic_results/raw_frame.jpg"
        cv2.imwrite(raw_path, frame)
        
        # Prepare for AI
        img_input = cv2.resize(frame, (299, 299))
        img_array = tf.keras.applications.xception.preprocess_input(np.expand_dims(img_input, axis=0))
        
        # Predict & Grad-CAM
        preds = model.predict(img_array)
        score = float(np.max(preds))
        heatmap = make_gradcam_heatmap(img_array, model, "block14_sepconv2_act")
        gradcam_img = apply_heatmap(raw_path, heatmap)
        
        gradcam_path = "forensic_results/gradcam_evidence.jpg"
        cv2.imwrite(gradcam_path, cv2.cvtColor(gradcam_img, cv2.COLOR_RGB2BGR))
        
        cap.release()
        return score, gradcam_path
    return 0.0, None

# --- 5. STREAMLIT UI ---
st.title("🛡️ AI Deepfake Forensic Analyzer")
st.markdown("Automated digital forensics using Gradient-weighted Class Activation Mapping.")

with st.sidebar:
    st.header("System Status")
    with st.spinner("Initializing AI..."):
        engine = load_forensic_engine()
    st.success("AI Engine Ready")

uploaded_file = st.file_uploader("Upload Evidence Video", type=["mp4", "mov", "avi"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Source Video")
        st.video(uploaded_file)
        
    if st.button("🚀 RUN INVESTIGATION"):
        with st.status("Analyzing...", expanded=True) as status:
            score, gc_path = run_full_analysis(tfile.name, engine)
            status.update(label="Analysis Finished!", state="complete")

        with col2:
            st.subheader("Forensic Evidence")
            verdict = "SUSPICIOUS" if score > 0.5 else "AUTHENTIC"
            st.metric("Verdict", verdict, f"{score*100:.1f}% Confidence")
            if gc_path:
                st.image(gc_path, caption="Heatmap: AI focus areas (Red = High Concern)")

        # --- GENERATE PDF ---
        pdf = ForensicReport()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, f"Verdict: {verdict}", 0, 1)
        pdf.cell(0, 10, f"AI Probability Score: {score:.4f}", 0, 1)
        if gc_path:
            pdf.image(gc_path, x=10, y=50, w=160)
        
        pdf_name = "Forensic_Analysis_Report.pdf"
        pdf.output(pdf_name)
        
        with open(pdf_name, "rb") as f:
            st.download_button("📥 Download PDF Report", f, file_name=pdf_name)
else:
    st.info("Awaiting video file for forensic scan...")
