import streamlit as st
import time
import logging
logging.basicConfig(level=logging.INFO)
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from PIL import Image

from src.custom_resnet import run_detection_and_classification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES = ['Non-Fractured', 'Fractured']

st.set_page_config(page_title="Bone Fracture Detection and Classification", layout="wide")
try:
    with open("assets/style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'
if 'uploader_key' not in st.session_state:
    st.session_state['uploader_key'] = 0
if 'pred_label' not in st.session_state:
    st.session_state['pred_label'] = None
app_mode = st.session_state['page']



if(app_mode == "home"):
    st.markdown('<h1 class="title"; style="text-align:center; margin: 0.5rem 0;">Bone Fracture Detection and Classification</h1>', unsafe_allow_html=True)
    st.markdown('<style>[data-testid="stSidebar"]{display:none;}</style>', unsafe_allow_html=True)
    colA, colB = st.columns([1.2, 1])
    with colA:
        st.markdown(
            """
            <div class="hero">
              <div>
                <p class="hero-t">Upload an X-ray image. The app runs YOLOv11 for object detection and EfficientNet-B3 for binary classification: <strong>Fractured</strong> or <strong>Non-Fractured</strong>.</p>
                <div class="cta"></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Model details", expanded=True):
            st.markdown(
                "Upload an X-ray image. The app runs YOLOv11 for object detection and EfficientNet-B3 for binary classification: **Fractured** or **Non-Fractured**.\n\n"
                "- **Detection**: YOLOv11s weights at `models/yolov11_trained.pt`.\n"
                "- **Classification**: EfficientNet-B3 weights at `models/best_model_efficient.pth`.\n"
                "- **Labels**: `Non-Fractured`, `Fractured`.\n"
                "- **Device**: CPU by default; uses CUDA automatically if available."
            )

        start = st.button("Start Analysis", type="primary")
        if start:
            st.session_state['page'] = 'analysis'
            # reset uploader by bumping key and clear prediction
            st.session_state['uploader_key'] += 1
            st.session_state['pred_label'] = None
            st.rerun()
        st.markdown(
            """
            <div class="card-grid">
              <div class="card"><h3>Two-Headed System</h3>
                <ul style="text-align:left; margin:0; padding-left:1.2rem;">
                  <li><strong>Object Detection</strong>: YOLOv11 draws bounding boxes.</li>
                  <li><strong>Classification</strong>: EfficientNet predicts Fractured/Non-Fractured.</li>
                </ul>
              </div>
              <div class="card"><h3>Fast Results</h3><p>Get detection and classification with confidence scores.</p></div>
            </div>
            
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="card-grid">
                <div class="card"><h3>On CPU/GPU</h3><p>
                    Custom ResNet CNN (2-class), trained on 4,000 images, 99.8% accuracy.<br/>
                    Runs on CPU by default; accelerates with CUDA if available.</p>
                </div>
              
            </div>
            """,
            unsafe_allow_html=True,
        )
    with colB:
        import os
        sample_image_path = "test_img/image1_145_png.rf.a69d928d011a93d25a95b7b8380ea25d.jpg"
        if os.path.exists(sample_image_path):
            st.image(sample_image_path, caption="Example X-ray Image", width=500)
        else:
            st.info("Sample image not found. Use the Analysis page to upload an image.")

    st.markdown("""
        <div class="footer">For education/demo only. Not for clinical use.</div>
    """, unsafe_allow_html=True)

elif(app_mode=="analysis"):
    # Top navigation
    nav_cols = st.columns([0.2, 0.6, 0.2])
    with nav_cols[0]:
        if st.button("Home"):
            st.session_state['page'] = 'home'
            st.session_state['uploader_key'] += 1
            st.session_state['pred_label'] = None
            st.rerun()
    with nav_cols[1]:
        st.markdown('<div id="stroke-analysis"></div>', unsafe_allow_html=True)
    with nav_cols[2]:
        pass

    st.markdown('<div id="cancer-analysis"><p>Bone Fracture Analysis</p></div>', unsafe_allow_html=True)
    #st.header("Brain Cancer Analysis")
    uploaded = st.file_uploader(
        "Upload X-ray Image",
        type=["jpg", "jpeg", "png"],
        key=f"uploader_{st.session_state['uploader_key']}"
    )

    if uploaded is None:
        st.info("Please upload an X-ray image to proceed.")
    else:
        image = Image.open(uploaded).convert('RGB')
        display_image = image.resize((300,300))
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(display_image, caption="Uploaded Image", width=300)
            c1, c2 = st.columns(2)
            with c1:
                predict_clicked = st.button("Predict", type="primary")
            with c2:
                clear_clicked = st.button("Clear")

        if clear_clicked:
            st.session_state['uploader_key'] += 1
            st.session_state['pred_label'] = None
            st.rerun()

        if predict_clicked:
            with st.spinner("Running detection and classification..."):
                start = time.time()
                annotated_np, pred_idx, conf, detections = run_detection_and_classification(image)
                end = time.time()
                logging.info(f"Prediction Response Time: {end - start:.4f} sec")

            annotated_img = Image.fromarray(annotated_np)
            pred_label = CLASS_NAMES[pred_idx]
            st.session_state['pred_label'] = pred_label

            with col2:
                st.image(annotated_img.resize((500,400)), caption=f"Detection + Prediction: {pred_label} ({conf:.2f})", width=500)

    st.markdown("""
            <div class="footer">For education/demo only. Not for clinical use.</div>
        """, unsafe_allow_html=True)
