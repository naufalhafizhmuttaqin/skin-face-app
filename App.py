import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

MODEL_PATH = "inceptionv3_faceskin.keras"
CLASS_NAMES = ["berjerawat", "berminyak", "kering", "normal"]  # urutan alfabet = urutan flow_from_directory
IMG_SIZE = (299, 299)   # InceptionV3. Kalau pakai MobileNetV2 ganti ke (224, 224)
THRESHOLD = 0.5         # di bawah ini dianggap "tidak yakin"


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(img: Image.Image) -> np.ndarray:
    img = ImageOps.exif_transpose(img).convert("RGB").resize(IMG_SIZE)
    # PENTING: training memakai rescale=1./255, jadi inference juga harus /255
    # (jangan pakai preprocess_input bawaan Keras, skalanya beda: -1..1)
    x = np.asarray(img, dtype="float32") / 255.0
    return np.expand_dims(x, axis=0)


st.set_page_config(page_title="Deteksi Jenis Kulit Wajah", page_icon="🧴")
st.title("🧴 Deteksi Jenis Kulit Wajah")
st.caption("Unggah foto wajah close-up dengan cahaya cukup, tanpa filter/makeup tebal.")

model = load_model()

tab_upload, tab_camera = st.tabs(["Upload foto", "Ambil foto"])
with tab_upload:
    file = st.file_uploader("Pilih foto", type=["jpg", "jpeg", "png"])
with tab_camera:
    cam = st.camera_input("Foto langsung")

source = file or cam
if source:
    img = Image.open(source)
    st.image(img, caption="Foto kamu", use_container_width=True)

    with st.spinner("Menganalisis..."):
        proba = model.predict(preprocess(img), verbose=0)[0]

    idx = int(np.argmax(proba))
    if proba[idx] >= THRESHOLD:
        st.success(f"Jenis kulit: **{CLASS_NAMES[idx]}** ({proba[idx]*100:.1f}%)")
    else:
        st.warning("Model kurang yakin. Coba foto lain dengan pencahayaan lebih baik.")

    df = pd.DataFrame({"Jenis kulit": CLASS_NAMES, "Probabilitas (%)": proba * 100})
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Jenis kulit:N", sort=None, axis=alt.Axis(labelAngle=0, title=None)),
            y=alt.Y("Probabilitas (%):Q", scale=alt.Scale(domain=[0, 100])),
            tooltip=["Jenis kulit", alt.Tooltip("Probabilitas (%):Q", format=".1f")],
        )
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption("Hasil ini hanya perkiraan dari model AI, bukan diagnosis medis.")