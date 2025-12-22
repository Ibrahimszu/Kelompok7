import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# 🌷 Konfigurasi Halaman
st.set_page_config(
    page_title="Prediksi Jenis Bunga 🌸",
    page_icon="🌼",
    layout="centered",
)

# 🌈 Custom CSS
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #f6d5f7 0%, #fbe9d7 100%);
            font-family: 'Poppins', sans-serif;
        }
        .main {
            background-color: rgba(255, 255, 255, 0.7);
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.1);
        }
        .stButton button {
            background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
            color: white;
            border-radius: 10px;
            border: none;
            padding: 0.6em 1.2em;
            font-weight: 600;
        }
        .stButton button:hover {
            background: linear-gradient(90deg, #f5576c 0%, #f093fb 100%);
        }
        h1 {
            color: #333;
            text-align: center;
            font-weight: 800;
        }
        .prediction {
            font-size: 1.3em;
            color: #f5576c;
            text-align: center;
            font-weight: 700;
            margin-top: 1em;
        }
        .footer {
            text-align: center;
            color: #777;
            margin-top: 3em;
            font-size: 0.9em;
        }
    </style>
""", unsafe_allow_html=True)

# 🌺 Judul
st.markdown("<h1>🌸 Prediksi Jenis Bunga 🌼</h1>", unsafe_allow_html=True)

# 💾 Load Model
model_path = 'my_model.keras'

if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("✅ Model berhasil dimuat!")
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        model = None
else:
    st.error(f"⚠️ File model tidak ditemukan di `{model_path}`.")
    model = None

# 📸 Upload Section
st.markdown("### Upload atau ambil gambar 🌼")

tab1, tab2 = st.tabs(["📁 Upload Gambar", "📷 Kamera"])

uploaded_file = None

with tab1:
    # 🔥 Bisa upload semua jenis gambar (tipe file bebas)
    uploaded_file = st.file_uploader("Pilih gambar...", type=None)

with tab2:
    camera_input = st.camera_input("Ambil foto")
    if camera_input is not None:
        uploaded_file = camera_input

# 🧠 Prediction Section
if uploaded_file is not None and model is not None:
    with st.container():
        image = Image.open(uploaded_file)
        st.image(image, caption="🌷 Gambar yang Dipilih", use_container_width=True)

        # Preprocess
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)[0]

        # Daftar nama kelas
        class_names = ['Lily', 'Lotus', 'Anggrek', 'Bunga Matahari', 'Tulip']

        # Probabilitas tertinggi
        max_prob = np.max(predictions)
        predicted_idx = np.argmax(predictions)
        predicted_class_name = class_names[predicted_idx]

        # Ambil dua nilai tertinggi
        sorted_preds = np.sort(predictions)[::-1]
        max1 = sorted_preds[0]
        max2 = sorted_preds[1]

        # 🔥 Aturan Deteksi Bukan Bunga
        CONF_THRESHOLD = 0.70
        DIFF_THRESHOLD = 0.20

        is_not_flower = (
            max_prob < CONF_THRESHOLD or
            (max1 - max2) < DIFF_THRESHOLD
        )

        if is_not_flower:
            st.markdown("""
                <div class="prediction">
                    🌺 Prediksi: <b>Bukan bunga</b>
                </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
                <div class="prediction">
                    🌺 Prediksi: <b>{predicted_class_name}</b><br>
                    🔮 Tingkat Kepastian: <b>{max_prob * 100:.2f}%</b>
                </div>
            """, unsafe_allow_html=True)

            # 📌 Tampilkan Probabilitas
            st.subheader("📌 Probabilitas Prediksi")
            prob_dict = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
            st.write(prob_dict)

            # 📊 Grafik Bar
            st.subheader("📊 Grafik Probabilitas")
            st.bar_chart(prob_dict)

elif uploaded_file is not None and model is None:
    st.warning("⚠️ Model gagal dimuat, prediksi tidak dapat dilakukan.")

# 🌻 Footer
st.markdown("<div class='footer'>Dibuat dengan ❤️ menggunakan Streamlit & TensorFlow</div>", unsafe_allow_html=True)
