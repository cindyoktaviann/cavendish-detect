import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
from inference_sdk import InferenceHTTPClient

# Konfigurasi halaman
st.set_page_config(
    page_title="Klasifikasi Kematangan Pisang Cavendish",
    page_icon="🍌",
    layout="wide"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E8B57;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #f0f8f0;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 1rem 0;
    }
    .result-box {
        background-color: #fff8dc;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #ffd700;
        margin: 1rem 0;
        text-align: center;
    }
    .prediction-text {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2E8B57;
    }
    .confidence-text {
        font-size: 1.2rem;
        color: #4169e1;
    }
    .category-text {
        font-size: 1.3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model('20_epoch.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Pastikan file '20_epoch.h5' ada di direktori yang sama dengan app.py")
        return None

# Inisialisasi klien Roboflow
rf_client = InferenceHTTPClient(
    api_url="https://classify.roboflow.com",
    api_key="eUGsaherE8Tm97SGgW1w"
)

# Fungsi klasifikasi jenis pisang (Roboflow)
def classify_banana_type(image_pil):
    # Simpan gambar sementara
    temp_path = "temp_uploaded.jpg"
    image_pil.save(temp_path)
    
    result = rf_client.infer(temp_path, model_id="jenis_pisang-s3nbl/1")  # Ganti dengan ID model kamu
    class_name = result['predictions'][0]['class'] if result['predictions'] else None
    
    return class_name

# Fungsi preprocessing gambar
def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess image untuk model yang sudah memiliki Rescaling layer built-in.
    Model mengharapkan input dalam range [0, 255] sebagai float32.
    """
    
    # Konversi PIL Image ke array numpy
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Pastikan gambar dalam format RGB (3 channel)
    if len(image.shape) == 3 and image.shape[2] == 3:
        # PIL Image sudah dalam format RGB
        pass
    elif len(image.shape) == 2:
        # Jika grayscale, konversi ke RGB menggunakan PIL
        image = Image.fromarray(image).convert('RGB')
        image = np.array(image)
    
    # Convert to tensor untuk resize (gunakan dtype yang sama dengan training)
    image_tensor = tf.convert_to_tensor(image, dtype=tf.uint8)
    
    # Resize menggunakan TensorFlow dengan metode yang sama seperti training
    resized_tensor = tf.image.resize(image_tensor, target_size, method='bilinear')
    
    # Convert to float32 (sama seperti tf.keras.utils.image_dataset_from_directory)
    image = tf.cast(resized_tensor, tf.float32)
    
    # Tambahkan batch dimension
    image = tf.expand_dims(image, 0)
    
    return image

# Fungsi untuk mendapatkan kategori kematangan
def get_maturity_category(day_prediction):

    day_num = int(day_prediction.split()[-1])  # Ekstrak angka dari "Day X"
    
    if day_num <= 1:
        return "🟢 Hijau Mentah", "#90EE90"
    elif day_num <= 5:
        return "🟡 Matang Optimal", "#FFD700"
    else:
        return "🟤 Terlalu Matang", "#DEB887"

# Fungsi prediksi
def predict_banana_maturity(model, image):
    # Sesuaikan urutan label
    class_labels = ['0', '1', '3', '5', '7', '9']
    day_map = {
        '0': 'Day 0',
        '1': 'Day 1',
        '3': 'Day 3',
        '5': 'Day 5',
        '7': 'Day 7',
        '9': 'Day 9'
    }

    # Preprocessing
    processed_image = preprocess_image(image)
    
    # Prediksi
    predictions = model.predict(processed_image, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_label = class_labels[predicted_class_idx]
    
    confidence = float(predictions[0][predicted_class_idx]) * 100
    predicted_day = day_map[predicted_label]  # Convert ke Day X

    category, color = get_maturity_category(predicted_day)
    
    return predicted_day, category, confidence, color


# Header aplikasi
st.markdown('<h1 class="main-header">🍌 Klasifikasi Kematangan Pisang Cavendish</h1>', 
            unsafe_allow_html=True)

# Layout dua kolom
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📸 Input Gambar Pisang")
    
    # Tab untuk opsi input
    tab1, tab2 = st.tabs(["📁 Upload File", "📹 Kamera Live"])
    
    uploaded_image = None
    camera_image = None
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Pilih gambar pisang", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload gambar pisang dalam format JPG, JPEG, atau PNG"
        )
        if uploaded_file:
            uploaded_image = Image.open(uploaded_file)
            st.image(uploaded_image, caption="Gambar yang diupload", width=400)
    
    with tab2:
        st.markdown("📱 **Ambil gambar langsung dari kamera:**")
        camera_image = st.camera_input("Ambil foto pisang")
        if camera_image:
            camera_image = Image.open(camera_image)
            st.image(camera_image, caption="Gambar dari kamera", width=400)

with col2:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("### 📊 Legenda Kematangan")
    st.markdown("""
    **🟢 Day 0-1: Hijau Mentah**
    - Pisang masih mentah
    - Warna hijau dominan
    - Belum siap dikonsumsi
    
    **🟡 Day 3-5: Matang Optimal**  
    - Pisang matang sempurna
    - Warna kuning cerah
    - Siap dikonsumsi
    
    **🟤 Day 7-9: Cenderung Busuk**
    - Pisang terlalu matang
    - Mulai muncul bintik coklat
    - Segera dikonsumsi

    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Load model
model = load_model()

# Proses prediksi
if model is not None:
    # Tentukan gambar yang akan diprediksi
    image_to_predict = None
    source = ""
    
    if uploaded_image is not None:
        image_to_predict = uploaded_image
        source = "file upload"
    elif camera_image is not None:
        image_to_predict = camera_image
        source = "kamera"
    
    # Tombol prediksi
    if image_to_predict is not None:
        if st.button("🔍 Analisis Kematangan Pisang", type="primary", use_container_width=True):
            with st.spinner("Mendeteksi jenis pisang..."):
                try:
                    banana_type = classify_banana_type(image_to_predict)
                    
                    if banana_type != "Pisang_cavendish":
                        st.error(f"🚫 Jenis Pisang Terdeteksi: {banana_type if banana_type else 'Tidak Dikenali'}")
                        st.warning("Hanya Pisang Cavendish yang dapat dianalisis tingkat kematangannya.")
                    else:
                        with st.spinner("Menganalisis kematangan pisang Cavendish..."):
                            predicted_day, category, confidence, color = predict_banana_maturity(
                                model, image_to_predict
                            )

                            st.markdown('<div class="result-box">', unsafe_allow_html=True)
                            st.markdown("### 🎯 Hasil Prediksi")

                            col_result1, col_result2 = st.columns(2)

                            with col_result1:
                                st.markdown(f'<p class="prediction-text">Prediksi Hari: {predicted_day}</p>', 
                                        unsafe_allow_html=True)

                            with col_result2:
                                st.markdown(f'<p class="category-text" style="color: {color}">{category}</p>', 
                                        unsafe_allow_html=True)

                            st.markdown('</div>', unsafe_allow_html=True)

                            st.markdown("### 🖼️ Gambar yang Dianalisis")
                            st.image(image_to_predict, 
                                    caption=f"Sumber: {source} | Prediksi: {predicted_day} ({category})", 
                                    width=400)

                            st.markdown("### 💡 Rekomendasi")
                            if "Hijau Mentah" in category:
                                st.info("🍌 Pisang masih mentah. Tunggu beberapa hari lagi untuk kematangan optimal.")
                            elif "Matang Optimal" in category:
                                st.success("🍌 Pisang dalam kondisi matang sempurna! Siap untuk dikonsumsi.")
                            else:
                                st.warning("🍌 Pisang sudah terlalu matang. Sebaiknya segera dikonsumsi atau digunakan untuk smoothie/kue.")
                
                except Exception as e:
                    st.error(f"❌ Error dalam proses klasifikasi: {str(e)}")

    else:
        st.info("👆 Silakan upload gambar atau ambil foto pisang untuk memulai analisis kematangan.")

else:
    st.error("❌ Model tidak dapat dimuat. Pastikan file '20_epoch.h5' tersedia.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8rem; margin-top: 2rem;'>
    🍌 Aplikasi Klasifikasi Kematangan Pisang Cavendish<br>
    Menggunakan CNN (MobileNetV2) untuk prediksi tingkat kematangan pisang
</div>

""", unsafe_allow_html=True)
