# 🍌 Klasifikasi Kematangan Pisang Cavendish

Aplikasi web untuk mengklasifikasikan tingkat kematangan pisang Cavendish menggunakan Deep Learning (MobileNetV2).

## 🎯 Fitur Utama
- Upload gambar pisang atau ambil foto langsung
- Klasifikasi otomatis ke 6 kategori kematangan (Day 0, 1, 3, 5, 7, 9)
- Confidence score dan rekomendasi konsumsi
- Interface yang user-friendly

## 🤖 Model
- **Arsitektur:** MobileNetV2 (Transfer Learning)
- **Akurasi:** 98.48%
- **Input Size:** 224x224x3
- **Classes:** 6 tingkat kematangan

## 🚀 Cara Menjalankan
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📊 Dataset
Model dilatih dengan dataset pisang Cavendish yang dikumpulkan selama 9 hari dengan berbagai tingkat kematangan.

## 🛠️ Tech Stack
- Streamlit
- TensorFlow/Keras
- OpenCV
- PIL/Pillow
- NumPy
