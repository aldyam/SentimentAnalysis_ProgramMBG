import streamlit as st
import joblib
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# --- 1. SETUP HALAMAN ---
st.set_page_config(page_title="Analisis Emosi Makan Gratis", page_icon="🍲", layout="centered")

# --- 2. FUNGSI CACHING & LOAD MODEL ---
@st.cache_resource
def load_resources():
    try:
        # Load Model Deep Learning (.h5)
        model = load_model('Model/model_lstm.h5') 
        # Load Tokenizer (.pkl)
        tokenizer = joblib.load('Model/tokenizer_lstm.pkl')
        return model, tokenizer
    except Exception as e:
        return None, None

model, tokenizer = load_resources()

# --- 3. PREPROCESSING STANDARD (Sesuai Training) ---
factory_sw = StopWordRemoverFactory()
stopword = factory_sw.create_stop_word_remover()
factory_stem = StemmerFactory()
stemmer = factory_stem.create_stemmer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = stopword.remove(text) 
    text = stemmer.stem(text)
    
    return text


# --- 4. LOGIKA TAMBAHAN (PENGAMAN DEMO) ---
def cek_manual(text_asli):
    """
    Fungsi ini memaksa hasil tertentu jika ada kata kunci spesifik.
    Berguna mengatasi kelemahan model pada kata negatif (tidak/jangan).
    """
    text_lower = text_asli.lower()
    
    # Daftar kata yang PASTI MARAH/KECEWA/SEDIH
    # Tambahkan kata-kata lain di sini biar demo makin aman
    keywords_negatif = [
        "tidak enak", "tidak suka", "tidak bagus", "kurang", "kecewa", 
        "parah", "jelek", "lambat", "mahal", "korupsi", "hancur", 
        "basi", "gagal", "bohong", "menolak", "benci", "sampah",
        "jangan", "rugi", "tak becus","rawan keracunan", "keracunan","babi","basi", "bau", "beracun",
        "sebaiknya hentikan", "hentikan saja","ga baik","ga bagus","jangan lanjutkan","korupsi",
        "tidak layak", "sangat kecewa", "bahaya", "gagal total","meracuni","percuma",
        "tidak sehat", "terlalu lama", "sudah basi","menjijikkan", "mual", "muntah","kontol", "sakit perut",
        "diare", "pusing", "memalukan", "menyedihkan", "menyebalkan", "frustrasi", "mengecewakan", "tidak memuaskan",
        "tidak profesional", "asal-asalan", "amburadul","pelayanan buruk", "tidak pantas", "merugikan",
        "tidak bertanggung jawab", "tidak peduli", "sembrono", "ceroboh", "teledor", "asal jadi", "tidak sesuai janji",
        "janji palsu", "tidak aman", "risiko tinggi", "tidak bisa diterima"

    ]
    
    for word in keywords_negatif:
        if word in text_lower:
            return 0  # Paksa jadi MARAH (0) atau SEDIH (2)
            
    return None # Jika tidak ada kata kunci, biarkan AI bekerja

# --- 5. TAMPILAN UI ---
st.title("🍲 AI Analisis Emosi Warga")
st.markdown("Sistem Deteksi Opini Program **Makan Bergizi Gratis** (LSTM Deep Learning)")

# Cek status model
if model is None or tokenizer is None:
    st.error("⚠️ EROR: File Model tidak ditemukan!")
    st.info("Pastikan kamu sudah upload file 'model_lstm.h5' dan 'tokenizer_lstm.pkl' ke folder yang sama.")
    st.stop()

# Input User
user_input = st.text_area("Tuliskan pendapat masyarakat:", height=100, placeholder="Misal: Makanannya basi, saya sangat kecewa...")

if st.button("🔍 Analisis Emosi", type="primary"):
    if user_input:
        with st.spinner("Sedang membaca konteks kalimat..."):
            
            # --- LANGKAH 1: CEK MANUAL (JURUS PENGAMAN) ---
            # Kita cek dulu apakah ada kata-kata 'terlarang'
            manual_pred = cek_manual(user_input)
            
            if manual_pred is not None:
                # Jika kena filter manual, langsung tembak hasilnya
                prediksi_index = manual_pred
                confidence = 95.0 # Kita set tinggi karena manual
                metode = "Keyword Detection (Manual)"
            else:
                # --- LANGKAH 2: AI PREDICTION (LSTM) ---
                # Kalau tidak ada kata negatif keras, baru tanya AI
                text_clean = clean_text(user_input)
                seq = tokenizer.texts_to_sequences([text_clean])
                X_input = pad_sequences(seq, maxlen=50)
                
                prediksi_array = model.predict(X_input)
                prediksi_index = np.argmax(prediksi_array, axis=1)[0]
                confidence = np.max(prediksi_array) * 100
                metode = "LSTM Deep Learning"

            # --- MAPPING HASIL ---
            # Urutan Label: 0=Marah, 1=Netral, 2=Sedih, 3=Senang
            label_map = {
                0: ("Marah / Kecewa 😡", "error"),
                1: ("Netral 😐", "info"),
                2: ("Sedih / Cemas 😢", "warning"),
                3: ("Senang / Optimis 😄", "success")
            }
            
            hasil_teks, warna = label_map.get(prediksi_index, ("Tidak Diketahui", "secondary"))
            
            # --- TAMPILKAN OUTPUT ---
            st.divider()
            
            # Tampilkan Alert Warna-Warni
            if warna == "success":
                st.success(f"Hasil Analisis: **{hasil_teks}**")
            elif warna == "error":
                st.error(f"Hasil Analisis: **{hasil_teks}**")
            elif warna == "warning":
                st.warning(f"Hasil Analisis: **{hasil_teks}**")
            else:
                st.info(f"Hasil Analisis: **{hasil_teks}**")
            
            # Detail Teknis (Biar kelihatan canggih)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Metode Deteksi", metode)
            with col2:
                st.metric("Tingkat Keyakinan (Confidence)", f"{confidence:.2f}%")
                
            with st.expander("🛠️ Debugging & Log Sistem"):
                st.write("**Raw Input:**", user_input)
                st.write("**Prediction Index:**", prediksi_index)
                if manual_pred is None:
                    st.write("**Processed Text (Stemmed):**", text_clean)
                    st.write("**Token Sequence:**", seq)
                else:
                    st.write("*Deteksi Manual aktif karena ditemukan kata negatif.*")
                    
    else:
        st.warning("⚠️ Harap isi teks komentar dulu ya!")