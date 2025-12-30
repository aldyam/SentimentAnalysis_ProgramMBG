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
st.set_page_config(page_title="Analisis Emosi LSTM", page_icon="🧠")

# --- 2. FUNGSI CACHING (Load Model LSTM & Tokenizer) ---
@st.cache_resource
def load_resources():
    try:
        # Load Model Deep Learning (.h5)
        model = load_model('Model/model_lstm.h5') # Pastikan path sesuai lokasi file
        
        # Load Tokenizer (.pkl)
        tokenizer = joblib.load('Model/tokenizer_lstm.pkl')
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, tokenizer = load_resources()

# --- 3. PREPROCESSING ---
factory_sw = StopWordRemoverFactory()
stopword = factory_sw.create_stop_word_remover()
factory_stem = StemmerFactory()
stemmer = factory_stem.create_stemmer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'@[A-Za-z0-9_]+', '', text) 
    text = re.sub(r'#\w+', '', text)           
    text = re.sub(r'http\S+', '', text)        
    text = re.sub(r'[^a-z\s]', '', text)       
    text = re.sub(r'\s+', ' ', text).strip()
    text = stopword.remove(text)
    text = stemmer.stem(text) 
    return text

# --- 4. TAMPILAN UI ---
st.title("🧠 AI Deteksi Emosi (LSTM Model)")
st.write("Aplikasi ini menggunakan **Deep Learning (Akurasi ~88%)** untuk membaca emosi program Makan Gratis.")

user_input = st.text_area("Masukkan komentar/pendapat:", height=100, placeholder="Contoh: Program ini sangat membantu anak-anak sekolah...")

if st.button("Analisis Emosi"):
    if user_input and model is not None and tokenizer is not None:
        with st.spinner("Sedang berpikir..."):
            # A. Bersihkan Teks
            text_clean = clean_text(user_input)
            
            # B. Ubah ke Sequence (Angka) - KHUSUS LSTM
            # Tokenizing
            seq = tokenizer.texts_to_sequences([text_clean])
            # Padding (agar panjangnya sama = 50, sesuai training)
            X_input = pad_sequences(seq, maxlen=50)
            
            # C. Prediksi
            prediksi_array = model.predict(X_input)
            prediksi_index = np.argmax(prediksi_array, axis=1)[0]
            confidence = np.max(prediksi_array) * 100 # Tingkat keyakinan (%)

            # D. Mapping Label (Sesuai Training Terakhir)
            # Urutan: 0=Marah, 1=Netral, 2=Sedih, 3=Senang
            label_map = {
                0: "Marah 😡",
                1: "Netral 😐",
                2: "Sedih 😢",
                3: "Senang 😄"
            }
            
            hasil_teks = label_map.get(prediksi_index, "Tidak Diketahui")
            
            # E. Tampilkan Hasil
            st.divider()
            if prediksi_index == 3: # Senang
                st.success(f"Hasil: **{hasil_teks}**")
            elif prediksi_index == 1: # Netral
                st.info(f"Hasil: **{hasil_teks}**")
            else: # Marah/Sedih
                st.error(f"Hasil: **{hasil_teks}**")
            
            st.caption(f"Tingkat Keyakinan AI: {confidence:.2f}%")

            with st.expander("Lihat Detail Proses"):
                st.write("**Teks Asli:**", user_input)
                st.write("**Teks Bersih:**", text_clean)
                st.write("**Data Sequence (Input ke LSTM):**", seq)

    elif model is None:
        st.error("❌ File model_lstm.h5 atau tokenizer_lstm.pkl tidak ditemukan! Pastikan sudah diupload.")
    else:
        st.warning("⚠️ Silakan masukkan teks terlebih dahulu.")