import streamlit as st
import joblib
import pandas as pd
import re
import numpy as np
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# --- 1. SETUP HALAMAN ---
st.set_page_config(page_title="Analisis Sentimen Makan Gratis", page_icon="🍲")

# --- 2. FUNGSI CACHING (Agar model tidak diload berulang kali) ---
@st.cache_resource
def load_models():
    # Pastikan file model ada di folder yang sama atau sesuaikan path-nya
    # Karena di GitHub nanti strukturnya rata, biasanya langsung panggil nama file
    try:
        model = joblib.load('model/model_nb.pkl') 
        vectorizer = joblib.load('model/vectorizer_tfidf.pkl')
        return model, vectorizer
    except:
        return None, None

model, vectorizer = load_models()

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

# --- 4. TAMPILAN UI (STREAMLIT) ---
st.title("🍲 Analisis Emosi: Program Makan Gratis")
st.write("Aplikasi ini menggunakan AI (Naive Bayes) untuk mendeteksi emosi masyarakat.")

user_input = st.text_area("Masukkan komentar/pendapat:", height=100)

if st.button("Analisis Emosi"):
    if user_input and model is not None:
        with st.spinner("Sedang menganalisis..."):
            # A. Bersihkan Teks
            text_clean = clean_text(user_input)
            
            # B. Prediksi
            text_vec = vectorizer.transform([text_clean])
            prediksi_angka = model.predict(text_vec)[0]
            
            # C. Mapping Label
            label_map = {
                0: "Cemas 😰", 
                1: "Marah 😡", 
                2: "Netral 😐", 
                3: "Optimis 🌟", 
                4: "Sedih 😢", 
                5: "Senang 😄"
            }
            hasil_teks = label_map.get(int(prediksi_angka), "Tidak Diketahui")
            
            # D. Tampilkan Hasil
            st.success("Selesai!")
            st.metric("Emosi Terdeteksi", hasil_teks)
            
            with st.expander("Lihat Detail Proses"):
                st.write("**Teks Asli:**", user_input)
                st.write("**Teks Bersih:**", text_clean)
    elif model is None:
        st.error("Model tidak ditemukan! Pastikan file .pkl sudah diupload ke GitHub.")
    else:
        st.warning("Silakan masukkan teks terlebih dahulu.")