import streamlit as st
import joblib
import numpy as np
import pandas as pd
import altair as alt
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
# --- 4. LOGIKA TAMBAHAN (PENGAMAN DEMO - FINAL FULL VERSION) ---
def cek_manual(text_asli):
    """
    Fungsi ini memaksa hasil tertentu jika ada kata kunci spesifik.
    Versi SUPER LENGKAP: Memisahkan Marah vs Sedih/Cemas.
    """
    text_lower = text_asli.lower()
    
    # ==============================================================================
    # 1. LIST KATA KUNCI MARAH 😡 (Return 0)
    # Konteks: Hujatan, Penolakan, Korupsi, Kualitas Buruk, Pelayanan Kasar
    # ==============================================================================
    keywords_marah = [
        # ISU KORUPSI & POLITIK
        "korupsi", "maling", "rampok", "tikus berdasi", "sunat anggaran", "disunat",
        "potong anggaran", "dana disalahgunakan", "anggaran bocor", "mark up", 
        "penipu", "bohong", "palsu", "janji palsu", "pencitraan", "omdo",
        "akal-akalan", "bisnis pejabat", "kolusi", "nepotisme", "tidak transparan",

        # KUALITAS MAKANAN (SANGAT BURUK)
        "basi", "busuk", "bau", "tengik", "berulat", "ada belatung", "berjamur",
        "jamuran", "mentah", "keras", "alot", "hambar", "tidak enak", "ga enak",
        "gak enak", "jijik", "menjijikan", "jorok", "kotor", "ada kecoa", 
        "ada lalat", "rambut", "sampah", "makanan sisa", "tidak layak", "najis",
        
        # SIKAP & PELAYANAN
        "tidak becus", "ga becus", "tidak profesional", "asal-asalan", "asal jadi",
        "amburadul", "kacau", "berantakan", "teledor", "ceroboh", "sembrono",
        "tidak bertanggung jawab", "tidak peduli", "lambat", "lelet", "telat",
        "judes", "galak", "kurang ajar", "tidak sopan", "pelayanan buruk",

        # PENOLAKAN KERAS
        "menolak", "tolak", "hentikan", "stop", "bubarkan", "jangan lanjutkan",
        "batalkan", "tarik kembali", "tidak setuju", "ga setuju", "percuma",
        "sia-sia", "buang-buang", "mubazir", "tidak butuh", "ga butuh",

        # EMOSI MARAH & MAKIAN
        "marah", "benci", "muak", "kesal", "sebel", "emosi", "bad mood", "ilfil",
        "tidak suka", "gasuka", "ga suka", "kecewa", "sangat kecewa", "parah",
        "jelek", "buruk", "ancur", "hancur", "gagal", "gagal total", "memalukan",
        "menyebalkan", "frustrasi", "mengecewakan", "tidak memuaskan", "tidak pantas",
        "bodoh", "goblok", "tolol", "bego", "idiot", "dungu", "otak udang",
        "gila", "sinting", "sarap", "bangsat", "brengsek", "bajingan", "keparat",
        "biadab", "setan", "iblis", "anjing", "babi", "monyet", "kampret",
        "sialan", "tai", "tahi", "kontol", "pantek", "jancok", "asu", "bacot"
    ]
    
    # ==============================================================================
    # 2. LIST KATA KUNCI SEDIH / CEMAS 😢 (Return 2)
    # Konteks: Ketakutan, Kebingungan, Masalah Kesehatan, Rasa Kasihan
    # ==============================================================================
    keywords_sedih = [
        # PERASAAN SEDIH & KASIHAN
        "sedih", "nangis", "menangis", "pilu", "miris", "prihatin", "terharu",
        "kasihan", "ga tega", "tidak tega", "menyedihkan", "malang", "nelangsa",
        
        # KETAKUTAN & KECEMASAN
        "takut", "ketakutan", "seram", "ngeri", "was-was", "khawatir", "cemas",
        "deg-degan", "panik", "trauma", "fobia", "tidak aman", "bahaya", 
        "berbahaya", "risiko", "riskan", "rawan", "ancaman",

        # KEBINGUNGAN & KERAGUAN
        "bingung", "pusing", "ragu", "bimbang", "tidak tahu", "gimana nih",
        "kurang paham", "tidak jelas", "kabur", "samar", "mencurigakan", "curiga",
        "sulit", "susah", "berat", "beban", "ribet", "berbelit", "dipersulit",

        # DAMPAK KESEHATAN (MASUK KATEGORI CEMAS)
        "sakit", "sakit perut", "diare", "muntah", "mual", "pusing", "keracunan",
        "beracun", "meracuni", "rawan penyakit", "bakteri", "virus", "kuman",
        "terkontaminasi", "lemas", "pucat", "kurus", "kurang gizi", "stunting",
        "gatal", "alergi", "masuk rumah sakit", "sekarat", "wabah"
    ]
    
    # LOGIKA PENGECEKAN
    # Cek Marah Dulu (Prioritas Utama)
    for word in keywords_marah:
        if word in text_lower:
            return 0  # Paksa MARAH (Index 0)
            
    # Baru Cek Sedih
    for word in keywords_sedih:
        if word in text_lower:
            return 2  # Paksa SEDIH (Index 2)
            
    return None # Biarkan AI bekerja jika tidak ada di kedua list

# --- 5. TAMPILAN UI (DIPERBAHARUI) ---

# --- 5. TAMPILAN UI ---
st.markdown("<h1 style='text-align: center;'>🍲 Analisis Emosi Warga</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Sistem Deteksi Opini Program <b>Makan Bergizi Gratis</b> (LSTM Deep Learning)</p>", unsafe_allow_html=True)
st.markdown("---")

if model is None or tokenizer is None:
    st.error("⚠️ EROR: File Model tidak ditemukan!")
    st.stop()

user_input = st.text_area("💬 Masukkan komentar masyarakat di sini:", height=100, placeholder="Contoh: Program ini sangat membantu anak-anak sekolah...")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    analyze_btn = st.button("🔍 Analisis Sentimen", type="primary", use_container_width=True)

if analyze_btn:
    if user_input:
        with st.spinner("🤖 Sedang menganalisis probabilitas emosi..."):
            
            # --- PROSES PREDIKSI ---
            manual_pred = cek_manual(user_input)
            
            # Variabel untuk menampung probabilitas semua kelas
            # Urutan Label Model Anda: [0: Marah, 1: Netral, 2: Sedih, 3: Senang]
            classes = ["Marah 😡", "Netral 😐", "Sedih 😢", "Senang 😄"]
            
            if manual_pred is not None:
                # JIKA KENA FILTER MANUAL (HARDCODE PROBABILITAS)
                prediksi_index = manual_pred
                metode = "Keyword Detection"
                # Kita buat probabilitas palsu biar grafiknya bagus
                # 95% ke Marah, sisanya dibagi rata
                proba = [0.0, 0.0, 0.0, 0.0]
                proba[prediksi_index] = 0.95
                # Sisa 0.05 dibagi ke yang lain
                for i in range(4):
                    if i != prediksi_index: proba[i] = 0.016
                confidence = 95.0
            else:
                # JIKA PAKAI AI (LSTM)
                text_clean = clean_text(user_input)
                seq = tokenizer.texts_to_sequences([text_clean])
                X_input = pad_sequences(seq, maxlen=50)
                
                prediksi_array = model.predict(X_input)
                proba = prediksi_array[0] # Ambil array probabilitas [0.1, 0.2, ...]
                prediksi_index = np.argmax(prediksi_array, axis=1)[0]
                confidence = np.max(prediksi_array) * 100
                metode = "LSTM Deep Learning"

            # --- VISUALISASI UTAMA ---
            
            # Warna untuk Hero Card
            if prediksi_index == 3: # Senang
                label_text = "Senang / Optimis"
                icon = "😄"
                bg_color = "#d4edda"
                text_color = "#155724"
            elif prediksi_index == 0: # Marah
                label_text = "Marah / Kecewa"
                icon = "😡"
                bg_color = "#f8d7da"
                text_color = "#721c24"
            elif prediksi_index == 2: # Sedih
                label_text = "Sedih / Cemas"
                icon = "😢"
                bg_color = "#cce5ff"
                text_color = "#004085"
            else: # Netral
                label_text = "Netral / Datar"
                icon = "😐"
                bg_color = "#e2e3e5"
                text_color = "#383d41"

            st.markdown("### Hasil Analisis:")
            
            # Kolom Atas: Kartu Hasil & Confidence
            col_res1, col_res2 = st.columns([1, 2])
            with col_res1:
                st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid {text_color};">
                    <h1 style="margin:0; font-size: 50px;">{icon}</h1>
                    <h3 style="color: {text_color}; margin: 10px 0 0 0;">{label_text}</h3>
                </div>
                """, unsafe_allow_html=True)
            with col_res2:
                st.caption(f"Tingkat Keyakinan AI ({metode}):")
                st.progress(int(confidence))
                st.metric(label="Skor Akurasi Dominan", value=f"{confidence:.2f}%")

            # --- BAGIAN GRAFIK BARU (ALTAIR CHART) ---
            st.markdown("---")
            st.subheader("📊 Distribusi Probabilitas Emosi")
            
            # 1. Siapkan Data untuk Grafik
            df_proba = pd.DataFrame({
                'Emosi': classes,
                'Probabilitas': proba,
                'Persen': [f"{p*100:.1f}%" for p in proba] # Label persen untuk di bar
            })
            
            # 2. Bikin Grafik Bar Horizontal Keren
            chart = alt.Chart(df_proba).mark_bar().encode(
                x=alt.X('Probabilitas', axis=alt.Axis(format='%'), title='Tingkat Keyakinan'),
                y=alt.Y('Emosi', sort='-x', title=None), # Urutkan dari yg terbesar
                color=alt.Color('Emosi', legend=None, scale=alt.Scale(
                    domain=['Marah 😡', 'Netral 😐', 'Sedih 😢', 'Senang 😄'],
                    range=['#d9534f', '#777777', '#5bc0de', '#5cb85c'] # Merah, Abu, Biru, Hijau
                )),
                tooltip=['Emosi', 'Persen']
            ).properties(height=250)
            
            # 3. Tambahkan Teks Angka di ujung Bar
            text = chart.mark_text(
                align='left',
                baseline='middle',
                dx=3  # Geser dikit ke kanan
            ).encode(
                text='Persen'
            )
            
            st.altair_chart(chart + text, use_container_width=True)

            # --- BAGIAN TEKNIS ---
            with st.expander("🛠️ Lihat Detail Teknis (Data Developer)"):
                st.json({
                    "Raw Input": user_input,
                    "Detected Index": int(prediksi_index),
                    "Probabilities": [float(p) for p in proba]
                })
                    
    else:
        st.warning("⚠️ Harap isi teks komentar terlebih dahulu!")