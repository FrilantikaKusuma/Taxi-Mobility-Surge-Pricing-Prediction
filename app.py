import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
import warnings
# Mengabaikan warning umum dari LightGBM/Sklearn agar tampilan Streamlit bersih
warnings.filterwarnings("ignore") 

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Prediksi Tipe Harga Lonjakan (LGBM + OVR)",
    layout="wide"
)

st.title("⚡ Aplikasi Prediksi Tipe Harga Lonjakan")
st.write("Menggunakan **OneVsRest Classifier** dengan **LightGBM** dan evaluasi **ROC-AUC**.")
st.markdown("---")

# --- Fungsi untuk Memuat dan Melatih Model ---
@st.cache_resource
def load_and_train_model():
    # Mapping untuk display
    SURGE_MAP = {1: 'Rendah', 2: 'Sedang', 3: 'Tinggi'}

    try:
        # 1. Muat Data
        df = pd.read_csv("label_encoded_data.csv")
    except FileNotFoundError:
        st.error("File 'label_encoded_data.csv' tidak ditemukan. Pastikan file berada di direktori yang sama.")
        return None, None, None
    
    # Hapus kolom ID
    df = df.drop('Trip_ID', axis=1, errors='ignore')

    # 2. Pra-pemrosesan Data (Imputasi Nilai Hilang)
    # Kolom numerik yang perlu diisi
    numerical_cols_to_impute = ['Customer_Since_Months', 'Life_Style_Index', 'Customer_Rating']
    for col in numerical_cols_to_impute:
        # Mengisi nilai yang hilang dengan mean
        if df[col].isnull().any():
            # **PERBAIKAN DARI CHAINED ASSIGNMENT**
            # Menggunakan penugasan eksplisit untuk menghindari SettingWithCopyWarning
            df[col] = df[col].fillna(df[col].mean())
    
    # Target (y) dan Fitur (X)
    X = df.drop('Surge_Pricing_Type', axis=1)
    # Target LightGBM harus dimulai dari 0 (1, 2, 3 -> 0, 1, 2)
    y = df['Surge_Pricing_Type'] - 1 

    if len(y.unique()) < 2:
        st.error("Target hanya memiliki satu kelas unik. Tidak dapat melakukan klasifikasi.")
        return None, None, None

    # Bagi Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Inisialisasi dan Latih Model (OneVsRest + LGBM)
    base_clf = LGBMClassifier(
        random_state=42, 
        n_jobs=-1, 
        verbose=-1, 
        metric='auc',
        boosting_type='gbdt', 
        n_estimators=150
    )
    model = OneVsRestClassifier(base_clf, n_jobs=-1)
    model.fit(X_train, y_train)

    # 4. Evaluasi Model (ROC-AUC)
    try:
        y_proba = model.predict_proba(X_test)
        auc_score = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        
        st.sidebar.subheader("📋 Status Pelatihan Model")
        st.sidebar.write(f"**ROC-AUC (Weighted OVR pada data uji):** {auc_score:.4f}")
    except Exception as e:
        st.sidebar.error(f"Kesalahan evaluasi ROC-AUC: {e}")
        auc_score = "N/A"

    return model, list(X.columns), SURGE_MAP

# Muat model dan kolom
model, feature_cols, SURGE_MAP = load_and_train_model()

if model is not None:
    # --- Input Pengguna untuk Prediksi ---
    st.sidebar.header("Masukan Fitur untuk Prediksi")

    # Mapping untuk input
    GENDER_MAP = {0: 'Laki-laki (0)', 1: 'Perempuan (1)'}
    CONFIDENCE_MAP = {0: 'A (0)', 1: 'B (1)', 2: 'C (2)'}

    # Widget input
    input_data = {}
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        input_data['Trip_Distance'] = st.number_input(
            'Jarak Perjalanan', min_value=0.1, max_value=200.0, value=25.0, step=0.1)
        
        input_data['Type_of_Cab'] = st.selectbox(
            'Tipe Taksi', options=sorted([1, 2, 3, 4, 5, 6]), index=0)
        
        input_data['Life_Style_Index'] = st.number_input(
            'Indeks Gaya Hidup', min_value=1.0, max_value=4.0, value=2.5, step=0.01)
        
        selected_confidence = st.selectbox(
            'Confidence Index', options=list(CONFIDENCE_MAP.keys()), format_func=lambda x: CONFIDENCE_MAP[x])
        input_data['Confidence_Life_Style_Index'] = selected_confidence
        
        input_data['Cancellation_Last_1Month'] = st.number_input(
            'Pembatalan 1 Bln Terakhir', min_value=0, max_value=10, value=0)
            
    with col2:
        input_data['Customer_Since_Months'] = st.slider(
            'Pelanggan Sejak (Bulan)', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
        
        input_data['Customer_Rating'] = st.slider(
            'Rating Pelanggan', min_value=1.0, max_value=5.0, value=3.5, step=0.01)
        
        input_data['Destination_Type'] = st.selectbox(
            'Tipe Destinasi', options=sorted(list(range(0, 15))), index=0)

        selected_gender = st.selectbox(
            'Gender', options=list(GENDER_MAP.keys()), format_func=lambda x: GENDER_MAP[x])
        input_data['Gender'] = selected_gender

        input_data['Var2'] = st.slider(
            'Var2', min_value=0, max_value=100, value=50)
        input_data['Var3'] = st.slider(
            'Var3', min_value=0, max_value=100, value=75)
    
    st.sidebar.markdown("---")
    
    # --- Tombol Prediksi ---
    if st.sidebar.button("Prediksi Tipe Harga Lonjakan"):
        
        # Persiapan Data Input
        input_df = pd.DataFrame([input_data])
        
        try:
            # Urutkan kolom sesuai urutan pelatihan
            input_df = input_df[feature_cols]
        except KeyError as e:
            st.error(f"Fitur input tidak cocok dengan fitur pelatihan. Fitur yang hilang: {e}")
            st.stop()
            
        # Melakukan Prediksi
        prediction_array = model.predict(input_df)
        prediction_index = prediction_array[0] 
        probability_array = model.predict_proba(input_df)[0]
        
        # Mapping hasil (0, 1, 2) ke label asli (1, 2, 3)
        predicted_label_val = prediction_index + 1
        predicted_label = SURGE_MAP.get(predicted_label_val, f"Kategori {predicted_label_val}")

        # --- Tampilkan Hasil ---
        st.header("✨ Hasil Prediksi")
        
        if predicted_label_val == 1:
            st.success(f"**Tipe Harga Lonjakan Diprediksi:** {predicted_label} (Kategori {predicted_label_val})")
        elif predicted_label_val == 2:
            st.info(f"**Tipe Harga Lonjakan Diprediksi:** {predicted_label} (Kategori {predicted_label_val})")
        else:
            st.warning(f"**Tipe Harga Lonjakan Diprediksi:** {predicted_label} (Kategori {predicted_label_val})")
        
        # Tampilkan Probabilitas
        st.subheader("Probabilitas Kelas (Kelas 1 vs Lainnya, Kelas 2 vs Lainnya, dst.)")
        
        prob_data = {
            f'{k} ({v})': probability_array[k-1] 
            for k, v in SURGE_MAP.items()
        }
        
        prob_df = pd.DataFrame(prob_data.items(), columns=['Kategori', 'Probabilitas'])
        
        # Visualisasi probabilitas
        st.bar_chart(prob_df.set_index('Kategori'))

    st.markdown("---")
    st.info("Catatan: **Surge Pricing Type** dikodekan sebagai: **1 (Rendah)**, **2 (Sedang)**, **3 (Tinggi)**.")
