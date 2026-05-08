import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Konfigurasi Halaman
st.set_page_config(page_title="Prediksi Produksi Padi Indonesia", layout="wide")

# Judul
st.title("🌾 Dashboard Analisis Produksi Padi Indonesia")
st.markdown("Menggunakan Dataset: **2010–2024** | Model: **Linear Regression, Random Forest, XGBoost**")

# --- 1. LOAD DATA ---
@st.cache_data
def load_data():
    # Pastikan file ini sudah di-push ke GitHub
    df = pd.read_csv("Dataset_Padi_Indonesia_2010_2024.csv")
    return df

try:
    df = load_data()
    
    # Preprocessing: Encoding Kolom Kategorikal
    le = LabelEncoder()
    df_encoded = df.copy()
    cat_cols = ['Provinsi', 'Varietas_Padi', 'Pola_Tanam', 'Serangan_Hama']
    for col in cat_cols:
        df_encoded[col] = le.fit_transform(df[col].astype(str))

    # Fitur & Target (Sesuai kolom di dataset Anda)
    fitur = ['Luas_Tanam_Ha', 'Luas_Panen_Ha', 'Curah_Hujan_mm', 'Kelembapan_Persen', 'Suhu_RataRata_C', 'Penggunaan_Pupuk_Ton', 'Irigasi_Persen']
    target = 'Produksi_Padi_Ton'

    # --- 2. TRAINING MODEL ---
    X = df_encoded[fitur]
    y = df_encoded[target]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inisialisasi Model
    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

    # Latih
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    # --- 3. TAMPILAN DASHBOARD ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 Sampel Data Dataset")
        st.dataframe(df[['Provinsi', 'Tahun', 'Produksi_Padi_Ton']].head(10))

    with col2:
        st.subheader("🎯 Akurasi Model (R² Score)")
        y_pred_lr = lr.predict(X_val)
        y_pred_rf = rf.predict(X_val)
        y_pred_xgb = xgb.predict(X_val)

        eval_df = pd.DataFrame({
            'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
            'R² Score': [r2_score(y_val, y_pred_lr), r2_score(y_val, y_pred_rf), r2_score(y_val, y_pred_xgb)],
            'MAE': [mean_absolute_error(y_val, y_pred_lr), mean_absolute_error(y_val, y_pred_rf), mean_absolute_error(y_val, y_pred_xgb)]
        })
        st.table(eval_df.style.format({'R² Score': '{:.4f}', 'MAE': '{:,.2f}'}))

    # --- 4. VISUALISASI FEATURE IMPORTANCE ---
    st.divider()
    st.subheader("📈 Fitur Paling Berpengaruh (Random Forest)")
    
    importances = rf.feature_importances_
    feat_importances = pd.Series(importances, index=fitur).sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    feat_importances.plot(kind='barh', color='teal', ax=ax)
    ax.set_title("Analisis Faktor Penentu Produksi Padi")
    st.pyplot(fig)

    # --- 5. PERBANDINGAN TREN ---
    st.subheader("📉 Perbandingan Hasil Prediksi vs Data Aktual")
    df['Prediksi_XGB'] = xgb.predict(X)
    
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=df, x='Tahun', y='Produksi_Padi_Ton', label='Data Aktual', color='black', marker='o')
    sns.lineplot(data=df, x='Tahun', y='Prediksi_XGB', label='Prediksi XGBoost', color='red', linestyle='--')
    ax2.set_title("Tren Produksi Padi Nasional")
    st.pyplot(fig2)

except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")
    st.info("Pastikan file 'Dataset_Padi_Indonesia_2010_2024.csv' sudah di-push ke GitHub.")
