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

# Judul
st.title("Prediksi Produksi Padi di Sumatera")
st.markdown("Menggunakan algoritma: **Linear Regression**, **Random Forest**, dan **XGBoost**")

# Load Data
df = pd.read_csv("Data_Tanaman_Padi_Sumatera_version_1.csv")

# Fitur & Target
fitur = ['Luas panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata']
target = 'Produksi'

# Bagi data latih (Data historis hingga tahun terbaru yang tersedia)
df_train = df[df['Tahun'] <= 2020]

# Split train-validation
X_train, X_val, y_train, y_val = train_test_split(
    df_train[fitur], df_train[target], test_size=0.2, random_state=42
)

# Latih model
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=42)
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# Prediksi data aktual
df_actual = df.copy()
df_actual['Prediksi (Linear Regression)'] = lr.predict(df[fitur])
df_actual['Prediksi (Random Forest)'] = rf.predict(df[fitur])
df_actual['Prediksi (XGBoost)'] = xgb.predict(df[fitur])

# Tampilkan tabel data historis
st.subheader("Hasil Prediksi pada Data Historis")
st.dataframe(df_actual[['Provinsi', 'Tahun', 'Produksi', 'Prediksi (Linear Regression)', 'Prediksi (Random Forest)', 'Prediksi (XGBoost)']])

# --- PROYEKSI MASA DEPAN (Contoh 2025-2030) ---
def generate_future_data(df_base, start_year, end_year):
    future_data = []
    for year in range(start_year, end_year + 1):
        temp = df_base.copy()
        temp['Tahun'] = year
        # Simulasi fluktuasi alamiah
        temp['Luas panen'] *= np.random.uniform(0.98, 1.03)
        temp['Curah hujan'] *= np.random.uniform(0.95, 1.05)
        temp['Kelembapan'] *= np.random.uniform(0.98, 1.02)
        temp['Suhu rata-rata'] *= np.random.uniform(0.99, 1.01)
        future_data.append(temp)
    return pd.concat(future_data, ignore_index=True)

df_last = df[df['Tahun'] == 2020].copy()
df_future = generate_future_data(df_last, 2025, 2030)

df_future['Prediksi (Linear Regression)'] = lr.predict(df_future[fitur])
df_future['Prediksi (Random Forest)'] = rf.predict(df_future[fitur])
df_future['Prediksi (XGBoost)'] = xgb.predict(df_future[fitur])

st.subheader("Proyeksi Produksi Padi Tahun 2025–2030")
st.dataframe(df_future[['Provinsi', 'Tahun', 'Prediksi (Linear Regression)', 'Prediksi (Random Forest)', 'Prediksi (XGBoost)']])

# --- VISUALISASI ---
st.subheader("Visualisasi Perbandingan Prediksi Model")
tahun_terpilih = st.selectbox("Pilih Tahun untuk Detail Provinsi", sorted(df_future['Tahun'].unique()))
df_tampil = df_future[df_future['Tahun'] == tahun_terpilih]

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(df_tampil))
width = 0.25

ax.bar(x - width, df_tampil['Prediksi (Linear Regression)'], width, label='Linear Regression', color='skyblue')
ax.bar(x, df_tampil['Prediksi (Random Forest)'], width, label='Random Forest', color='orange')
ax.bar(x + width, df_tampil['Prediksi (XGBoost)'], width, label='XGBoost', color='lightgreen')

ax.set_xticks(x)
ax.set_xticklabels(df_tampil['Provinsi'], rotation=45)
ax.set_ylabel("Produksi (Ton)")
ax.set_title(f"Perbandingan Model Tahun {tahun_terpilih}")
ax.legend()
st.pyplot(fig)

# --- EVALUASI ---
st.subheader("Evaluasi Performa Model (Data Validasi)")
y_pred_lr = lr.predict(X_val)
y_pred_rf = rf.predict(X_val)
y_pred_xgb = xgb.predict(X_val)

eval_results = {
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
    'R² Score': [r2_score(y_val, y_pred_lr), r2_score(y_val, y_pred_rf), r2_score(y_val, y_pred_xgb)],
    'MAE': [mean_absolute_error(y_val, y_pred_lr), mean_absolute_error(y_val, y_pred_rf), mean_absolute_error(y_val, y_pred_xgb)]
}
st.table(pd.DataFrame(eval_results))
