import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Tambahan untuk grafik yang lebih bagus
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load Dataset
df = pd.read_csv('Dataset_Padi_Indonesia_2010_2024.csv')

# 2. Preprocessing
le = LabelEncoder()
categorical_cols = ['Provinsi', 'Varietas_Padi', 'Pola_Tanam', 'Serangan_Hama']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Menentukan Fitur (X) dan Target (y)
X = df.drop(['Produksi_Padi_Ton', 'Produktivitas_Ton_per_Ha'], axis=1)
y = df['Produksi_Padi_Ton']

# Split data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling khusus untuk Linear Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Inisialisasi Model
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

# 4. Training dan Evaluasi
print("--- Hasil Evaluasi Model ---")
for name, model in models.items():
    if name == "Linear Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel: {name}")
    print(f"MAE      : {mae:.2f}")
    print(f"RMSE     : {rmse:.2f}")
    print(f"R2 Score : {r2:.4f}")

# 5. Visualisasi Fitur Paling Berpengaruh (Feature Importance)
rf_model = models["Random Forest"]
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Fitur': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Membuat Grafik
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Fitur', data=feature_importance_df, palette='viridis')
plt.title('Fitur yang Paling Mempengaruhi Produksi Padi (Random Forest)')
plt.xlabel('Tingkat Kepentingan')
plt.ylabel('Nama Fitur')
plt.tight_layout()

# PENTING: Simpan grafik menjadi file gambar agar bisa dilihat di GitHub
plt.savefig('grafik_feature_importance.png')
print("\n[INFO] Grafik telah disimpan dengan nama: grafik_feature_importance.png")

# Menampilkan tabel di terminal
print("\n--- Tabel Fitur Paling Berpengaruh ---")
print(feature_importance_df)
