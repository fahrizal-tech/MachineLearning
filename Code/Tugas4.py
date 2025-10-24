
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

print("Skrip ML Pertemuan 4 Dimulai...")
print("-" * 40)


print("\n=== Langkah 2: Collection ===")
try:
    df = pd.read_csv("kelulusan_mahasiswa.csv")
    print("Berhasil memuat 'kelulusan_mahasiswa.csv'")
except FileNotFoundError:
    print("ERROR: File 'kelulusan_mahasiswa.csv' tidak ditemukan.")
    print("Pastikan file CSV ada di folder yang sama dengan skrip Python ini.")
    exit() 

print("\nInfo Dataset Awal:")
print(df.info())

print("\nHead Dataset Awal (5 baris pertama):")
print(df.head())
print("-" * 40)


# =========================================
print("\n=== Langkah 3: Cleaning ===")
print("Cek Missing Values (Jumlah NaN per kolom):")
print(df.isnull().sum())
print("\nTidak ada missing values.")

print(f"\nJumlah baris sebelum drop duplicates: {len(df)}")
df = df.drop_duplicates()
print(f"Jumlah baris setelah drop duplicates: {len(df)}")
print("Tidak ada data duplikat.")

print("\nMenampilkan Boxplot IPK (untuk cek outlier)...")
plt.figure(figsize=(6, 4)) 
sns.boxplot(x=df['IPK'])
plt.title('Boxplot IPK')
plt.grid(True) 

plt.show() 
print("-" * 40)



# =========================================
print("\n=== Langkah 4: EDA ===")
print("Statistik Deskriptif:")
print(df.describe())

print("\nMenampilkan Histogram Distribusi IPK...")
plt.figure(figsize=(8, 5))
sns.histplot(df['IPK'], bins=5, kde=True) 
plt.title('Distribusi IPK')
plt.show()

print("\nMenampilkan Scatterplot (IPK vs Waktu Belajar)...")
plt.figure(figsize=(8, 5))
sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus', s=100, palette='coolwarm')
plt.title('IPK vs Waktu Belajar (diwarnai oleh Status Lulus)')
plt.grid(True)
plt.show()

print("\nMenampilkan Heatmap Korelasi...")
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap Korelasi Awal')
plt.show()
print("-" * 40)


# =========================================
print("\n=== Langkah 5: Feature Engineering ===")
df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14 
df['IPK_x_Study'] = df['IPK'] * df['Waktu_Belajar_Jam']

print("Dataset setelah Feature Engineering (5 baris pertama):")
print(df.head())

df.to_csv("processed_kelulusan.csv", index=False)
print("\nDataset telah diproses dan disimpan ke 'processed_kelulusan.csv'")
print("-" * 40)


# =========================================
print("\n=== Langkah 6: Splitting Dataset ===")

X = df.drop('Lulus', axis=1)
y = df['Lulus']

print("Fitur (X) yang digunakan untuk model:")
print(X.columns.tolist())
print("\nTarget (y) yang diprediksi:")
print(y.name)


X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, 
    test_size=0.3,          
    stratify=y,             
    random_state=42)        
    

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, 
    test_size=0.5,          
    stratify=None,         
    random_state=42)

print("\nUkuran dataset setelah di-split:")
print(f"Total data:        {len(df)} (100%)")
print(f"Data Train set:    {len(X_train)} (~70%)")
print(f"Data Validation set: {len(X_val)} (~15%)")
print(f"Data Test set:     {len(X_test)} (~15%)")
print("-" * 40)
print("=== PROSES SELESAI ===")