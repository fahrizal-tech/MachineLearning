import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

print(f"Skrip ML Pertemuan 7 Dimulai (TensorFlow v{tf.__version__})...")
print("-" * 50)


print("\n=== Langkah 1: Siapkan Data ===")
try:
    df = pd.read_csv("processed_kelulusan.csv")
    print(f"Berhasil memuat 'processed_kelulusan.csv'. Total data: {len(df)} baris.")
except FileNotFoundError:
    print("ERROR: File 'processed_kelulusan.csv' tidak ditemukan.")
    exit()

X = df.drop('Lulus', axis=1)
y = df['Lulus']


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


sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_val = sc.transform(X_val)
X_test = sc.transform(X_test)

print(f"Data train: {X_train.shape}, Data val: {X_val.shape}, Data test: {X_test.shape}")
print("Data berhasil di-split dan di-scaling (tanpa data leakage).")
print("-" * 50)


print("\n=== Langkah 2: Bangun Model ANN ===")


tf.random.set_seed(42)
np.random.seed(42)


input_shape = (X_train.shape[1],) 

model = keras.Sequential([
    layers.Input(shape=input_shape),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")  
])


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", "AUC"] 
)

model.summary()
print("-" * 50)


print("\n=== Langkah 3: Training Model ===")


es = EarlyStopping(
    monitor="val_loss", 
    patience=10, 
    restore_best_weights=True 
)


history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[es],
    verbose=1
)

print("Training Selesai.")
print("-" * 50)


print("\n=== Langkah 4: Evaluasi di Test Set ===")


loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.4f}")
print(f"Test AUC:      {auc:.4f}")
print(f"Test Loss:     {loss:.4f}")


y_proba = model.predict(X_test).ravel() 

y_pred = (y_proba > 0.5).astype(int)

print("\nConfusion Matrix (Test):")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report (Test):")
print(classification_report(y_test, y_pred, digits=3, zero_division=0))
print("-" * 50)



print("\n=== Langkah 5: Visualisasi Learning Curve ===")

plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Learning Curve (Train vs Validation Loss)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=120)
print("Grafik 'learning_curve.png' berhasil disimpan.")

# --- PERBAIKAN 4: Tampilkan plot ke layar ---
plt.show()

print("=== PROSES SELESAI ===")