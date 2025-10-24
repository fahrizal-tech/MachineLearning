
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import (
    train_test_split, 
    StratifiedKFold, 
    cross_val_score, 
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, 
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve, 
    precision_recall_curve
)

from sklearn.inspection import permutation_importance

print("Skrip ML Pertemuan 6 Dimulai (Fokus: Random Forest)...")
print("-" * 50)


print("\n=== Langkah 1: Muat & Split Data ===")
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

print(f"Data train: {X_train.shape}, Data val: {X_val.shape}, Data test: {X_test.shape}")
print("-" * 50)


print("\n=== Langkah 2: Baseline Model (Random Forest) ===")


num_cols = X_train.select_dtypes(include="number").columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), 
    ('scaler', StandardScaler())                  
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols)
    ], 
    remainder='drop')


rf = RandomForestClassifier(
    n_estimators=300, 
    max_features='sqrt', 
    class_weight='balanced', 
    random_state=42
)


pipe = Pipeline(steps=[
    ('pre', preprocessor), 
    ('clf', rf)            
])


pipe.fit(X_train, y_train)


y_val_pred = pipe.predict(X_val)
print(f"Baseline RF F1(val): {f1_score(y_val, y_val_pred, average='macro', zero_division=0):.4f}")
print("Classification Report (Validation Set):")
print(classification_report(y_val, y_val_pred, digits=3, zero_division=0))
print("-" * 50)


print("\n=== Langkah 3: Validasi Silang (Cross-Validation) ===")


min_class_count = y_train.value_counts().min()
n_splits_cv = min(5, min_class_count) 

if n_splits_cv < 2:
    print(f"WARNING: Kelas terkecil di y_train hanya {min_class_count}. Tidak bisa melakukan CV.")
    scores_mean = 0.0
    scores_std = 0.0
else:
    print(f"Menggunakan StratifiedKFold dengan n_splits={n_splits_cv} (dibatasi oleh data)")
    skf = StratifiedKFold(n_splits=n_splits_cv, shuffle=True, random_state=42)

    
    scores = cross_val_score(
        pipe, 
        X_train, 
        y_train, 
        cv=skf, 
        scoring="f1_macro", 
        n_jobs=-1
    )
    scores_mean = scores.mean()
    scores_std = scores.std()
    print(f"CV F1-macro (train): {scores_mean:.4f} \u00B1 {scores_std:.4f}")

print("-" * 50)



print("\n=== Langkah 4: Tuning Model (GridSearchCV) ===")


if n_splits_cv < 2:
    print("WARNING: Skip GridSearchCV karena data tidak cukup untuk CV.")
    best_model = pipe 
    f1_val_best = f1_score(y_val, y_val_pred, average='macro', zero_division=0)
else:
    
    param = {
        'clf__max_depth': [None, 10, 20, 30],
        'clf__min_samples_split': [2, 5, 10]
    }

    
    skf_grid = StratifiedKFold(n_splits=n_splits_cv, shuffle=True, random_state=42)
    gs = GridSearchCV(
        pipe, 
        param_grid=param, 
        cv=skf_grid, 
        scoring="f1_macro", 
        n_jobs=-1, 
        verbose=0 
    )

    
    gs.fit(X_train, y_train)
    print("GridSearchCV Selesai.")
    
    
    print(f"Best params: {gs.best_params_}")
    best_model = gs.best_estimator_

    
    y_val_best = best_model.predict(X_val)
    f1_val_best = f1_score(y_val, y_val_best, average='macro', zero_division=0)
    print(f"Best RF - F1(val): {f1_val_best:.4f}")

print("-" * 50)



print("\n=== Langkah 5: Evaluasi Akhir di Test Set ===")


final_model = best_model
model_name = "Tuned Random Forest"

y_test_pred = final_model.predict(X_test)

print(f"--- HASIL TEST SET ({model_name}) ---")
print(f"F1(test): {f1_score(y_test, y_test_pred, average='macro', zero_division=0):.4f}")
print("\nClassification Report (Test):")
print(classification_report(y_test, y_test_pred, digits=3, zero_division=0))
print("\nConfusion Matrix (Test):")
print(confusion_matrix(y_test, y_test_pred))


try:
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    
    
    auc_score = roc_auc_score(y_test, y_test_proba)
    print(f"\nROC-AUC(test): {auc_score:.4f}")
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], 'r--') 
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve (Test Set)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("rf_roc_test.png", dpi=120) 
    print("Grafik ROC disimpan ke 'rf_roc_test.png'")
    plt.show() 

    
    prec, rec, _ = precision_recall_curve(y_test, y_test_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(rec, prec, label=f'{model_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Test Set)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("rf_pr_test.png", dpi=120) 
    print("Grafik PR disimpan ke 'rf_pr_test.png'")
    plt.show() 

except Exception as e:
    print(f"\nTidak bisa menghitung/plot ROC/PR (mungkin data test terlalu kecil): {e}")

print("-" * 50)



print("\n=== Langkah 6: Feature Importance ===")


print("\n--- 6a: Gini Importance (dari model RF) ---")
try:
    
    feature_names = final_model.named_steps['pre'].get_feature_names_out()
    
    importances = final_model.named_steps['clf'].feature_importances_

    
    forest_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    
    print("Top Feature Importance:")
    print(forest_importances.head(10)) 

    # Plot
    plt.figure(figsize=(10, 6))
    forest_importances.plot(kind='bar')
    plt.title("Feature Importance (Gini)")
    plt.ylabel("Mean decrease in impurity")
    plt.tight_layout()
    plt.savefig("rf_feature_importance.png")
    plt.show()

except Exception as e:
    print(f"Feature importance tidak tersedia: {e}")


# 6b) (Opsional) Permutation Importance
# print("\n--- 6b: Permutation Importance (Validation Set) ---")
# # Kode ini dikomen sesuai instruksi dosen
# r = permutation_importance(
#     final_model, 
#     X_val, y_val, 
#     n_repeats=10, 
#     random_state=42, 
#     n_jobs=-1
# )

# # Urutkan dan tampilkan
# sorted_idx = r.importances_mean.argsort()[::-1] # descending
# for i in sorted_idx:
#     print(f"{X_val.columns[i]:<20}: {r.importances_mean[i]:.4f} \u00B1 {r.importances_std[i]:.4f}")

print("-" * 50)



print("\n=== Langkah 7: Simpan Model Final ===")
model_filename = "rf_model.pkl"
joblib.dump(final_model, model_filename)
print(f"Model final ({model_name}) telah disimpan ke '{model_filename}'")
print("-" * 50)



print("\n=== Langkah 8: Cek Inference Lokal ===")
try:
    mdl = joblib.load("rf_model.pkl")
    
    # Data fiktif (Contoh baris pertama dari data asli)
    sample_data = {
        "IPK": 3.8,
        "Jumlah_Absensi": 3,
        "Waktu_Belajar_Jam": 10,
        "Rasio_Absensi": 3/14,      # 0.214
        "IPK_x_Study": 3.8 * 10   # 38.0
    }
    
    
    sample_df = pd.DataFrame([sample_data]) 
    
    prediksi = mdl.predict(sample_df)[0]
    print(f"Data Sample: {sample_data}")
    print(f"Hasil Prediksi (0=Tidak Lulus, 1=Lulus): {int(prediksi)}")

except Exception as e:
    print(f"Gagal melakukan inference lokal: {e}")
    
print("-" * 50)
print("=== PROSES SELESAI ===")