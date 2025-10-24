
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib 
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

print("Skrip ML Pertemuan 5 Dimulai...")
print("-" * 50)


print("\n=== Langkah 1: Muat & Split Data ===")
try:
    df = pd.read_csv("processed_kelulusan.csv")
    print(f"Berhasil memuat 'processed_kelulusan.csv'. Total data: {len(df)} baris.")
except FileNotFoundError:
    print("ERROR: File 'processed_kelulusan.csv' tidak ditemukan.")
    print("Pastikan file ini ada dari hasil Pertemuan 4.")
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
print("Fitur yang digunakan:", X.columns.tolist())
print("-" * 50)


print("\n=== Langkah 2: Baseline Model (Logistic Regression) ===")


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


logreg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)


pipe_lr = Pipeline(steps=[
    ('pre', preprocessor),
    ('clf', logreg) 
])


pipe_lr.fit(X_train, y_train)


y_val_pred_lr = pipe_lr.predict(X_val)

f1_lr_val = f1_score(y_val, y_val_pred_lr, average='macro', zero_division=0)

print(f"Baseline (LogReg) F1 (val): {f1_lr_val:.4f}")
print("Classification Report (Validation Set):")
print(classification_report(y_val, y_val_pred_lr, digits=3, zero_division=0))
print("-" * 50)


print("\n=== Langkah 3: Model Alternatif (Random Forest) ===")

rf = RandomForestClassifier(
    n_estimators=300, 
    max_features='sqrt', 
    class_weight='balanced', 
    random_state=42
)

pipe_rf = Pipeline(steps=[
    ('pre', preprocessor), 
    ('clf', rf)
])


pipe_rf.fit(X_train, y_train)


y_val_pred_rf = pipe_rf.predict(X_val)
f1_rf_val = f1_score(y_val, y_val_pred_rf, average='macro', zero_division=0)

print(f"RandomForest F1 (val): {f1_rf_val:.4f}")
print("-" * 50)


print("\n=== Langkah 4: Tuning Model (Random Forest) ===")
print("Memulai GridSearchCV...")


min_class_count = y_train.value_counts().min()
n_splits_cv = min(5, min_class_count) 

if n_splits_cv < 2:
    print(f"WARNING: Data latih terlalu sedikit (kelas terkecil hanya {min_class_count}). Skip GridSearchCV.")
    
    best_rf = pipe_rf 
    f1_best_rf_val = f1_rf_val
else:
    print(f"Menggunakan StratifiedKFold dengan n_splits={n_splits_cv}")
    skf = StratifiedKFold(n_splits=n_splits_cv, shuffle=True, random_state=42)

    
    param_grid = {
        'clf__max_depth': [None, 10, 20],
        'clf__min_samples_split': [2, 5] 
    }

    
    gs = GridSearchCV(
        estimator=pipe_rf,
        param_grid=param_grid,
        cv=skf,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=0
    )

    
    gs.fit(X_train, y_train)

    
    print("GridSearchCV Selesai.")
    print(f"Best params: {gs.best_params_}")
    print(f"Best CV F1 score: {gs.best_score_:.4f}")

    
    best_rf = gs.best_estimator_
    y_val_pred_best_rf = best_rf.predict(X_val)
    f1_best_rf_val = f1_score(y_val, y_val_pred_best_rf, average='macro', zero_division=0)
    print(f"Best RF (Tuned) F1 (val): {f1_best_rf_val:.4f}")

print("-" * 50)



print("\n=== Langkah 5: Evaluasi Akhir di Test Set ===")


if f1_best_rf_val >= f1_lr_val and f1_best_rf_val >= f1_rf_val:
    final_model = best_rf
    model_name = "Tuned Random Forest"
elif f1_rf_val >= f1_lr_val:
    final_model = pipe_rf
    model_name = "Untuned Random Forest"
else:
    final_model = pipe_lr
    model_name = "Baseline Logistic Regression"
    
print(f"Model final yang dipilih (F1 Val terbaik): {model_name}")


y_test_pred = final_model.predict(X_test)

print(f"\n--- HASIL TEST SET ({model_name}) ---")
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
    plt.savefig("roc_test_curve.png", dpi=120) 
    print("Grafik ROC-AUC disimpan ke 'roc_test_curve.png'")
    
    plt.show() 

except Exception as e:
    print(f"\nTidak bisa menghitung/plot ROC-AUC: {e}")

print("-" * 50)


print("\n=== Langkah 6: Simpan Model Final ===")
model_filename = "final_model.pkl"
joblib.dump(final_model, model_filename)
print(f"Model final ({model_name}) telah disimpan ke '{model_filename}'")
print("=== PROSES SELESAI ===")