# Implementasi SVM dan Random Forest untuk Klasifikasi Mutu Kopi Arabika

Project ini dibuat untuk eksperimen machine learning skripsi dengan judul:

> Implementasi model SVM dan Random Forest untuk klasifikasi mutu Kopi Arabika berbasis Web

Fokus project saat ini adalah pipeline machine learning berbasis Python. Dataset asli tetap berada di `DATASET/Coffee-modified.csv`.

## Ringkasan Dataset

Dataset berisi data cupping kopi Arabika dengan fitur sensorik, fisik, geografis, dan proses pascapanen. Karena dataset belum menyediakan label kelas mutu eksplisit, target klasifikasi dibuat dari kolom `Total.Cup.Points`.

Kelas mutu yang digunakan:

| Rentang Total Cup Points | Label |
| --- | --- |
| 0 sampai < 80 | Below Specialty |
| 80 sampai < 85 | Very Good |
| 85 sampai <= 100 | Excellent |

Catatan metodologis: kolom `Total.Cup.Points` hanya dipakai untuk membentuk label, tidak dipakai sebagai fitur training. Ini penting agar model tidak langsung membaca jawaban target.

## Struktur Project

```text
.
├── DATASET/
│   └── Coffee-modified.csv
├── configs/
│   └── training_config.json
├── examples/
│   └── sample_input.json
├── scripts/
│   ├── predict_sample.py
│   └── train_models.py
├── src/
│   └── coffee_quality/
│       ├── config.py
│       ├── data.py
│       ├── evaluation.py
│       ├── pipeline.py
│       └── visualization.py
├── artifacts/
│   ├── figures/
│   ├── metrics/
│   ├── models/
│   └── processed/
└── requirements.txt
```

## Cara Kerja Program

1. Membaca dataset CSV.
2. Membersihkan baris tidak valid, termasuk baris yang bukan `Arabica`, nilai target kosong, skor target 0, dan skor sensorik di luar rentang 0 sampai 10.
3. Membuat label mutu dari `Total.Cup.Points`.
4. Memilih fitur numerik dan kategorikal yang relevan.
5. Membagi data menjadi train dan test dengan stratifikasi label.
6. Melatih dua model:
   - Support Vector Machine dengan scaling, class weight, dan GridSearchCV.
   - Random Forest dengan class weight dan RandomizedSearchCV.
7. Mengevaluasi model memakai accuracy, balanced accuracy, precision macro, recall macro, F1 macro, F1 weighted, ROC AUC, classification report, dan confusion matrix.
8. Membuat visualisasi EDA, confusion matrix, ROC curve, feature importance, dan permutation importance.
9. Menyimpan model terbaik tiap algoritma ke folder `artifacts/models/`.

## Instalasi

Buat virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependency:

```bash
pip install -r requirements.txt
```

## Menjalankan Training

```bash
python scripts/train_models.py
```

Jika ingin training lebih cepat dan melewati permutation importance:

```bash
python scripts/train_models.py --skip-permutation
```

Output utama:

```text
artifacts/models/svm_model.joblib
artifacts/models/random_forest_model.joblib
artifacts/metrics/model_comparison.csv
artifacts/metrics/training_summary.json
artifacts/figures/*.png
artifacts/processed/coffee_cleaned_with_quality_label.csv
```

## Menjalankan Prediksi Contoh

Setelah training selesai:

```bash
python scripts/predict_sample.py --model artifacts/models/random_forest_model.joblib
```

Input contoh ada di `examples/sample_input.json`.

## Menjalankan Website Dashboard

Website dibuat dengan Flask dan memakai output machine learning yang sama dari folder `artifacts/`.

Jalankan server:

```bash
source .venv/bin/activate
python web/app.py
```

Buka browser:

```text
http://127.0.0.1:5000
```

Akun login:

```text
Username: Risyad
Password: Telkom321!
```

Fitur website:

- Login page dengan satu user.
- Dashboard slide untuk ringkasan dataset, performa model, visualisasi, prediksi, dan training ulang.
- Galeri visual dari output training: distribusi label, sebaran skor, heatmap korelasi, confusion matrix, ROC curve, dan feature importance.
- Form prediksi mutu kopi menggunakan model `svm_model.joblib` atau `random_forest_model.joblib`.
- Tombol `Run Training` untuk generate ulang output training cepat dari dashboard.

Screenshot verifikasi lokal tersimpan di:

```text
reports/screenshots/dashboard-overview.png
reports/screenshots/dashboard-prediction.png
reports/screenshots/dashboard-overview-desktop.png
reports/screenshots/dashboard-model-desktop.png
```

## Connect Project ke GitHub

Jika repository GitHub Anda adalah:

```text
https://github.com/brewokpincang/skripsh1t.git
```

Jalankan dari root project:

```bash
git init
git branch -M main
git remote add origin https://github.com/brewokpincang/skripsh1t.git
git add .
git commit -m "Build coffee quality ML dashboard"
```

Karena repository GitHub Anda sudah punya commit awal, ambil isi remote dulu:

```bash
git pull origin main --allow-unrelated-histories
```

Jika ada conflict di `README.md`, pilih isi README project lokal ini atau gabungkan manual di Cursor, lalu:

```bash
git add .
git commit -m "Merge initial GitHub README"
git push -u origin main
```

Jika GitHub meminta login dari terminal, gunakan GitHub Personal Access Token sebagai password saat diminta.

## Parameter Model

SVM diuji dengan:

- Kernel: `rbf`, `linear`
- C: `0.1`, `1`, `10`, `30`
- Gamma untuk RBF: `scale`, `0.01`, `0.1`
- Class weight: `balanced`

Random Forest diuji dengan:

- n_estimators: `200`, `400`, `600`
- max_depth: `None`, `6`, `10`, `16`, `24`
- min_samples_split: `2`, `5`, `10`
- min_samples_leaf: `1`, `2`, `4`
- max_features: `sqrt`, `log2`
- class_weight: `balanced_subsample`

Scoring utama untuk pemilihan model adalah `f1_macro` karena dataset tidak seimbang antar kelas.
