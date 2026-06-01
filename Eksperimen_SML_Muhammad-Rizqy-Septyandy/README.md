# Eksperimen SML — Muhammad Rizqy Septyandy

Kriteria 1 (Advance) SMSML Dicoding: eksperimen, EDA, preprocessing **Personality Classification** + otomatisasi via GitHub Actions.

## Struktur

```
Eksperimen_SML_Muhammad-Rizqy-Septyandy/
├── .github/workflows/preprocess.yml         # Otomatisasi GitHub Actions
├── personality_raw/
│   └── personality_dataset.csv               # Dataset mentah
└── preprocessing/
    ├── Eksperimen_Muhammad-Rizqy-Septyandy.ipynb   # Notebook eksperimen (template MSML)
    ├── automate_Muhammad-Rizqy-Septyandy.py        # Script otomatisasi (Skilled+)
    └── personality_preprocessing/                   # Hasil preprocessing
```

## Cara menjalankan lokal

```bash
cd preprocessing
pip install pandas==2.2.2 scikit-learn==1.5.1 joblib==1.4.2 numpy==1.26.4 \
            jupyter matplotlib seaborn
jupyter nbconvert --execute Eksperimen_Muhammad-Rizqy-Septyandy.ipynb \
                  --to notebook --inplace
python automate_Muhammad-Rizqy-Septyandy.py
```

## CI otomatis

Workflow `.github/workflows/preprocess.yml` dipantik setiap push ke `main`
yang menyentuh dataset raw / script automate. Hasilnya:
1. Menjalankan `automate_Muhammad-Rizqy-Septyandy.py`
2. Upload folder `personality_preprocessing/` sebagai GitHub Actions artifact
3. Commit kembali dataset preprocessed ke repo
