# Workflow-CI — SMSML Personality

Repository ini berisi MLflow Project + GitHub Actions workflow untuk
otomatisasi re-training model klasifikasi kepribadian (Extrovert vs
Introvert) **setiap push ke `main`** atau saat dipantik manual.

## Struktur

```
Workflow-CI/
├── .github/workflows/ci.yml      # GitHub Actions CI
├── MLProject/
│   ├── MLProject                  # Definisi MLflow Project
│   ├── conda.yaml                 # Environment conda
│   ├── modelling.py               # Entry point training
│   └── personality_preprocessing/ # Dataset hasil preprocessing
├── DockerHub.txt                  # Tautan ke Docker Hub image
└── README.md
```

## Cara menjalankan lokal

```bash
cd MLProject
mlflow run . --env-manager=local \
  -P data_dir=personality_preprocessing \
  -P n_estimators=300 -P max_depth=12
```

## Secrets yang harus diset di GitHub repo

- `DOCKERHUB_USERNAME` = `septyandy08`
- `DOCKERHUB_TOKEN`    = Personal Access Token Docker Hub

## Alur CI

1. Checkout repo
2. Setup Python 3.12.7 + install MLflow
3. `mlflow run` MLProject → menghasilkan model + metrik + artefak
4. Upload artefak ke GitHub Actions artifact
5. Commit `mlruns/` kembali ke repo (penyimpanan via GitHub)
6. `mlflow models build-docker` untuk membungkus model sebagai image
7. Push image ke Docker Hub (`septyandy08/smsml-personality:latest`)
