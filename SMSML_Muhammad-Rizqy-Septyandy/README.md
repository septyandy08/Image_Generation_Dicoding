# SMSML — Muhammad Rizqy Septyandy

Submission akhir kelas **Membangun Sistem Machine Learning (MSML) Dicoding**, semua 4 kriteria di level **Advance (4 pts)**.

| Kriteria | Topik | Level | Lokasi |
|---|---|---|---|
| 1 | Eksperimen, EDA, preprocessing + GitHub Actions | Advance | repo terpisah `Eksperimen_SML_Muhammad-Rizqy-Septyandy/` |
| 2 | MLflow + DagsHub + manual logging + 2 artefak | Advance | `Membangun_model/` |
| 3 | MLflow Project + CI + Docker Hub (`mlflow build-docker`) | Advance | repo terpisah `Workflow-CI/` |
| 4 | Serving + Prometheus + Grafana 10 metrik + 3 alert | Advance | `Monitoring dan Logging/` |

## Cara reproduksi end-to-end

### Kriteria 1
```bash
cd Eksperimen_SML_Muhammad-Rizqy-Septyandy/preprocessing
jupyter nbconvert --execute Eksperimen_Muhammad-Rizqy-Septyandy.ipynb --to notebook --inplace
python automate_Muhammad-Rizqy-Septyandy.py
```
GitHub Actions (`.github/workflows/preprocess.yml`) menjalankan pipeline ini setiap push.

### Kriteria 2
```bash
cd Membangun_model
pip install -r requirements.txt

# Lokal (sanity check)
python modelling.py
python modelling_tuning.py

# Tracking ke DagsHub (Advance)
export MLFLOW_TRACKING_USERNAME=septyandy08
export MLFLOW_TRACKING_PASSWORD=<DAGSHUB_TOKEN>
python modelling_tuning.py --use_dagshub
```

### Kriteria 3
```bash
cd Workflow-CI/MLProject
mlflow run . --env-manager=local -P n_estimators=300 -P max_depth=12
```
CI workflow build Docker image via `mlflow models build-docker` lalu push ke `septyandy08/smsml-personality`.

### Kriteria 4
```bash
# 1. Serve model
mlflow models serve -m runs:/<RUN_ID>/model -h 127.0.0.1 -p 5001 --no-conda

# 2. Run exporter + Prometheus + Grafana
cd "Monitoring dan Logging"
python 3.prometheus_exporter.py &
docker compose up -d

# 3. Generate trafik
python 7.inference.py --n 100

# 4. Grafana di http://localhost:3000 (login septyandy08/septyandy08)
#    - Buat dashboard bernama "septyandy08" dengan 10+ panel
#    - Import alert rules dari alert_rules.yml (3 alert)
```

## TODO oleh siswa sebelum submit

1. Ganti placeholder `.txt` screenshot di `Membangun_model/` dengan `.jpg` asli dari DagsHub.
2. Isi folder `Monitoring dan Logging/1.bukti_serving/` dst. dengan screenshot asli.
3. Buat dua repo GitHub baru (`Eksperimen_SML_Muhammad-Rizqy-Septyandy` dan `Workflow-CI`), Public visibility, lalu update tautan di file `.txt` di sini.
4. Set secrets di repo Workflow-CI:
   - `DOCKERHUB_USERNAME` = `septyandy08`
   - `DOCKERHUB_TOKEN`    = PAT Docker Hub
5. Push semua, pastikan Actions hijau, lalu zip folder ini sebagai `SMSML_Muhammad-Rizqy-Septyandy.zip`.
