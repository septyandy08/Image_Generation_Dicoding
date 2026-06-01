# Bukti Serving Model

Tempatkan screenshot berikut di folder ini:
- `serving_terminal.png` - terminal yang menjalankan `mlflow models serve -m runs:/<RUN_ID>/model -h 127.0.0.1 -p 5001 --no-conda`
- `serving_curl_response.png` - hasil `curl -X POST http://127.0.0.1:5001/invocations -H "Content-Type: application/json" -d '...'`
- Atau jika pakai Docker Hub image: `docker_serving.png` (output `docker run -p 5001:8080 septyandy08/smsml-personality:latest`).
