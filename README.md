# Face Compare Service

Microservice **Python FastAPI + OpenCV YuNet/SFace** untuk kebutuhan **face verification 1:1** pada aplikasi absensi.

Service ini dibuat sebagai pengganti API face compare eksternal seperti Face++ atau KBY-AI, dengan tujuan:

- Gratis dan open-source.
- Bisa jalan CPU-only tanpa GPU.
- Mudah dipanggil dari backend lain, misalnya Go Gin.
- Cocok untuk alur absensi 1:1:
  - foto referensi karyawan
  - dibandingkan dengan foto absensi terbaru

---

## 1. Arsitektur

```text
Aplikasi Absensi / Backend Go Gin
        |
        | POST multipart/form-data
        v
Face Compare Service
        |
        | OpenCV YuNet: deteksi wajah
        | OpenCV SFace: ekstraksi dan compare wajah
        v
Response JSON
```

Flow absensi:

```text
1. User upload foto absensi
2. Backend mengambil foto referensi karyawan dari server/database
3. Backend mengirim 2 foto ke endpoint /verify
4. Service mengembalikan similarity dan status match
5. Backend menentukan absensi diterima atau ditolak
```

---

## 2. Struktur Project

```text
face-compare-service/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── face_service.py
│   └── main.py
├── models/
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
├── .env
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 3. Requirement Server

Minimal:

```text
CPU      : 2 core
RAM      : 2 GB atau lebih
Storage  : 1 GB kosong
OS       : Linux, contoh LMDE 7 / Ubuntu 20.04+
Docker   : Docker Engine + Docker Compose Plugin
GPU      : Tidak wajib
```

Untuk production dengan traffic ramai, sesuaikan CPU/RAM berdasarkan jumlah request bersamaan.

---

## 4. Install Docker

### Ubuntu 20.04

```bash
sudo apt update
sudo apt install -y ca-certificates curl

sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources <<DOCKER_EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: focal
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
DOCKER_EOF

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo systemctl enable --now docker
sudo docker run hello-world
```

### LMDE 7

Cek codename Debian:

```bash
cat /etc/os-release
```

Install Docker:

```bash
sudo apt update
sudo apt install -y ca-certificates curl

sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources <<DOCKER_EOF
Types: deb
URIs: https://download.docker.com/linux/debian
Suites: trixie
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
DOCKER_EOF

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo systemctl enable --now docker
sudo docker run hello-world
```

Jika LMDE 7 yang digunakan tidak memakai codename `trixie`, ganti bagian ini:

```text
Suites: trixie
```

sesuai hasil:

```bash
. /etc/os-release && echo "$VERSION_CODENAME"
```

---

## 5. Download Model OpenCV YuNet dan SFace

Dari root project:

```bash
cd ~/face-compare-service
mkdir -p models
```

Download model:

```bash
wget -O models/face_detection_yunet_2023mar.onnx \
https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

wget -O models/face_recognition_sface_2021dec.onnx \
https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
```

Cek file:

```bash
ls -lh models
```

Pastikan file model tidak berukuran sangat kecil. Jika hanya beberapa KB, kemungkinan yang terdownload adalah halaman HTML error, bukan file ONNX.

---

## 6. Konfigurasi Environment

File konfigurasi ada di:

```text
.env
```

Contoh isi:

```env
APP_NAME=face-compare-service

FACE_DETECT_MODEL=/app/models/face_detection_yunet_2023mar.onnx
FACE_RECOGNITION_MODEL=/app/models/face_recognition_sface_2021dec.onnx

FACE_THRESHOLD=0.363

DETECT_SCORE_THRESHOLD=0.90
DETECT_NMS_THRESHOLD=0.30
DETECT_TOP_K=5000

MAX_UPLOAD_MB=8

ALLOW_MULTIPLE_FACES=false

OPENCV_THREADS=1
```

Penjelasan:

| Variable | Fungsi |
|---|---|
| `APP_NAME` | Nama service |
| `FACE_DETECT_MODEL` | Path model YuNet di dalam container |
| `FACE_RECOGNITION_MODEL` | Path model SFace di dalam container |
| `FACE_THRESHOLD` | Ambang batas similarity untuk menentukan match |
| `DETECT_SCORE_THRESHOLD` | Minimal confidence deteksi wajah |
| `DETECT_NMS_THRESHOLD` | Nilai NMS untuk deteksi wajah |
| `DETECT_TOP_K` | Jumlah kandidat wajah maksimal |
| `MAX_UPLOAD_MB` | Maksimal ukuran upload per file |
| `ALLOW_MULTIPLE_FACES` | Jika false, foto dengan lebih dari 1 wajah akan ditolak |
| `OPENCV_THREADS` | Jumlah thread OpenCV |

Default threshold:

```env
FACE_THRESHOLD=0.363
```

Nilai ini adalah default awal untuk model SFace dengan cosine similarity. Untuk production, threshold sebaiknya dikalibrasi memakai data absensi nyata.

---

## 7. Build Docker Image

Dari root project:

```bash
cd ~/face-compare-service
sudo docker compose build
```

---

## 8. Menjalankan Service

Jalankan container:

```bash
sudo docker compose up -d
```

Cek container:

```bash
sudo docker ps
```

Cek log:

```bash
sudo docker logs -f face-compare-service
```

Service berjalan di:

```text
http://127.0.0.1:8088
```

Secara default di `docker-compose.yml`, port hanya dibuka ke localhost:

```yaml
ports:
  - "127.0.0.1:8088:8088"
```

Artinya service hanya bisa diakses dari server yang sama. Ini lebih aman untuk production.

---

## 9. Health Check Endpoint

Endpoint:

```http
GET /health
```

Contoh request:

```bash
curl http://127.0.0.1:8088/health
```

Contoh response:

```json
{
  "status": "ok",
  "service": "face-compare-service",
  "opencv_version": "4.12.0",
  "threshold": 0.363,
  "opencv_threads": 1
}
```

---

## 10. Verify Face Endpoint

Endpoint:

```http
POST /verify
```

Content type:

```text
multipart/form-data
```

Field:

| Field | Wajib | Keterangan |
|---|---:|---|
| `reference_image` | Ya | Foto referensi karyawan |
| `probe_image` | Ya | Foto absensi terbaru |
| `threshold` | Tidak | Override threshold dari `.env` |

Contoh request:

```bash
curl -X POST http://127.0.0.1:8088/verify \
  -F "reference_image=@/home/ichi/test/ref.jpg" \
  -F "probe_image=@/home/ichi/test/absen.jpg"
```

Contoh dengan threshold manual:

```bash
curl -X POST http://127.0.0.1:8088/verify \
  -F "reference_image=@/home/ichi/test/ref.jpg" \
  -F "probe_image=@/home/ichi/test/absen.jpg" \
  -F "threshold=0.40"
```

---

## 11. Contoh Response Berhasil

Jika wajah cocok:

```json
{
  "match": true,
  "similarity": 0.512345,
  "threshold": 0.363,
  "distance_l2": 0.982345,
  "message": "same person",
  "reference_face": {
    "face_count": 1,
    "box": [120.0, 80.0, 180.0, 180.0],
    "score": 0.998123
  },
  "probe_face": {
    "face_count": 1,
    "box": [100.0, 70.0, 175.0, 175.0],
    "score": 0.997456
  }
}
```

Jika wajah tidak cocok:

```json
{
  "match": false,
  "similarity": 0.201234,
  "threshold": 0.363,
  "distance_l2": 1.456789,
  "message": "different person",
  "reference_face": {
    "face_count": 1,
    "box": [120.0, 80.0, 180.0, 180.0],
    "score": 0.998123
  },
  "probe_face": {
    "face_count": 1,
    "box": [100.0, 70.0, 175.0, 175.0],
    "score": 0.997456
  }
}
```

---

## 12. Contoh Response Error

Jika gambar tidak valid:

```json
{
  "detail": {
    "error": "INVALID_IMAGE",
    "message": "File tidak bisa dibaca sebagai gambar. Gunakan JPG/PNG yang valid."
  }
}
```

Jika wajah tidak terdeteksi:

```json
{
  "detail": {
    "error": "NO_FACE_DETECTED",
    "message": "Tidak ada wajah yang terdeteksi pada gambar."
  }
}
```

Jika terdeteksi lebih dari satu wajah:

```json
{
  "detail": {
    "error": "MULTIPLE_FACES_DETECTED",
    "message": "Terdeteksi 2 wajah. Untuk absensi, kirim foto dengan satu wajah saja."
  }
}
```

---

## 13. Integrasi dengan Backend Go Gin

Tambahkan `.env` di aplikasi Go:

```env
FACE_COMPARE_PROVIDER=opencv_sface
FACE_COMPARE_URL=http://127.0.0.1:8088/verify
FACE_COMPARE_THRESHOLD=0.363
```

Flow backend:

```text
1. Ambil path foto referensi karyawan dari database
2. Ambil path foto absensi yang baru diupload
3. Kirim keduanya ke FACE_COMPARE_URL
4. Baca response JSON
5. Jika match=true, absensi diterima
6. Simpan similarity_score dan threshold_used ke database
```

Data yang disarankan disimpan di tabel absensi:

```text
similarity_score
threshold_used
face_compare_provider
face_compare_message
face_reference_path
face_probe_path
```

---

## 14. Contoh Curl dari Backend / Terminal

```bash
REFERENCE_IMAGE="/path/foto_karyawan.jpg"
PROBE_IMAGE="/path/foto_absensi.jpg"

curl -X POST http://127.0.0.1:8088/verify \
  -F "reference_image=@${REFERENCE_IMAGE}" \
  -F "probe_image=@${PROBE_IMAGE}"
```

Dengan threshold dari environment:

```bash
REFERENCE_IMAGE="/path/foto_karyawan.jpg"
PROBE_IMAGE="/path/foto_absensi.jpg"
THRESHOLD="0.363"

curl -X POST http://127.0.0.1:8088/verify \
  -F "reference_image=@${REFERENCE_IMAGE}" \
  -F "probe_image=@${PROBE_IMAGE}" \
  -F "threshold=${THRESHOLD}"
```

---

## 15. Operasional Docker

Start service:

```bash
sudo docker compose up -d
```

Stop service:

```bash
sudo docker compose down
```

Restart service:

```bash
sudo docker compose restart
```

Lihat log:

```bash
sudo docker logs -f face-compare-service
```

Lihat penggunaan resource:

```bash
sudo docker stats face-compare-service
```

Rebuild setelah ubah code:

```bash
sudo docker compose down
sudo docker compose build --no-cache
sudo docker compose up -d
```

---

## 16. Update Konfigurasi

Jika mengubah `.env`, restart container:

```bash
sudo docker compose restart
```

Jika mengubah source code Python, rebuild:

```bash
sudo docker compose down
sudo docker compose build --no-cache
sudo docker compose up -d
```

---

## 17. Catatan Threshold

Default awal:

```env
FACE_THRESHOLD=0.363
```

Threshold ini belum tentu final untuk kondisi lapangan. Kamera, pencahayaan, kualitas foto referensi, dan sudut wajah dapat mempengaruhi hasil.

Rekomendasi kalibrasi:

```text
1. Simpan similarity setiap absensi
2. Ambil sampel data beberapa hari
3. Tandai manual mana yang benar cocok dan salah cocok
4. Jika terlalu banyak false accept, naikkan threshold
5. Jika terlalu banyak false reject, turunkan threshold
```

Contoh penyesuaian:

```text
Lebih ketat   : 0.38 - 0.42
Default awal  : 0.363
Lebih longgar : 0.34 - 0.35
```

---

## 18. Catatan Keamanan

Untuk production, jangan expose service ini langsung ke internet.

Disarankan:

```text
Go Gin Backend dan Face Compare Service berada di server yang sama
Face Compare Service hanya listen di 127.0.0.1
Public request hanya masuk ke Go Gin Backend
```

Default `docker-compose.yml` sudah membatasi port:

```yaml
ports:
  - "127.0.0.1:8088:8088"
```

Jika service harus dipanggil dari server lain, gunakan salah satu:

```text
1. Private network
2. VPN
3. Firewall whitelist IP
4. Internal Docker network
5. Reverse proxy dengan authentication
```

---

## 19. Troubleshooting

### 19.1 Container tidak jalan

Cek log:

```bash
sudo docker logs -f face-compare-service
```

### 19.2 Model tidak ditemukan

Error contoh:

```text
MODEL_NOT_FOUND
```

Cek file model:

```bash
ls -lh models
```

Pastikan file ini ada:

```text
models/face_detection_yunet_2023mar.onnx
models/face_recognition_sface_2021dec.onnx
```

Jika belum ada, download ulang:

```bash
wget -O models/face_detection_yunet_2023mar.onnx \
https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

wget -O models/face_recognition_sface_2021dec.onnx \
https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
```

Lalu rebuild:

```bash
sudo docker compose build --no-cache
sudo docker compose up -d
```

### 19.3 Health check gagal

Cek apakah container berjalan:

```bash
sudo docker ps
```

Cek port:

```bash
curl http://127.0.0.1:8088/health
```

### 19.4 Wajah tidak terdeteksi

Kemungkinan:

```text
1. Foto terlalu gelap
2. Wajah terlalu kecil
3. Wajah terlalu miring
4. Foto blur
5. Masker/topi/kacamata menutup wajah
```

Coba gunakan foto dengan wajah lebih jelas.

### 19.5 Terdeteksi lebih dari satu wajah

Secara default:

```env
ALLOW_MULTIPLE_FACES=false
```

Untuk absensi, ini lebih aman.

Jika tetap ingin mengambil wajah terbesar, ubah:

```env
ALLOW_MULTIPLE_FACES=true
```

Lalu restart:

```bash
sudo docker compose restart
```

### 19.6 Service berat saat jam masuk kantor

Cek resource:

```bash
sudo docker stats face-compare-service
```

Opsi optimasi:

```text
1. Pastikan OPENCV_THREADS=1
2. Jalankan beberapa instance service
3. Gunakan queue di backend
4. Batasi ukuran upload foto
5. Kompres foto sebelum dikirim ke service
```

---

## 20. Lisensi

Project microservice ini menggunakan library OpenCV dan model ONNX dari OpenCV Zoo.

Sebelum production skala besar, pastikan kembali lisensi model dan library yang digunakan sesuai kebutuhan organisasi.

---

## 21. Endpoint Ringkas

| Method | Endpoint | Fungsi |
|---|---|---|
| GET | `/health` | Cek status service |
| POST | `/verify` | Compare wajah 1:1 |

Field endpoint `/verify`:

| Field | Type | Keterangan |
|---|---|---|
| `reference_image` | file | Foto referensi |
| `probe_image` | file | Foto absensi |
| `threshold` | form number optional | Override threshold |

Contoh paling singkat:

```bash
curl -X POST http://127.0.0.1:8088/verify \
  -F "reference_image=@ref.jpg" \
  -F "probe_image=@absen.jpg"
```
