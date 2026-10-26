# 404 UFO

404 UFO adalah game arcade sederhana berbasis Python yang menggabungkan kontrol gerakan tangan melalui kamera dengan gameplay menghindari rintangan. Pemain menggerakkan pesawat luar angkasa ke atas atau ke bawah dengan mengubah posisi tangan yang terdeteksi kamera. Game ini dibuat sebagai proyek untuk mata kuliah Pengolahan Citra dan Video (PCV)
---

##  Tentang Game

Pemain mengendalikan posisi vertikal pesawat UFO untuk menghindari rintangan dan mengumpulkan item penambah poin serta nyawa. Gerakan pesawat dikendalikan tanpa menyentuh keyboard/mouse, melainkan murni menggunakan posisi tangan di depan kamera.

* **Pesawat UFO:** Bergerak naik-turun mengikuti pergerakan tangan pemain.
* **Asteroid:** Mengurangi 1 nyawa jika tertabrak.
* **Bintang:** Menambah skor (+50 poin) dan memulihkan 1 nyawa.
* **Tujuan:** Bertahan hidup selama mungkin untuk mengumpulkan skor tertinggi.

---

### Detail Penjelasan Pipeline:
1. **Gaussian Blur:** Menghaluskan citra input untuk mereduksi *noise* akibat kualitas sensor webcam.
2. **BGR to HSV:** Konversi ruang warna karena HSV (*Hue, Saturation, Value*), lebih stabil daripada RGB/BGR.
3. **Color Thresholding:** Menghasilkan citra biner hitam-putih (*mask*), di mana piksel bernilai putih mewakili objek yang dideteksi.
4. **Morfologi (Opening & Closing):** 
   - **Opening:** Menghilangkan derau putih kecil pada latar belakang.
   - **Closing:** Menutup lubang-lubang hitam kecil di dalam objek tangan.
5. **Bounding Box ($y + h/2$):** Mencari kontur dengan area terbesar, lalu mengambil titik tengah kotak pembungkusnya.
6. **Deadband & Low-Pass Filter:** Mengabaikan tremor tangan di bawah 2 piksel dan meratakan pergerakan dengan agar UFO tidak bergetar.
7. **Interpolasi Rentang (`np.interp`):** Mengonversi koordinat rentang kamera (30 - 210 px) ke rentang tinggi arena permainan (56 - 474 px).

---

## Teknologi

- Python
- OpenCV (`cv2`) — akses kamera dan pemrosesan citra
- NumPy — operasi array dan perhitungan numerik
- Pillow (`PIL`) — rendering teks
- Modul bawaan Python: `random`, `time`, dan `os`

---

## Fitur

- **Kontrol menggunakan tangan:** posisi tangan dideteksi dari warna merah pada video kamera.
- **Pemrosesan citra real-time:** menggunakan OpenCV untuk memproses frame, membuat mask warna, dan mendeteksi kontur.
- **Gameplay arcade:** hindari asteroid dan ambil bintang untuk bonus 50 poin.
- **Sistem skor dan nyawa:** skor bertambah seiring waktu dan saat bintang; pemain memulai dengan 5 nyawa.
- **Bintang:** menambahkan 50 poin dan memulihkan 1 nyawa hingga batas maksimum.
- **Pause dan restart:** permainan dapat dijeda dan dimulai ulang.
- **Tampilan visual:** menggunakan sprite, background, dan font vcd osr mono.

---

## Kontrol & Navigasi

| Aksi / Tombol | Fungsi |
| :--- | :--- |
| **Gerakan Tangan (Naik / Turun)** | Mengendalikan posisi pesawat secara vertikal |
| **SPACE** | Memulai permainan (*Start Game*) |
| **P** | Menjeda / Melanjutkan permainan (*Pause / Resume*) |
| **R** | Mengulang permainan saat Game Over (*Restart*) |
| **ESC** | Keluar dari aplikasi |

---

## Struktur Direktori

```text
pcv-project-1/
├── assets/
│   ├── background.png      # Gambar latar belakang game
│   ├── heart.png           # Ikon nyawa penuh
│   ├── heart_empty.png     # Ikon nyawa kosong
│   ├── obstacle1.png       # Sprite bintang 
│   ├── obstacle2.png       # Sprite asteroid
│   ├── ship.png            # Sprite pesawat UFO
│   └── vcr_osd_mono.ttf    # Font 
├── main.py                 # File utama program
└── README.md               # Dokumentasi proyek
```

---

## Instalasi & Menjalankan Program

### 1. Prasyarat
Pastikan komputer sudah terpasang:
* Python 3.8 atau lebih baru
* Kamera webcam aktif

### 2. Pasang Dependensi
Jalankan perintah berikut di terminal / command prompt:
```bash
pip install opencv-python numpy pillow
```

### 3. Jalankan Program
```bash
python main.py
```

---

## Dokumentasi Game