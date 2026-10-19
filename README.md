# 🧴 Deteksi Jenis Kulit Wajah

Aplikasi web untuk memprediksi jenis kulit wajah dari sebuah foto menggunakan model deep learning (InceptionV3). Pengguna cukup mengunggah foto atau mengambil foto langsung lewat kamera, lalu aplikasi menampilkan jenis kulit beserta tingkat keyakinannya.

**🔗 Demo:** https://NAMA-APP-KAMU.streamlit.app

## Jenis kulit yang dikenali

| Kelas | Keterangan |
|---|---|
| `berjerawat` | Kulit berjerawat |
| `berminyak` | Kulit berminyak |
| `kering` | Kulit kering |
| `normal` | Kulit normal |

## Fitur

- Upload foto (JPG/PNG) atau ambil foto langsung dari kamera
- Hasil prediksi beserta persentase keyakinan untuk setiap kelas
- Peringatan otomatis jika model kurang yakin (keyakinan di bawah 50%)
- Foto tidak disimpan oleh aplikasi

## Model

- **Arsitektur:** InceptionV3 (pre-trained ImageNet, dipotong di layer `mixed4`) + GlobalAveragePooling, BatchNormalization, Dense 256, Dropout 0.5, Dense 4 (softmax)
- **Input:** 299 × 299 piksel, normalisasi `/255`
- **Training:** transfer learning (100 epoch) dilanjutkan fine-tuning (25 epoch)
- **Dataset:** [Face Skin Type](https://www.kaggle.com/datasets/muttaqin1113/face-skin-type) — 1.735 gambar, dibagi 70% train / 20% validasi / 10% test

Hasil pada data validasi:

| Metrik | Nilai |
|---|---|
| Accuracy | 0.90 |
| Macro F1-score | 0.90 |

> Catatan: performa pada foto di luar dataset (misalnya selfie dengan pencahayaan atau kamera berbeda) dapat lebih rendah dari angka di atas.

## Struktur proyek

```
.
├── app.py                      # aplikasi Streamlit
├── requirements.txt            # dependensi Python
├── inceptionv3_faceskin.keras  # model terlatih
└── README.md
```

## Menjalankan di komputer sendiri

Membutuhkan Python 3.11 atau 3.12.

```bash
git clone https://github.com/USERNAME/NAMA-REPO.git
cd NAMA-REPO

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
streamlit run app.py
```

Aplikasi akan terbuka di `http://localhost:8501`.

## Tips agar hasil lebih akurat

- Gunakan foto wajah close-up dengan pencahayaan yang cukup
- Hindari filter, makeup tebal, atau foto yang buram
- Pastikan wajah terlihat jelas dan tidak tertutup

## Disclaimer

Aplikasi ini hanya memberikan perkiraan berdasarkan model AI dan **bukan pengganti diagnosis atau saran dari dokter kulit**. Untuk masalah kulit yang serius, konsultasikan dengan tenaga medis profesional.

## Teknologi

Python · TensorFlow/Keras · Streamlit · Altair · Pillow
