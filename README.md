# Klasifikasi Masakan Padang

Aplikasi web untuk mengklasifikasikan jenis masakan Padang menggunakan Flask dan PyTorch.

## Fitur

- Upload gambar masakan Padang
- Klasifikasi otomatis menggunakan deep learning
- Menampilkan nama masakan dan deskripsi
- Interface web yang responsif dan user-friendly
- Mendukung format gambar: JPG, PNG, GIF

## Jenis Masakan yang Dapat Dikenali

1. **Ayam Goreng** - Ayam goreng dengan bumbu rempah khas Padang
2. **Ayam Pop** - Ayam rebus yang digoreng dengan sambal hijau
3. **Daging Rendang** - Masakan daging sapi dengan santan dan rempah
4. **Dendeng Batokok** - Daging sapi yang dipukul dan dikeringkan
5. **Gulai Ikan** - Masakan ikan dengan kuah santan dan rempah
6. **Gulai Tambusu** - Gulai usus sapi muda
7. **Gulai Tunjang** - Gulai kaki sapi dengan kuah santan
8. **Telur Balado** - Telur dengan sambal balado
9. **Telur Dadar** - Telur dadar bumbu khas Padang

## Struktur Folder

```
masakan-padang-classifier/
├── app.py                 # Aplikasi Flask utama
├── requirements.txt       # Dependensi Python
├── models/
│   └── naspad_classifier_best.pt  # Model PyTorch (Anda perlu menyediakan ini)
├── templates/
│   └── index.html        # Template HTML
└── static/
    └── uploads/          # Folder untuk menyimpan gambar upload
```

## Instalasi

1. **Clone atau download project ini**

2. **Buat virtual environment (opsional tapi direkomendasikan)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # atau
   venv\Scripts\activate     # Windows
   ```

3. **Install dependensi**
   ```bash
   pip install -r requirements.txt
   ```

4. **Pastikan model PyTorch Anda ada di folder models/**
   ```
   models/naspad_classifier_best.pt
   ```

5. **Buat folder yang diperlukan**
   ```bash
   mkdir -p static/uploads
   mkdir -p templates
   ```

6. **Simpan file HTML ke folder templates/**
   - Simpan konten HTML sebagai `templates/index.html`

## Menjalankan Aplikasi

```bash
python app.py
```

Aplikasi akan berjalan di: `http://localhost:5000`

## Cara Penggunaan

1. Buka browser dan akses `http://localhost:5000`
2. Klik tombol "Pilih Gambar" untuk upload foto masakan Padang
3. Gambar akan muncul sebagai preview
4. Klik tombol "Klasifikasi Gambar" untuk memulai analisis
5. Hasil akan menampilkan nama masakan dan deskripsinya beserta tingkat confidence

## Kustomisasi Model

Aplikasi ini menggunakan arsitektur ResNet18 sebagai backbone. Jika model Anda menggunakan arsitektur yang berbeda, Anda perlu memodifikasi class `MasakanPadangClassifier` di `app.py`.

Contoh untuk model dengan arsitektur berbeda:
```python
class MasakanPadangClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super(MasakanPadangClassifier, self).__init__()
        # Ganti dengan arsitektur model Anda
        self.backbone = models.efficientnet_b0(pretrained=True)
        self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, num_classes)
```

## Troubleshooting

### Model tidak ditemukan
Jika muncul pesan "Model file tidak ditemukan", pastikan:
- File model ada di `models/naspad_classifier_best.pt`
- Path model sudah benar di kode

### Error saat loading model
- Pastikan arsitektur model di kode sesuai dengan model yang disimpan
- Cek apakah PyTorch version compatibility

### Gambar tidak bisa diupload
- Pastikan format gambar didukung (JPG, PNG, GIF)
- Ukuran file maksimal 16MB

## Teknologi yang Digunakan

- **Flask** - Web framework
- **PyTorch** - Deep learning framework
- **torchvision** - Computer vision library
- **Pillow (PIL)** - Image processing
- **HTML/CSS/JavaScript** - Frontend

## Lisensi

Project ini bersifat open source untuk keperluan pembelajaran dan pengembangan.