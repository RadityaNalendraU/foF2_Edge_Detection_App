# ☀️ Aplikasi Deteksi Tepi FTI-Plot Badai Matahari (foF2 Edge Detection)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep_Learning-U--Net-FF9900?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

[cite_start]Aplikasi desktop berbasis **Deep Learning** untuk mengotomatisasi deteksi parameter frekuensi kritis ionosfer (*foF2* dan *fmin*) pada citra pengamatan *Frequency-Time Intensity* (FTI) Plot [cite: 20-22]. 

[cite_start]Proyek ini dikembangkan sebagai bagian dari Kerja Praktek di **Pusat Riset Antariksa, Badan Riset dan Inovasi Nasional (BRIN)** bekerja sama dengan **Universitas Komputer Indonesia (UNIKOM)**[cite: 39, 279, 281].

---

## 📖 Latar Belakang
[cite_start]Ionosfer adalah lapisan atmosfer yang memantulkan gelombang radio frekuensi tinggi (HF), yang sangat rentan terhadap gangguan cuaca antariksa seperti badai matahari[cite: 15, 16]. [cite_start]Parameter **foF2** (frekuensi kritis lapisan F2) digunakan untuk memantau dinamika ini melalui citra *FTI-Plot*[cite: 17, 18]. 

[cite_start]Sebelumnya, peneliti mengekstraksi nilai foF2 secara manual menggunakan teknik *tracing* visual (aplikasi QuickScale) yang memakan waktu lama dan bersifat subjektif[cite: 19, 20]. [cite_start]Aplikasi ini memecahkan masalah tersebut dengan menerapkan **Arsitektur U-Net** untuk melakukan segmentasi garis tepi secara otomatis, cepat, dan presisi[cite: 20, 22].

## ✨ Fitur Utama
- [cite_start]**🖼️ Image Selector**: Antarmuka GUI yang intuitif untuk memuat data citra mentah FTI-Plot (.png)[cite: 403, 404].
- [cite_start]**🧠 AI Edge Detection (U-Net)**: Menjalankan inferensi model Deep Learning secara lokal untuk mendeteksi batas lapisan *foF2* dan *fmin*[cite: 406].
- [cite_start]**📊 Visualisasi Overlay**: Menampilkan hasil *tracing* berupa garis tepi (garis biru) langsung di atas citra asli untuk divalidasi secara visual[cite: 407, 733].
- [cite_start]**💾 Automated Data Export**: Mengonversi hasil deteksi piksel menjadi data numerik ilmiah (Frekuensi vs Waktu) dan menyimpannya ke format `.csv` beserta gambar hasil trace `.png` secara bersamaan [cite: 408-411].

## 🛠️ Teknologi yang Digunakan
- **Bahasa Pemrograman**: Python 3.10+
- **Deep Learning Framework**: PyTorch
- [cite_start]**Arsitektur Model**: U-Net [cite: 22]
- [cite_start]**Computer Vision & Preprocessing**: OpenCV / Scikit-image (Resize, Padding, Normalisasi) [cite: 405]
- **GUI Desktop**: Tkinter / PyQt
- **Data Handling**: Pandas, NumPy

## 📈 Performa Model
[cite_start]Model U-Net dilatih menggunakan dataset yang dibangun dari *ground truth* *tracing* manual sebanyak 579 data *training* (dari total 743 raw data)[cite: 20, 353, 358].
- [cite_start]**Loss Function**: DiceBCELoss [cite: 419]
- [cite_start]**Optimizer**: Adam Optimizer [cite: 419]
- [cite_start]**Epochs**: 25 [cite: 714]
- [cite_start]**Hasil**: *Train Loss* menurun secara konsisten dari `0.8245` menjadi `0.6041`, dengan *Validation Loss* konvergen di angka `0.6361`[cite: 716, 717, 718]. [cite_start]Model mampu mengenali pola lapisan F tanpa mengalami *overfitting*[cite: 721, 742].

## 📸 Demo Aplikasi

> (https://youtu.be/ZOYOSARFwaY)


## 🚀 Cara Instalasi dan Penggunaan

1. **download zip di link googledrive :**
    ```bash
    https://drive.google.com/drive/folders/1sZWT1tD4spwFeeCo0zV-07g99lyz6o2l?usp=sharing

2. **ekstrak zip**
3. **note**
    ```bash
    Aplikasi dan Model Harus Di Tempatkan Dalam 1 Folder yang Sama
4. **jalankan aplikasi tersebut**

## 👨‍💻 Tim Pengembang (IF-4 UNIKOM 2026)

| Anggota | NIM |
| :--- | :--- | 
| **Raditya Nalendra Utomo**  | 10122119 | 
| **Mochammad Rafly**  | 10122126 |
| **Gabriel Cornelius Lumbantoruan**    | 10122142 |
 

### Dosen Pembimbing: Hanhan Maulana, M.Kom., Ph.D. 
### Pembimbing Kerja Praktek (BRIN): Adi Purwono, M.T.