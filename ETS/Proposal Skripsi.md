# PERBANDINGAN NAIVE FORECAST, ARIMA, DAN GATED RECURRENT UNIT (GRU) DENGAN OPTIMASI HYPERPARAMETER (BAYESIAN OPTIMIZATION) UNTUK PERAMALAN HARGA KOMODITAS EMAS, NIKEL, DAN BATUBARA

## Deskripsi Umum

Repositori ini merupakan bagian dari penyusunan **Proposal Skripsi** untuk mata kuliah **Riset Informatika**.  
Penelitian ini berfokus pada **perbandingan model peramalan (forecasting)** menggunakan pendekatan statistik klasik dan deep learning, yaitu **Naive Forecast**, **ARIMA**, dan **Gated Recurrent Unit (GRU)**.  
Selain itu, dilakukan **optimasi hyperparameter GRU menggunakan Bayesian Optimization** untuk memperoleh performa model terbaik.

Dataset yang digunakan berasal dari **World Bank – Pink Sheet Commodity Prices**, yang berisi data bulanan harga komoditas dunia seperti **emas, nikel, dan batubara**.

Struktur proposal terdiri dari tiga bab utama:

1. **Bab I – Research Problem Formulation (Perumusan Masalah Penelitian)**  
2. **Bab II – Gap Research Analysis & Mind Map**  
3. **Bab III – Methodology**

---

## Bab I — Research Problem Formulation

### 1.1 Identify Broad Field

| Bidang Luas | Fokus |
| ------------ | ------ |
| Data Science & Artificial Intelligence | Penerapan algoritma Machine Learning dan Deep Learning untuk peramalan harga komoditas |
| Financial & Commodity Informatics | Analisis data ekonomi dan harga pasar global berbasis model prediksi |

---

### 1.2 Dissect the Subareas

- **Time Series Forecasting** – pemodelan data deret waktu untuk prediksi nilai masa depan.  
- **Statistical Modeling (ARIMA)** – model klasik yang mengandalkan kestasioneran dan autokorelasi.  
- **Deep Learning (GRU)** – varian RNN yang mampu menangkap pola nonlinier dan dependensi jangka panjang.  
- **Hyperparameter Optimization** – proses sistematis untuk meningkatkan performa model GRU dengan Bayesian Optimization.  

---

### 1.3 Select Interested Sub-Area

Fokus penelitian:

> Perbandingan performa antara Naive Forecast, ARIMA, dan GRU dengan optimasi hyperparameter berbasis Bayesian Optimization dalam peramalan harga komoditas (emas, nikel, dan batubara) menggunakan data World Bank Pink Sheet.

---

### 1.4 Raise Research Questions

1. Bagaimana performa model Naive Forecast, ARIMA, dan GRU dalam memprediksi harga komoditas?  
2. Apakah optimasi hyperparameter menggunakan Bayesian Optimization dapat meningkatkan akurasi model GRU?  
3. Model mana yang memberikan hasil prediksi paling akurat untuk masing-masing komoditas (emas, nikel, dan batubara)?  

---

### 1.5 Formulate Objectives

- Menerapkan dan membandingkan tiga pendekatan forecasting: Naive Forecast, ARIMA, dan GRU.  
- Mengoptimasi model GRU menggunakan Bayesian Optimization untuk memperoleh parameter terbaik.  
- Mengevaluasi dan membandingkan performa model berdasarkan metrik seperti RMSE, MAE, dan DA.  
- Mengidentifikasi model paling efektif untuk masing-masing komoditas.  

---

### 1.6 Assessment of Objectives

| Kriteria | Status | Catatan |
| -------- | ------- | ------- |
| Feasible | ✅ | Dataset World Bank tersedia publik dan format time series mudah diolah |
| Relevant | ✅ | Relevan untuk analisis ekonomi dan sistem cerdas berbasis data |
| Significant | ✅ | Kontribusi dalam menemukan model prediksi terbaik untuk harga komoditas global |

---

### 1.7 Double Check

Pemilihan kombinasi ARIMA dan GRU dianggap tepat karena merepresentasikan perbandingan metode klasik dan modern, sedangkan Bayesian Optimization memberikan pendekatan efisien untuk mencari konfigurasi terbaik secara otomatis.

---

## Bab II — Gap Research Analysis

### 2.1 Status Quo: Kekuatan dan Kelemahan Pendekatan Eksisting

| Aspek | Penjelasan |
| ------ | ----------- |
| **Kekuatan** | ARIMA kuat untuk data stasioner, GRU mampu menangani nonlinieritas dan long-term dependency. |
| **Kelemahan** | Banyak penelitian hanya menggunakan parameter default atau grid search manual. |
| **Celah Penelitian** | Belum banyak studi yang menerapkan Bayesian Optimization untuk tuning GRU dalam konteks harga komoditas multi-jenis. |

---

### 2.2 Research Gap (Celah Penelitian)

Sebagian besar penelitian sebelumnya fokus pada satu komoditas atau tanpa optimasi hyperparameter.  
Penelitian ini mengisi celah tersebut dengan membandingkan tiga model (Naive, ARIMA, GRU) dan menggunakan Bayesian Optimization untuk meningkatkan akurasi GRU terhadap tiga komoditas utama (emas, nikel, batubara).

---

### 2.3 Novelty Penelitian

| Aspek | Kebaruan |
| ------ | -------- |
| **Metode** | Integrasi GRU dengan Bayesian Optimization sebagai metode optimasi otomatis. |
| **Objek Data** | Penggunaan dataset World Bank Pink Sheet yang memuat tiga komoditas global. |
| **Evaluasi** | Perbandingan kuantitatif antara model klasik, baseline, dan deep learning. |

---

### 2.4 Referensi Kunci (Format IEEE)

| No | Referensi | Dukungan |
| -- | ---------- | -------- |
| [1] | X. Zhang, Y. Liu, “Hybrid ARIMA–GRU Model for Commodity Price Forecasting,” *J. Econ. Comput.*, vol. 12, no. 3, pp. 45–56, 2024. | Studi relevan yang membandingkan ARIMA dan GRU. |
| [2] | A. Rahman, “Bayesian Optimization for Deep Learning Hyperparameter Tuning,” *IEEE Trans. Neural Netw.*, vol. 36, no. 4, pp. 512–520, 2025. | Menunjukkan efektivitas Bayesian Optimization. |
| [3] | L. Wang et al., “Performance Comparison of Forecasting Models on Metal Prices,” *Appl. Soft Comput.*, vol. 142, 2025. | Studi pembanding untuk data nikel dan emas. |
| [4] | World Bank, “Commodity Price Data (Pink Sheet),” *World Development Indicators*, 2025. | Sumber data utama penelitian. |
| [5] | K. Cho et al., “Learning Phrase Representations using GRU-RNN Encoder–Decoder,” *EMNLP*, 2014. | Landasan teoritis GRU. |

---

### 2.5 Kerangka Berpikir (Mind Map)

<img width="2525" height="3006" alt="Mindmap png" src="https://github.com/user-attachments/assets/e7093fee-251d-4452-b1a3-ddd4a5b5f566" />


---

## Bab III — Methodology

### 3.1 Desain Penelitian

Jenis penelitian: **Eksperimen Kuantitatif**  
Pendekatan: **Supervised Learning (Time Series Forecasting)**

---

### 3.2 Alur Penelitian (Flowchart)

<img width="162" height="982" alt="Flowchart png" src="https://github.com/user-attachments/assets/91be0835-fb48-4a95-b454-be660a260b3e" />


---

### 3.3 Dataset dan Preprocessing

- **Sumber Data:** World Bank – Pink Sheet Commodity Prices  
- **Komoditas:** Emas, Nikel, Batubara  
- **Frekuensi:** Bulanan  
- **Langkah-langkah:**
  - Menangani missing value dan outlier  
  - Normalisasi data menggunakan MinMaxScaler  
  - Pembagian data menjadi train (80%) dan test (20%)  
  - Uji stasioneritas (ADF Test) untuk ARIMA  
  - Pembuatan sequence data untuk GRU  

---

### 3.4 Model Development

1. **Naive Forecast (Baseline)**  
   - Prediksi nilai selanjutnya = nilai terakhir sebelumnya.  

2. **ARIMA Model**  
   - Parameter (p, d, q) ditentukan menggunakan ACF–PACF dan validasi grid.  

3. **GRU Model + Bayesian Optimization**  
   - Optimasi hyperparameter (hidden units, learning rate, batch size, epoch) menggunakan Bayesian Optimization.  

4. **Evaluasi Model**  
   - Metrik evaluasi: RMSE, MAE dan DA.  
   - Visualisasi hasil prediksi vs aktual.  

---

### 3.5 Tools dan Environment

| Komponen | Keterangan |
| -------- | --------- |
| Bahasa | Python |
| Platform | Google Colab / Jupyter Notebook |
| Library | pandas, numpy, scikit-learn, statsmodels, keras, tensorflow, bayesian-optimization |
| Dataset | [World Bank Pink Sheet Commodity Prices](https://www.worldbank.org/en/research/commodity-markets) |

---

### 3.6 Ekspektasi Hasil

- Bayesian Optimization menghasilkan peningkatan akurasi GRU dibanding konfigurasi default.  
- GRU outperform ARIMA dan Naive Forecast dalam memprediksi harga komoditas nonlinier.  
- Model GRU dengan parameter optimal memberikan RMSE lebih rendah ≥10% dibanding model klasik.  

---
