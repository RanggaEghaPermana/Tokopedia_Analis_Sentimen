# Tokopedia Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-4.x-ffd343?logo=huggingface&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Git LFS](https://img.shields.io/badge/Git%20LFS-enabled-brightgreen)

Analisis sentimen ulasan Tokopedia. Repositori berisi **notebook eksplorasi**, **dataset contoh**, serta **aplikasi (Streamlit) dengan model IndoBERT** yang disimpan lokal menggunakan **Git LFS**.

---

## Fitur

- **Klasifikasi Sentimen (3 kelas)**  
  *Positive*, *Neutral*, *Negative* dengan skor keyakinan.

- **Aplikasi siap jalan** (`sentimen_app-token_indobert/app.py`)  
  Memuat tokenizer & bobot model dari folder lokal `indoBERT_tokped/`. Termasuk aset animasi *lottie*.

- **Notebook analisis** (`notebooks/tokopedia.ipynb`)  
  Eksplorasi/persiapan data. *(Output notebook disarankan dibersihkan sebelum commit).*

- **Dataset contoh** (`data/…`)  
  - `dataset_review_tokped.csv`  
  - `dataset_review_tokped_cleaned.csv`  
  - `dataset_review_tokped_labelled.csv`  
  Seluruhnya dikelola via **Git LFS**.

---

## Struktur

