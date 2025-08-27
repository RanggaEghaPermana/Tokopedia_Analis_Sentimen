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

- **Klasifikasi Sentimen (3 kelas)** — *Positive*, *Neutral*, *Negative* + skor keyakinan.  
- **Aplikasi siap jalan** — `sentimen_app-token_indobert/app.py` memuat tokenizer & bobot model dari folder lokal `indoBERT_tokped/`.  
- **Notebook analisis** — `notebooks/tokopedia.ipynb` untuk eksplorasi/persiapan data.  
- **Dataset contoh** — `data/…` (ditrack via **Git LFS**).

---

## Struktur


> Catatan: file CSV/model dikelola oleh **Git LFS**. Setelah `git clone`, jalankan `git lfs pull` agar file besarnya terunduh.

---

## Prasyarat

- **Git** & **Git LFS** terpasang.
- **Python 3.x** + **pip**.
- (Opsional) **virtual environment** agar dependensi rapi.

---

## Instalasi (Cepat)

### Windows (PowerShell)
```powershell
# 1) Clone dan unduh file LFS
git clone https://github.com/RanggaEghaPermana/Tokopedia_Analis_Sentimen.git
cd Tokopedia_Analis_Sentimen
git lfs install
git lfs pull

# 2) Buat & aktifkan virtualenv
python -m venv .venv
.venv\Scripts\activate

# 3) Pasang dependensi
# a) Jika repository sudah punya requirements.txt
# pip install -r requirements.txt

# b) Jika belum, hasilkan otomatis dari import aktual
pip install pipreqs
pipreqs . --force
pip install -r requirements.txt

# 4) Jalankan aplikasi (jika app.py menggunakan Streamlit)
pip install streamlit
streamlit run sentimen_app-token_indobert/app.py

# 5) (Opsional) Jalankan notebook
pip install notebook
jupyter notebook
# lalu buka notebooks/tokopedia.ipynb
