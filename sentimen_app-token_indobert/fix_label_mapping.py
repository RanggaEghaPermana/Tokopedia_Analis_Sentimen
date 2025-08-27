# ------------------------------------------------------------------
# 02_fix_label_mapping.py  – jalankan SEKALI setelah ekstrak ZIP
# ------------------------------------------------------------------
from transformers import AutoModelForSequenceClassification
from pathlib import Path

MODEL_DIR = Path(r"C:\Users\Rangg\OneDrive\Dokumen\kuliah\Tugas\NLP\indoBERT_tokped")

mdl = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, local_files_only=True)

mdl.config.id2label = {0: "Positive", 1: "Neutral", 2: "Negative"}
mdl.config.label2id = {v: k for k, v in mdl.config.id2label.items()}
mdl.save_pretrained(MODEL_DIR)
print("✅ Mapping fixed :", mdl.config.id2label)
