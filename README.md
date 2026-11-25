# TrustX

A browser extension that analyzes and flags misinformation on Twitter (X) in real time.

## How it works

1. The Chrome extension (`content.js`, `popup.html`, `popup (1).js`) reads tweet text on the page.
2. The text is sent to a Python backend (`app.py`) that classifies it.
3. The result is shown to the user in the extension popup and flagged on the page.

## Models

The ML pipelines in this repo use text features from **FastText**, combined with:

- **XGBoost** (`fasttext_xgboost_hybrid_pipeline.py`)
- **TabPFN** (`tabpfn_fasttext_pipeline.py`)



## Project structure

| File | Purpose |
|------|---------|
| `app.py` | Backend that serves predictions to the extension |
| `content.js` | Content script that reads tweets on the page |
| `popup.html`, `style.css` | Extension popup UI |
| `manifest.json` | Chrome extension configuration |
| `clean_fasttext_dataset.py` | Dataset cleaning and preprocessing |
| `fasttext_xgboost_hybrid_pipeline.py` | FastText + XGBoost training pipeline |
| `tabpfn_fasttext_pipeline.py` | FastText + TabPFN training pipeline |
| `clean_dataset.csv`, `final_fasttext_features_dataset.csv`, `train.txt` | Cleaned data and training files |

## Setup

### 1. Run the backend
```bash
pip install flask fasttext xgboost tabpfn pandas scikit-learn
python app.py

```

### 2. Load the extension in Chrome
1. Open `chrome://extensions`
2. Turn on **Developer mode**
3. Click **Load unpacked** and select this project folder
4. Open Twitter/X and click the TrustX icon

## Team

Built by Varun Naik, Sravya Ullamgunta, Kartheev, Amulya.

## Tech stack

Python, JavaScript, FastText, XGBoost, TabPFN, Chrome Extensions (Manifest)
