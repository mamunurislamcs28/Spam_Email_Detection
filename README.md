# 📧 SpamShield v2 — Advanced Email Spam Detector

A production-grade spam detection system with ensemble ML, advanced NLP,
SQLite persistence, REST API, batch prediction, and a full-featured web UI.

---

## 🆕 What's New vs v1

| Feature                    | v1       | v2                          |
|----------------------------|----------|-----------------------------|
| Model                      | NB or LR | **Ensemble (NB+LR+SVM)**    |
| NLP                        | Stemming | **spaCy Lemmatization**      |
| Features                   | TF-IDF   | **Word + Char TF-IDF + 12 custom features** |
| Hyperparameter tuning       | ✗        | **GridSearchCV**            |
| Cross-validation           | ✗        | **5-fold StratifiedKFold**  |
| Class imbalance handling   | ✗        | **class_weight='balanced'** |
| Metrics                    | 4        | **6 (+ ROC-AUC, AP)**       |
| Plots                      | 4        | **6 (+ ROC/PR curves, CV)** |
| Database                   | ✗        | **SQLite persistence**      |
| Batch prediction           | ✗        | **CSV upload + JSON API**   |
| User feedback              | ✗        | **Feedback collection**     |
| Threshold control          | ✗        | **Adjustable slider**       |
| Logging                    | print()  | **Structured rotating logs** |
| Tests                      | ✗        | **pytest suite**            |
| API auth                   | ✗        | **Optional API key**        |

---

## 📁 Project Structure

```
spam_v2/
├── main.py              ← CLI (train / predict / batch / stats)
├── preprocess.py        ← Advanced NLP pipeline
├── model.py             ← Ensemble training, tuning, evaluation
├── utils.py             ← DB, logging, prediction, batch
├── app.py               ← Flask REST API (6 endpoints)
├── templates/
│   └── index.html       ← Full-featured web UI
├── tests/
│   └── test_pipeline.py ← pytest test suite
├── requirements.txt
├── logs/                ← Rotating log files (auto-created)
├── plots/               ← Diagnostic charts (auto-created)
├── spam_model.pkl       ← Saved ensemble model
├── vectorizers.pkl      ← Saved word + char vectorizers
└── spam_detector.db     ← SQLite database
```

---

## ⚡ Setup

```bash
# 1. Virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install spaCy English model (for lemmatization)
python -m spacy download en_core_web_sm

# 4. Download NLTK data (auto-downloads on first run, or manually):
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt_tab')"
```

---

## 🚀 Running

### Train (ensemble — recommended)
```bash
python main.py --train
```

### Train specific model
```bash
python main.py --train --model svm
python main.py --train --model logistic_regression
python main.py --train --model naive_bayes
```

### Interactive CLI prediction
```bash
python main.py --predict
python main.py --predict --threshold 0.35   # catch more spam
```

### Batch predict from CSV
```bash
python main.py --batch my_emails.csv
# CSV must have a 'message' column
# Output: my_emails_predictions.csv
```

### Show statistics
```bash
python main.py --stats
```

### Run Flask web app
```bash
python app.py
# Open: http://localhost:5000
```

### Optional: enable API key auth
```bash
API_KEY=mysecretkey python app.py
# Then add header: X-API-Key: mysecretkey
```

### Run tests
```bash
pytest tests/ -v
pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## 🌐 API Reference

### POST /predict
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"message": "FREE PRIZE WIN NOW!!!", "threshold": 0.5}'
```
Response:
```json
{
  "label": "SPAM",
  "confidence": 98.72,
  "spam_prob": 0.9872,
  "ham_prob": 0.0128,
  "clean_text": "free prize win",
  "features": { "num_exclamations": 3, "caps_ratio": 0.81, ... },
  "db_id": 42
}
```

### POST /predict/batch (JSON)
```bash
curl -X POST http://localhost:5000/predict/batch \
     -H "Content-Type: application/json" \
     -d '{"messages": ["msg1", "msg2"], "threshold": 0.5}'
```

### POST /predict/batch (CSV upload)
```bash
curl -X POST http://localhost:5000/predict/batch \
     -F "file=@emails.csv" -F "threshold=0.5"
```

### POST /feedback/<id>
```bash
curl -X POST http://localhost:5000/feedback/42 \
     -H "Content-Type: application/json" \
     -d '{"correct": false, "true_label": "ham"}'
```

### GET /history?limit=20
### GET /stats
### GET /health

---

## 📊 Expected Results

| Metric    | Ensemble | SVM   | LR    | NB    |
|-----------|----------|-------|-------|-------|
| Accuracy  | ~98.5%   | ~98%  | ~98%  | ~97%  |
| Precision | ~97%     | ~96%  | ~97%  | ~95%  |
| Recall    | ~97%     | ~96%  | ~95%  | ~93%  |
| F1-Score  | ~97%     | ~96%  | ~96%  | ~94%  |
| ROC-AUC   | ~99.5%   | ~99%  | ~99%  | ~98%  |

---

## 🧠 Architecture: Feature Pipeline

```
Raw Message
    │
    ├──► [Custom Features]  num_exclamations, caps_ratio, num_urls,
    │                       num_currency, has_html, repeated_chars ...
    │                                          ↓ MinMaxScaler
    │
    ├──► [Word TF-IDF]      clean_text → 8000 word/bigram features
    │    ngram(1,2), sublinear_tf
    │
    └──► [Char TF-IDF]      raw_text  → 3000 char 3–5-gram features
         ngram(3,5)         catches: 'fr33', 'w!n', 'c@sh'
                                          ↓
                            hstack([word | char | custom])
                                          ↓
                    VotingClassifier (soft vote, weights=[1,2,2])
                    ├── ComplementNB(alpha=0.1)
                    ├── LogisticRegression(C=tuned, balanced)
                    └── CalibratedLinearSVC(C=0.5, balanced)
```
