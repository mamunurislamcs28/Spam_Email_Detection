"""
utils.py - Utilities: Dataset, Prediction, Database, Logging
"""

import os
import sqlite3
import logging
import logging.handlers
import urllib.request
import zipfile
import datetime
import numpy as np
import pandas as pd
from typing import List, Optional

DATASET_PATH      = "SMSSpamCollection"
SPAMASSASSIN_PATH = "spam_assassin.csv"
DATASET_URL       = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DB_PATH           = "spam_detector.db"
LOGS_DIR          = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)


# ── Logging Setup ───────────────────────────────────────────────────────────────

def setup_logging(level=logging.INFO) -> logging.Logger:
    fmt = logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] %(name)s — %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    root = logging.getLogger()
    root.setLevel(level)
    if root.handlers:
        return root
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)
    fh = logging.handlers.RotatingFileHandler(
        os.path.join(LOGS_DIR, 'spam_detector.log'),
        maxBytes=5 * 1024 * 1024, backupCount=3
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)
    return root

logger = logging.getLogger(__name__)


# ── Dataset ─────────────────────────────────────────────────────────────────────

def ensure_dataset() -> dict:
    """
    Check both dataset files exist. Auto-downloads UCI SMS if missing.
    SpamAssassin must be manually downloaded from Kaggle (requires login).

    Returns:
        dict with keys 'sms' and 'spamassassin' pointing to file paths.
        Either value may be None if that file is unavailable.
    """
    result = {'sms': None, 'spamassassin': None}

    # ── UCI SMS Spam Collection ─────────────────────────────────────────
    if os.path.exists(DATASET_PATH):
        logger.info("SMS dataset found: %s", DATASET_PATH)
        result['sms'] = DATASET_PATH
    else:
        logger.info("SMS dataset not found — attempting auto-download...")
        zip_path = "smsspamcollection.zip"
        try:
            urllib.request.urlretrieve(DATASET_URL, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(".")
            os.remove(zip_path)
            logger.info("SMS dataset ready: %s", DATASET_PATH)
            result['sms'] = DATASET_PATH
        except Exception as e:
            logger.error("Auto-download failed: %s", e)
            print("\n" + "="*60)
            print("  SMS DATASET — MANUAL SETUP REQUIRED")
            print("="*60)
            print("  1. Visit: https://archive.ics.uci.edu/dataset/228")
            print("  2. Download the ZIP")
            print("  3. Extract 'SMSSpamCollection' into this folder")
            print("="*60 + "\n")

    # ── SpamAssassin ────────────────────────────────────────────────────
    if os.path.exists(SPAMASSASSIN_PATH):
        logger.info("SpamAssassin dataset found: %s", SPAMASSASSIN_PATH)
        result['spamassassin'] = SPAMASSASSIN_PATH
    else:
        print("\n" + "="*60)
        print("  SPAMASSASSIN DATASET — MANUAL DOWNLOAD REQUIRED")
        print("="*60)
        print("  1. Visit: https://www.kaggle.com/datasets/ganiyuolalekan/")
        print("            spam-assassin-email-classification-dataset")
        print("  2. Download spam_assassin.csv")
        print("  3. Place it in this folder as: spam_assassin.csv")
        print("  Training will continue with SMS data only if missing.")
        print("="*60 + "\n")

    if not result['sms'] and not result['spamassassin']:
        raise FileNotFoundError(
            "No datasets found. At least one of SMSSpamCollection or "
            "spam_assassin.csv must be present."
        )

    return result


# ── SQLite Database ─────────────────────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            message     TEXT NOT NULL,
            clean_text  TEXT,
            label       TEXT NOT NULL,
            spam_prob   REAL NOT NULL,
            ham_prob    REAL NOT NULL,
            confidence  REAL NOT NULL,
            source      TEXT DEFAULT 'cli'
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id  INTEGER REFERENCES predictions(id),
            timestamp      TEXT NOT NULL,
            correct        INTEGER NOT NULL,
            true_label     TEXT
        )
    """)
    conn.commit()
    conn.close()
    logger.debug("Database initialized: %s", DB_PATH)


def save_prediction(message, clean_text, label, spam_prob, ham_prob, confidence, source='cli') -> int:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO predictions (timestamp, message, clean_text, label, spam_prob, ham_prob, confidence, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.datetime.utcnow().isoformat(),
        message, clean_text, label,
        float(spam_prob), float(ham_prob), float(confidence), source
    ))
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def save_feedback(prediction_id: int, correct: bool, true_label: Optional[str] = None):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO feedback (prediction_id, timestamp, correct, true_label)
        VALUES (?, ?, ?, ?)
    """, (prediction_id, datetime.datetime.utcnow().isoformat(),
          int(correct), true_label))
    conn.commit()
    conn.close()


def get_prediction_history(limit: int = 50) -> List[dict]:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()
    cur.execute("""
        SELECT p.*, f.correct, f.true_label
        FROM predictions p
        LEFT JOIN feedback f ON f.prediction_id = p.id
        ORDER BY p.id DESC LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_stats() -> dict:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM predictions")
    total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM predictions WHERE label='SPAM'")
    spam_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM feedback WHERE correct=1")
    correct_fb = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM feedback")
    total_fb   = cur.fetchone()[0]
    conn.close()
    return {
        'total_predictions': total,
        'spam_predictions' : spam_count,
        'ham_predictions'  : total - spam_count,
        'feedback_given'   : total_fb,
        'feedback_accuracy': round(correct_fb / total_fb * 100, 1) if total_fb else None,
    }


# ── Prediction ──────────────────────────────────────────────────────────────────

def predict_message(message: str, model, word_vec, char_vec,
                    threshold: float = 0.5, save_to_db: bool = True,
                    source: str = 'cli') -> dict:
    from preprocess import clean_text, extract_custom_features
    from model import build_feature_matrix, CUSTOM_FEATURE_COLS
    import pandas as pd

    if not message or not message.strip():
        raise ValueError("Message cannot be empty.")

    cleaned  = clean_text(message)
    features = extract_custom_features(message)

    row = {'clean_message': cleaned}
    row.update(features)
    df_row = pd.DataFrame([row])

    for col in CUSTOM_FEATURE_COLS:
        if col not in df_row.columns:
            df_row[col] = 0.0

    X = build_feature_matrix(df_row, word_vec, char_vec, fit=False)

    probs      = model.predict_proba(X)[0]
    spam_prob  = float(probs[1])
    ham_prob   = float(probs[0])
    label      = 'SPAM' if spam_prob >= threshold else 'HAM'
    confidence = max(spam_prob, ham_prob) * 100

    result = {
        'label'      : label,
        'confidence' : round(confidence, 2),
        'spam_prob'  : round(spam_prob, 4),
        'ham_prob'   : round(ham_prob, 4),
        'clean_text' : cleaned,
        'features'   : {k: round(v, 3) for k, v in features.items()},
        'db_id'      : None,
    }

    if save_to_db:
        db_id = save_prediction(
            message, cleaned, label,
            spam_prob, ham_prob, confidence, source
        )
        result['db_id'] = db_id

    return result


def batch_predict(messages: List[str], model, word_vec, char_vec,
                  threshold: float = 0.5) -> pd.DataFrame:
    from preprocess import clean_text, extract_custom_features
    from model import build_feature_matrix, CUSTOM_FEATURE_COLS

    rows = []
    for msg in messages:
        cleaned  = clean_text(str(msg))
        features = extract_custom_features(str(msg))
        row = {'clean_message': cleaned}
        row.update(features)
        rows.append(row)

    df_batch = pd.DataFrame(rows)
    for col in CUSTOM_FEATURE_COLS:
        if col not in df_batch.columns:
            df_batch[col] = 0.0

    X = build_feature_matrix(df_batch, word_vec, char_vec, fit=False)
    probs  = model.predict_proba(X)
    labels = ['SPAM' if p[1] >= threshold else 'HAM' for p in probs]

    return pd.DataFrame({
        'message'   : messages,
        'label'     : labels,
        'spam_prob' : [round(float(p[1]), 4) for p in probs],
        'ham_prob'  : [round(float(p[0]), 4) for p in probs],
        'confidence': [round(max(float(p[0]),float(p[1]))*100, 2) for p in probs],
    })


# ── Display ─────────────────────────────────────────────────────────────────────

def print_prediction_result(result: dict, original: str):
    icon = "SPAM" if result['label'] == 'SPAM' else "HAM"
    print("\n" + "="*60)
    print(f"  VERDICT:     {icon}")
    print(f"  Confidence:  {result['confidence']:.2f}%")
    print(f"  Spam Prob:   {result['spam_prob']:.4f}")
    print(f"  Ham Prob:    {result['ham_prob']:.4f}")
    print("-"*60)
    print(f"  Original:    {original[:80]}{'...' if len(original) > 80 else ''}")
    print(f"  Cleaned:     {result['clean_text'][:80]}{'...' if len(result['clean_text']) > 80 else ''}")
    print("-"*60)
    print("  Spam Signal Features:")
    feats = result.get('features', {})
    for k, v in feats.items():
        if v > 0:
            print(f"    {k:<22} = {v}")
    print("="*60 + "\n")


def print_banner():
    print("""
  +----------------------------------------------------------+
  |     EMAIL SPAM DETECTOR  v2.0                           |
  |   Ensemble ML  Advanced NLP  SQLite  Flask REST API     |
  +----------------------------------------------------------+
    """)







#
#
# """
# utils.py - Utility Functions
# """
#
# import os
# import urllib.request
# import zipfile
#
# SMS_DATASET_URL      = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
# DATASET_FILENAME     = "SMSSpamCollection"
# LARGE_DATASET_FILENAME = "spam_Emails_data.csv"   # 190K dataset from Kaggle
#
#
# def download_dataset(save_dir: str = ".") -> str:
#     filepath = os.path.join(save_dir, DATASET_FILENAME)
#     if os.path.exists(filepath):
#         print(f"Dataset already exists at '{filepath}'")
#         return filepath
#
#     print("Downloading SMS Spam Collection dataset...")
#     zip_path = os.path.join(save_dir, "smsspamcollection.zip")
#     try:
#         urllib.request.urlretrieve(SMS_DATASET_URL, zip_path)
#         with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#             zip_ref.extractall(save_dir)
#         os.remove(zip_path)
#         print(f"Dataset extracted to '{filepath}'")
#     except Exception as e:
#         print(f"Auto-download failed: {e}")
#         print("\nManual download instructions:")
#         print("1. Visit: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection")
#         print("2. Download 'smsspamcollection.zip'")
#         print("3. Extract 'SMSSpamCollection' to the project root directory")
#         raise
#     return filepath
#
#
# def create_sample_dataset(save_dir: str = ".") -> str:
#     filepath = os.path.join(save_dir, DATASET_FILENAME)
#     sample_data = [
#         ("spam", "CONGRATULATIONS! You've WON 1000! Call 09061702893 NOW to claim your prize!"),
#         ("spam", "FREE entry in 2 a weekly comp to win FA Cup final tickets! Text FA to 87121"),
#         ("spam", "URGENT! Your mobile number has been awarded a 2000 prize. CALL 09061790674"),
#         ("spam", "Winner! You have been selected to receive a 900 prize reward."),
#         ("spam", "Claim your 300 shopping spree NOW. Valid only for 2 hours. Text SHOP to 89555"),
#         ("spam", "Your account has been suspended. Login at http://secure.update-account.com"),
#         ("spam", "Limited time offer: Get 50% off all purchases! Use code SAVE50 at checkout."),
#         ("spam", "Your PayPal account will be suspended unless you verify now"),
#         ("ham", "Hey, what time are we meeting tonight?"),
#         ("ham", "Can you pick up some milk on your way home?"),
#         ("ham", "Just finished work, heading home now. Want to grab dinner?"),
#         ("ham", "Don't forget about the meeting at 3pm tomorrow"),
#         ("ham", "Are you free this weekend? We should catch up"),
#         ("ham", "I'll be there in about 20 minutes. Traffic is bad"),
#         ("ham", "Did you see the game last night? Incredible finish!"),
#         ("ham", "Mom says dinner is at 7. Will you make it?"),
#     ]
#     with open(filepath, 'w', encoding='utf-8') as f:
#         for label, message in sample_data:
#             f.write(f"{label}\t{message.replace(chr(9), ' ')}\n")
#     print(f"Sample dataset created at '{filepath}' ({len(sample_data)} messages)")
#     print("Note: Using a small sample. Download the full dataset for production.")
#     return filepath
#
#
# def ensure_dataset_exists(filepath: str = DATASET_FILENAME) -> str | dict:
#     """
#     Ensure at least one dataset exists. Returns either:
#       - A string path  if only one dataset is available
#       - A dict {'sms': path, '190k': path}  if both are available
#
#     The 190K dataset must be manually downloaded from Kaggle (requires login).
#     The SMS dataset will be auto-downloaded from UCI if missing.
#     """
#     sms_path   = DATASET_FILENAME
#     large_path = LARGE_DATASET_FILENAME
#
#     has_sms   = os.path.exists(sms_path)
#     has_large = os.path.exists(large_path)
#
#     # If 190K dataset exists, always mention it
#     if has_large:
#         print(f"190K dataset found: {large_path} ({_size_mb(large_path):.0f} MB)")
#
#     # Ensure SMS dataset
#     if not has_sms:
#         print(f"SMS dataset not found at '{sms_path}'")
#         print("Attempting to auto-download...")
#         try:
#             download_dataset()
#             has_sms = True
#         except Exception:
#             if not has_large:
#                 print("\nFalling back to sample dataset for demonstration...")
#                 create_sample_dataset()
#                 has_sms = True
#
#     if has_sms:
#         print(f"SMS dataset found: {sms_path}")
#
#     # Return dict if both present, string if only one
#     if has_sms and has_large:
#         print("\nBoth datasets found — training on combined data.")
#         return {'sms': sms_path, '190k': large_path}
#     elif has_large:
#         return {'sms': None, '190k': large_path}
#     else:
#         return sms_path   # Legacy single-path string
#
#
# def _size_mb(filepath: str) -> float:
#     return os.path.getsize(filepath) / (1024 * 1024)
#
#
# def print_banner():
#     banner = """
# +----------------------------------------------------------+
# |          EMAIL SPAM DETECTOR                             |
# |          Built with scikit-learn + TF-IDF + NB           |
# +----------------------------------------------------------+
#     """
#     print(banner)
#
#
# def print_prediction_result(result: dict, message: str):
#     label      = result['label']
#     confidence = result['confidence'] * 100
#     spam_prob  = result['spam_probability'] * 100
#     ham_prob   = result['ham_probability'] * 100
#     is_spam    = label == 'SPAM'
#     border     = "!" * 60 if is_spam else "-" * 60
#
#     print(f"\n{border}")
#     print(f"  PREDICTION: {label}")
#     print(f"{border}")
#     print(f"  Message   : {message[:80]}{'...' if len(message) > 80 else ''}")
#     print(f"  Confidence: {confidence:.1f}%")
#     print(f"  Spam Prob : {spam_prob:.1f}%")
#     print(f"  Ham Prob  : {ham_prob:.1f}%")
#     print(f"{border}\n")
#
#
# def print_metrics_summary(metrics: dict, model_name: str = "Model"):
#     print(f"\n{'='*40}")
#     print(f"  {model_name} - Final Metrics Summary")
#     print(f"{'='*40}")
#     for metric, value in metrics.items():
#         bar_len = int(value * 30)
#         bar = '#' * bar_len + '.' * (30 - bar_len)
#         print(f"  {metric.capitalize():12s} [{bar}] {value:.4f}")
#     print(f"{'='*40}\n")