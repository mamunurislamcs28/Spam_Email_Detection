"""
preprocess.py - Advanced NLP Preprocessing Pipeline
"""

import re
import string
import logging
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

def _ensure_nltk():
    for resource, path in [('stopwords', 'corpora/stopwords'),
                            ('punkt_tab', 'tokenizers/punkt_tab')]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)

_ensure_nltk()
STOP_WORDS = set(stopwords.words('english'))

try:
    import spacy
    _nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    USE_SPACY = True
    logger.info("spaCy loaded — using lemmatization.")
except Exception:
    from nltk.stem import PorterStemmer
    _stemmer = PorterStemmer()
    USE_SPACY = False
    logger.warning("spaCy not available — falling back to NLTK stemmer.")


def extract_custom_features(text: str) -> dict:
    if not isinstance(text, str) or not text:
        return {k: 0.0 for k in [
            'num_exclamations', 'num_question_marks', 'num_currency',
            'num_urls', 'num_phone_numbers', 'caps_ratio', 'digit_ratio',
            'msg_length', 'num_words', 'avg_word_length', 'has_html',
            'repeated_chars'
        ]}
    words = text.split()
    alpha_chars = [c for c in text if c.isalpha()]
    digit_chars = [c for c in text if c.isdigit()]
    return {
        'num_exclamations' : float(text.count('!')),
        'num_question_marks': float(text.count('?')),
        'num_currency'     : float(len(re.findall(r'[\$£€¥]', text))),
        'num_urls'         : float(len(re.findall(r'http\S+|www\.\S+', text, re.I))),
        'num_phone_numbers': float(len(re.findall(r'\b\d{10,11}\b|\b\d{3}[\s\-]\d{4}\b', text))),
        'caps_ratio'       : sum(1 for c in alpha_chars if c.isupper()) / max(len(alpha_chars), 1),
        'digit_ratio'      : len(digit_chars) / max(len(text), 1),
        'msg_length'       : float(len(text)),
        'num_words'        : float(len(words)),
        'avg_word_length'  : np.mean([len(w) for w in words]) if words else 0.0,
        'has_html'         : float(bool(re.search(r'<[^>]+>', text))),
        'repeated_chars'   : float(len(re.findall(r'(.)\1{2,}', text))),
    }


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\b\d{10,}\b', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    if USE_SPACY:
        doc = _nlp(' '.join(tokens))
        tokens = [
            token.lemma_
            for token in doc
            if token.lemma_ not in STOP_WORDS
            and not token.is_stop
            and len(token.lemma_) > 1
            and token.lemma_.isalpha()
        ]
    else:
        tokens = [
            _stemmer.stem(w)
            for w in tokens
            if w not in STOP_WORDS and len(w) > 1
        ]
    return ' '.join(tokens)


# ── Dataset Loader (UPDATED — loads and merges both datasets) ──────────────────

def _load_sms(filepath: str) -> pd.DataFrame:
    """Load UCI SMS Spam Collection (tab-separated, label + message)."""
    df = pd.read_csv(
        filepath, sep='\t', header=None,
        names=['label', 'message'], encoding='latin-1'
    )
    df['source'] = 'sms'
    return df


def _load_spamassassin(filepath: str) -> pd.DataFrame:
    """
    Load SpamAssassin CSV (text + target columns).
    Normalizes column names to match the rest of the pipeline.
    Truncates emails to first 2,000 chars to keep feature extraction fast —
    full email headers are very long but the signal is concentrated early.
    """
    df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')

    # Normalize column names (lowercase, strip spaces)
    df.columns = df.columns.str.lower().str.strip()

    if 'text' not in df.columns or 'target' not in df.columns:
        raise ValueError(
            f"spam_assassin.csv must have 'text' and 'target' columns. "
            f"Found: {list(df.columns)}"
        )

    df = df.rename(columns={'text': 'message'})
    df['label'] = df['target'].map({0: 'ham', 1: 'spam'})
    df['message'] = df['message'].astype(str).str[:2000]
    df['source'] = 'spamassassin'
    return df[['label', 'message', 'source']]



def load_and_preprocess_data(dataset_paths) -> pd.DataFrame:
    """
    Load one or both datasets, merge them, and run the full preprocessing pipeline.

    Args:
        dataset_paths: either a string (single file path, legacy support)
                       or a dict {'sms': path_or_None, 'spamassassin': path_or_None}
                       as returned by utils.ensure_dataset()

    Returns:
        DataFrame with columns: label, message, source, label_num, clean_message,
        + all 12 custom feature columns
    """
    frames = []

    # ── Accept legacy single-path string call ───────────────────────────
    if isinstance(dataset_paths, str):
        dataset_paths = {'sms': dataset_paths, 'spamassassin': None}

    # ── Load SMS ─────────────────────────────────────────────────────────
    if dataset_paths.get('sms'):
        logger.info("Loading SMS dataset from %s", dataset_paths['sms'])
        try:
            sms_df = _load_sms(dataset_paths['sms'])
            logger.info("  SMS: %d records loaded", len(sms_df))
            frames.append(sms_df)
        except Exception as e:
            logger.error("Failed to load SMS dataset: %s", e)

    # ── Load SpamAssassin ─────────────────────────────────────────────────
    if dataset_paths.get('spamassassin'):
        logger.info("Loading SpamAssassin dataset from %s", dataset_paths['spamassassin'])
        try:
            sa_df = _load_spamassassin(dataset_paths['spamassassin'])
            logger.info("  SpamAssassin: %d records loaded", len(sa_df))
            frames.append(sa_df)
        except Exception as e:
            logger.error("Failed to load SpamAssassin dataset: %s", e)

    if not frames:
        raise RuntimeError("No data could be loaded from any dataset.")

    # ── Merge ─────────────────────────────────────────────────────────────
    df = pd.concat(frames, ignore_index=True)
    logger.info("Combined: %d total records before cleaning", len(df))

    # ── Clean ─────────────────────────────────────────────────────────────
    df.drop_duplicates(subset=['message'], inplace=True)
    df.dropna(subset=['label', 'message'], inplace=True)
    df = df[df['label'].isin(['ham', 'spam'])].copy()
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

    logger.info("After dedup/clean: %d records", len(df))
    logger.info("Label distribution:\n%s", df['label'].value_counts().to_string())
    logger.info("Source distribution:\n%s", df['source'].value_counts().to_string())

    # ── Feature extraction ────────────────────────────────────────────────
    logger.info("Extracting custom features...")
    custom_feats = df['message'].apply(extract_custom_features)
    feat_df = pd.DataFrame(list(custom_feats))
    df = pd.concat([df.reset_index(drop=True), feat_df], axis=1)

    logger.info("Cleaning and lemmatizing text...")
    df['clean_message'] = df['message'].apply(clean_text)

    logger.info("Preprocessing complete. %d records ready.", len(df))
    return df










# """
# preprocess.py - Text Preprocessing Pipeline
# """
#
# import re
# import string
# import nltk
# import pandas as pd
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
#
# def download_nltk_data():
#     packages = ['stopwords', 'punkt', 'punkt_tab']
#     for pkg in packages:
#         try:
#             nltk.download(pkg, quiet=True)
#         except Exception as e:
#             print(f"Warning: Could not download NLTK package '{pkg}': {e}")
#
# download_nltk_data()
#
# stemmer = PorterStemmer()
# STOP_WORDS = set(stopwords.words('english'))
#
#
# def clean_text(text: str) -> str:
#     if not isinstance(text, str):
#         return ""
#     text = text.lower()
#     text = re.sub(r'http\S+|www\.\S+', '', text)
#     text = re.sub(r'\S+@\S+', '', text)
#     text = re.sub(r'\d+', '', text)
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text
#
#
# def remove_stopwords(text: str) -> str:
#     words = text.split()
#     return ' '.join(w for w in words if w not in STOP_WORDS)
#
#
# def stem_text(text: str) -> str:
#     words = text.split()
#     return ' '.join(stemmer.stem(w) for w in words)
#
#
# def preprocess_text(text: str, use_stemming: bool = True) -> str:
#     text = clean_text(text)
#     text = remove_stopwords(text)
#     if use_stemming:
#         text = stem_text(text)
#     return text
#
#
# def _load_sms(filepath: str) -> pd.DataFrame:
#     """
#     Load UCI SMS Spam Collection.
#     Format: tab-separated, no header, columns = [label, message]
#     Labels are already lowercase: 'ham' / 'spam'
#     """
#     df = pd.read_csv(
#         filepath, sep='\t',
#         names=['label', 'message'],
#         encoding='latin-1'
#     )
#     df['source'] = 'sms'
#     return df
#
#
# def _load_190k(filepath: str) -> pd.DataFrame:
#     """
#     Load the 190K Spam/Ham Email Dataset.
#     Format: CSV with header, columns = [label, text]
#     Labels are Title Case: 'Ham' / 'Spam' — normalised to lowercase here.
#     Text column renamed to 'message' to match the rest of the pipeline.
#     Emails are truncated to 3,000 chars — signal is concentrated early
#     and full emails slow preprocessing significantly at 190K rows.
#     """
#     df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')
#     df.columns = df.columns.str.lower().str.strip()
#
#     if 'label' not in df.columns or 'text' not in df.columns:
#         raise ValueError(
#             f"Expected columns 'label' and 'text' in {filepath}. "
#             f"Found: {list(df.columns)}"
#         )
#
#     df = df.rename(columns={'text': 'message'})
#
#     # Normalise Title Case labels -> lowercase
#     df['label'] = df['label'].str.lower().str.strip()
#
#     # Truncate long emails for speed
#     df['message'] = df['message'].astype(str).str[:3000]
#     df['source'] = '190k'
#     return df[['label', 'message', 'source']]
#
#
# def load_and_preprocess_dataset(filepath_or_dict) -> pd.DataFrame:
#     """
#     Load one or both datasets, merge, and preprocess.
#
#     Args:
#         filepath_or_dict:
#             - A string path  → loads that single file (auto-detects format)
#             - A dict         → {'sms': path_or_None, '190k': path_or_None}
#
#     Returns:
#         DataFrame with columns:
#             label, message, source, cleaned_message, label_encoded
#     """
#     frames = []
#
#     # ── Normalise input to dict ───────────────────────────────────────────
#     if isinstance(filepath_or_dict, str):
#         # Single path — auto-detect which dataset it is
#         path = filepath_or_dict
#         if path.endswith('.csv'):
#             filepath_or_dict = {'sms': None, '190k': path}
#         else:
#             filepath_or_dict = {'sms': path, '190k': None}
#
#     # ── Load SMS ──────────────────────────────────────────────────────────
#     sms_path = filepath_or_dict.get('sms')
#     if sms_path:
#         print(f"Loading SMS dataset from '{sms_path}'...")
#         try:
#             sms_df = _load_sms(sms_path)
#             print(f"  SMS: {len(sms_df):,} records")
#             frames.append(sms_df)
#         except Exception as e:
#             print(f"  Warning: could not load SMS dataset: {e}")
#
#     # ── Load 190K ─────────────────────────────────────────────────────────
#     large_path = filepath_or_dict.get('190k')
#     if large_path:
#         print(f"Loading 190K dataset from '{large_path}'...")
#         try:
#             large_df = _load_190k(large_path)
#             print(f"  190K: {len(large_df):,} records")
#             frames.append(large_df)
#         except Exception as e:
#             print(f"  Warning: could not load 190K dataset: {e}")
#
#     if not frames:
#         raise RuntimeError("No data loaded. Check your dataset paths.")
#
#     # ── Merge + clean ─────────────────────────────────────────────────────
#     df = pd.concat(frames, ignore_index=True)
#     df.dropna(subset=['label', 'message'], inplace=True)
#     df.drop_duplicates(subset=['message'], inplace=True)
#
#     # Keep only valid labels
#     df = df[df['label'].isin(['ham', 'spam'])].copy()
#
#     print(f"\nCombined: {len(df):,} records after dedup")
#     print(f"Distribution:\n{df['label'].value_counts().to_string()}")
#     if 'source' in df.columns:
#         print(f"By source:\n{df['source'].value_counts().to_string()}")
#
#     # ── Encode labels ─────────────────────────────────────────────────────
#     df['label_encoded'] = df['label'].map({'ham': 0, 'spam': 1})
#
#     # ── Preprocess text ───────────────────────────────────────────────────
#     total = len(df)
#     print(f"\nPreprocessing {total:,} messages (cleaning, stopwords, stemming)...")
#     print("  This may take a few minutes for the 190K dataset...")
#     df['cleaned_message'] = df['message'].apply(preprocess_text)
#
#     print("Preprocessing complete!\n")
#     return df
#
#
# if __name__ == "__main__":
#     sample_texts = [
#         "CONGRATULATIONS! You've WON a FREE iPhone! Click http://win.com/prize now!!!",
#         "Hey, are we still on for lunch tomorrow at 1pm?",
#     ]
#     print("=== Preprocessing Pipeline Demo ===\n")
#     for text in sample_texts:
#         result = preprocess_text(text)
#         print(f"Original : {text}")
#         print(f"Processed: {result}")
#         print("-" * 60)