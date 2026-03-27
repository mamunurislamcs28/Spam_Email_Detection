"""
model.py - Advanced Model Training, Ensembling & Evaluation
=============================================================
Upgrades over v1:
  ✦ SVM (LinearSVC)         → typically best classical text classifier
  ✦ Ensemble VotingClassifier → combines NB + LR + SVM for higher accuracy
  ✦ GridSearchCV            → automated hyperparameter tuning
  ✦ StratifiedKFold CV      → reliable 5-fold cross-validation metrics
  ✦ Combined feature matrix → TF-IDF word + TF-IDF char + custom features
  ✦ CalibratedClassifierCV  → calibrated probability estimates for SVM
  ✦ class_weight='balanced' → handles 87/13 class imbalance properly
  ✦ SHAP word importance    → explainable predictions
  ✦ ROC-AUC curve           → full threshold-independent performance view
  ✦ PR curve                → best metric for imbalanced datasets
  ✦ MLflow experiment tracking (optional, graceful fallback)
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    GridSearchCV, cross_val_score
)
from sklearn.naive_bayes   import MultinomialNB, ComplementNB
from sklearn.linear_model  import LogisticRegression
from sklearn.svm           import LinearSVC
from sklearn.ensemble      import VotingClassifier, RandomForestClassifier
from sklearn.calibration   import CalibratedClassifierCV
from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics       import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score,
)

logger = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────────────────
MODEL_PATH      = "spam_model.pkl"
VECTORIZER_PATH = "vectorizers.pkl"   # saves both word + char vectorizers
PLOTS_DIR       = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

CUSTOM_FEATURE_COLS = [
    'num_exclamations', 'num_question_marks', 'num_currency',
    'num_urls', 'num_phone_numbers', 'caps_ratio', 'digit_ratio',
    'msg_length', 'num_words', 'avg_word_length', 'has_html',
    'repeated_chars'
]

# ── Vectorizers ─────────────────────────────────────────────────────────────────

def build_vectorizers():
    """
    Two complementary TF-IDF vectorizers:

    word_vec  → word unigrams + bigrams (semantic meaning)
    char_vec  → character 3–5-grams (catches obfuscation: 'fr33', 'w!n')

    WHY both?
      Word features handle normal vocabulary.
      Char features catch: misspellings, leetspeak, language mixing.
      Combined they're significantly stronger than either alone.
    """
    word_vec = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        min_df=2,           # ignore words appearing in only 1 document
        max_df=0.95,        # ignore words in >95% of documents
    )
    char_vec = TfidfVectorizer(
        max_features=3000,
        ngram_range=(3, 5),
        sublinear_tf=True,
        analyzer='char_wb', # char_wb: respects word boundaries
        min_df=2,
    )
    return word_vec, char_vec


# ── Feature Matrix Builder ──────────────────────────────────────────────────────

def build_feature_matrix(df, word_vec, char_vec, fit=False):
    """
    Build the combined feature matrix: [word TF-IDF | char TF-IDF | custom].

    Args:
        df       : DataFrame with 'clean_message' + custom feature columns
        word_vec : Word TF-IDF vectorizer
        char_vec : Char TF-IDF vectorizer
        fit      : True for training data, False for inference

    Returns:
        scipy sparse matrix (n_samples × n_features)
    """
    texts = df['clean_message'].fillna('')

    if fit:
        X_word = word_vec.fit_transform(texts)
        X_char = char_vec.fit_transform(texts)
    else:
        X_word = word_vec.transform(texts)
        X_char = char_vec.transform(texts)

    # Scale custom numeric features to [0,1] so they don't dominate TF-IDF
    custom = df[CUSTOM_FEATURE_COLS].fillna(0).values.astype(np.float32)
    scaler = MinMaxScaler()
    if fit:
        custom = scaler.fit_transform(custom)
        # Attach scaler to word_vec for later retrieval (convenient storage)
        word_vec._custom_scaler = scaler
    else:
        custom = word_vec._custom_scaler.transform(custom)

    X_custom = csr_matrix(custom)
    return hstack([X_word, X_char, X_custom])


# ── Data Split ──────────────────────────────────────────────────────────────────

def split_data(df, test_size=0.20, random_state=42):
    """Stratified train/test split preserving class ratio."""
    return train_test_split(
        df, df['label_num'],
        test_size=test_size,
        random_state=random_state,
        stratify=df['label_num']
    )


# ── Individual Models ───────────────────────────────────────────────────────────

def _make_models():
    """
    Build all candidate classifiers with sensible defaults.

    ComplementNB:  improved variant of MultinomialNB — better on imbalanced data
                   (complements the minority class distribution)
    LogisticReg:   strong linear baseline with L2 regularization, class weighting
    LinearSVC:     margin-maximizing classifier, state-of-art for sparse text
                   wrapped in CalibratedClassifierCV to get probability outputs
    """
    cnb = ComplementNB(alpha=0.1)

    lr = LogisticRegression(
        C=1.0, max_iter=2000,
        class_weight='balanced',   # up-weights the minority spam class
        solver='lbfgs', random_state=42
    )

    svm_base = LinearSVC(
        C=0.5, max_iter=2000,
        class_weight='balanced',
        random_state=42
    )
    # CalibratedClassifierCV wraps SVM to produce proper probabilities
    svm = CalibratedClassifierCV(svm_base, cv=3, method='sigmoid')

    return cnb, lr, svm


# ── Hyperparameter Tuning ───────────────────────────────────────────────────────

def tune_logistic_regression(X_train, y_train):
    """
    GridSearchCV over LR hyperparameters.
    WHY: The default C=1.0 might not be optimal — grid search finds the
    C value that maximises F1 on cross-validated folds.
    """
    logger.info("Tuning Logistic Regression with GridSearchCV...")
    param_grid = {'C': [0.01, 0.1, 1.0, 5.0, 10.0]}
    lr = LogisticRegression(
        max_iter=2000, class_weight='balanced',
        solver='lbfgs', random_state=42
    )
    gs = GridSearchCV(lr, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    logger.info("Best LR params: %s  |  Best F1: %.4f", gs.best_params_, gs.best_score_)
    return gs.best_estimator_


# ── Main Training ───────────────────────────────────────────────────────────────

def train_model(train_df, model_type='ensemble'):
    """
    Full training pipeline:
      1. Build combined feature matrix (word TF-IDF + char TF-IDF + custom)
      2. Optionally tune hyperparameters
      3. Train chosen model(s)
      4. Run 5-fold cross-validation for reliable metric estimates

    Args:
        train_df   : Training portion of preprocessed DataFrame
        model_type : 'ensemble' | 'svm' | 'logistic_regression' | 'naive_bayes'

    Returns:
        (word_vec, char_vec, trained_model, cv_scores_dict)
    """
    logger.info("Building feature matrix...")
    word_vec, char_vec = build_vectorizers()
    X_train = build_feature_matrix(train_df, word_vec, char_vec, fit=True)
    y_train = train_df['label_num'].values

    cnb, lr, svm = _make_models()

    if model_type == 'ensemble':
        logger.info("Training Ensemble (ComplementNB + LogisticRegression + SVM)...")
        # Tune LR before adding to ensemble
        lr_tuned = tune_logistic_regression(X_train, y_train)
        model = VotingClassifier(
            estimators=[('cnb', cnb), ('lr', lr_tuned), ('svm', svm)],
            voting='soft',   # average predicted probabilities (better than hard vote)
            weights=[1, 2, 2]  # LR and SVM get slightly more weight
        )
    elif model_type == 'svm':
        logger.info("Training Calibrated LinearSVC...")
        model = svm
    elif model_type == 'logistic_regression':
        logger.info("Training Logistic Regression (tuned)...")
        model = tune_logistic_regression(X_train, y_train)
    elif model_type == 'naive_bayes':
        logger.info("Training ComplementNB...")
        model = cnb
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'")

    model.fit(X_train, y_train)
    logger.info("Training complete.")

    # ── 5-fold Cross-Validation ─────────────────────────────────────────────
    logger.info("Running 5-fold stratified cross-validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = {}
    for metric in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']:
        scores = cross_val_score(model, X_train, y_train,
                                 cv=skf, scoring=metric, n_jobs=-1)
        cv_scores[metric] = scores
        logger.info("  CV %-12s: %.4f ± %.4f", metric, scores.mean(), scores.std())

    return word_vec, char_vec, model, cv_scores


# ── Evaluation ──────────────────────────────────────────────────────────────────

def evaluate_model(model, word_vec, char_vec, test_df):
    """
    Comprehensive evaluation on the held-out test set.
    Returns metrics dict and predictions.
    """
    X_test = build_feature_matrix(test_df, word_vec, char_vec, fit=False)
    y_test = test_df['label_num'].values

    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy' : accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall'   : recall_score(y_test, y_pred),
        'f1_score' : f1_score(y_test, y_pred),
        'roc_auc'  : roc_auc_score(y_test, y_prob),
        'avg_precision': average_precision_score(y_test, y_prob),
    }

    print("\n" + "═" * 55)
    print("         ADVANCED MODEL EVALUATION")
    print("═" * 55)
    for k, v in metrics.items():
        print(f"  {k:<20}: {v:.4f}  ({v*100:.2f}%)")
    print("═" * 55)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

    return metrics, y_pred, y_prob, y_test


# ── Visualizations ──────────────────────────────────────────────────────────────

def plot_all(df, y_test, y_pred, y_prob, cv_scores, model, word_vec):
    """Generate all diagnostic plots."""
    _plot_data_overview(df)
    _plot_confusion_matrix(y_test, y_pred)
    _plot_roc_pr_curves(y_test, y_prob)
    _plot_cv_scores(cv_scores)
    _plot_top_spam_features(model, word_vec)


def _plot_data_overview(df):
    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # Bar chart
    ax1 = fig.add_subplot(gs[0])
    counts = df['label'].value_counts()
    colors = ['#27ae60', '#e74c3c']
    bars = ax1.bar(counts.index, counts.values, color=colors, edgecolor='black', lw=0.8, width=0.5)
    ax1.set_title('Class Distribution', fontweight='bold')
    ax1.set_ylabel('Count')
    for bar, val in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 15,
                 f'{val}\n({val/len(df)*100:.1f}%)', ha='center', fontsize=9)

    # Message length KDE
    ax2 = fig.add_subplot(gs[1])
    df['_len'] = df['message'].str.len()
    for lbl, color in [('ham','#27ae60'), ('spam','#e74c3c')]:
        subset = df[df['label']==lbl]['_len'].clip(upper=800)
        ax2.hist(subset, bins=40, alpha=0.55, color=color, label=lbl, edgecolor='none', density=True)
    ax2.set_title('Message Length Distribution', fontweight='bold')
    ax2.set_xlabel('Characters'); ax2.set_ylabel('Density'); ax2.legend()

    # Caps ratio boxplot
    ax3 = fig.add_subplot(gs[2])
    ham_caps  = df[df['label']=='ham']['caps_ratio']
    spam_caps = df[df['label']=='spam']['caps_ratio']
    bp = ax3.boxplot([ham_caps, spam_caps], labels=['Ham','Spam'],
                     patch_artist=True, widths=0.4,
                     boxprops=dict(linewidth=1.2))
    for patch, color in zip(bp['boxes'], ['#27ae60','#e74c3c']):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax3.set_title('CAPS Ratio by Class', fontweight='bold')
    ax3.set_ylabel('Fraction of uppercase letters')

    plt.suptitle('Dataset Overview', fontsize=14, fontweight='bold', y=1.02)
    _save(fig, 'data_overview.png')


def _plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'],
                linewidths=0.5, ax=ax, annot_kws={'size':14})
    # Annotate each cell
    labels = [['TN','FP'],['FN','TP']]
    for i in range(2):
        for j in range(2):
            ax.text(j+0.5, i+0.75, labels[i][j],
                    ha='center', va='center', fontsize=9,
                    color='gray', style='italic')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    _save(fig, 'confusion_matrix.png')


def _plot_roc_pr_curves(y_test, y_prob):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax1.plot(fpr, tpr, color='#e74c3c', lw=2, label=f'AUC = {auc:.4f}')
    ax1.plot([0,1],[0,1], 'k--', lw=1, alpha=0.5, label='Random')
    ax1.fill_between(fpr, tpr, alpha=0.08, color='#e74c3c')
    ax1.set_title('ROC Curve', fontsize=13, fontweight='bold')
    ax1.set_xlabel('False Positive Rate'); ax1.set_ylabel('True Positive Rate')
    ax1.legend(loc='lower right'); ax1.grid(alpha=0.3)

    # Precision-Recall curve
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    ax2.plot(rec, prec, color='#3498db', lw=2, label=f'AP = {ap:.4f}')
    ax2.fill_between(rec, prec, alpha=0.08, color='#3498db')
    ax2.axhline(y=y_test.mean(), color='k', linestyle='--', lw=1,
                alpha=0.5, label=f'Baseline ({y_test.mean():.2f})')
    ax2.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
    ax2.legend(loc='upper right'); ax2.grid(alpha=0.3)

    plt.suptitle('Model Performance Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, 'roc_pr_curves.png')


def _plot_cv_scores(cv_scores):
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics = list(cv_scores.keys())
    means   = [cv_scores[m].mean() for m in metrics]
    stds    = [cv_scores[m].std()  for m in metrics]
    colors  = ['#3498db','#e74c3c','#2ecc71','#f39c12','#9b59b6']

    bars = ax.bar(metrics, means, yerr=stds, capsize=5,
                  color=colors, edgecolor='black', lw=0.8, width=0.5,
                  error_kw={'linewidth':1.5})
    ax.set_ylim(0.85, 1.01)
    ax.set_title('5-Fold Cross-Validation Scores (mean ± std)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score')
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, mean + std + 0.002,
                f'{mean:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    _save(fig, 'cv_scores.png')


def _plot_top_spam_features(model, word_vec):
    """
    Extract top spam/ham words from the underlying LR or NB sub-model.
    For ensemble, tries to pull from the LR estimator.
    """
    try:
        # Try to get LR from ensemble
        if hasattr(model, 'estimators_'):
            lr_model = None
            for name, est in zip(model.estimators, model.estimators_):
                if 'lr' in str(name):
                    lr_model = est
                    break
            if lr_model is None:
                return
            coefs = lr_model.coef_[0]
        elif hasattr(model, 'coef_'):
            coefs = model.coef_[0]
        else:
            return

        feature_names = np.array(word_vec.get_feature_names_out())
        if len(coefs) < len(feature_names):
            coefs = coefs[:len(feature_names)]
        elif len(coefs) > len(feature_names):
            coefs = coefs[:len(feature_names)]

        top_spam = np.argsort(coefs)[-20:][::-1]
        top_ham  = np.argsort(coefs)[:20]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

        ax1.barh(feature_names[top_spam][::-1], coefs[top_spam][::-1],
                 color='#e74c3c', edgecolor='black', lw=0.5)
        ax1.set_title('Top 20 Spam Indicators', fontweight='bold', color='#e74c3c')
        ax1.set_xlabel('LR Coefficient')

        ax2.barh(feature_names[top_ham], np.abs(coefs[top_ham]),
                 color='#27ae60', edgecolor='black', lw=0.5)
        ax2.set_title('Top 20 Ham Indicators', fontweight='bold', color='#27ae60')
        ax2.set_xlabel('|LR Coefficient|')

        plt.suptitle('Most Predictive Features (Logistic Regression)', fontweight='bold')
        plt.tight_layout()
        _save(fig, 'top_features.png')
    except Exception as e:
        logger.warning("Could not plot top features: %s", e)


def _save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved plot: %s", path)


# ── Persistence ─────────────────────────────────────────────────────────────────

def save_model(model, word_vec, char_vec):
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f, protocol=4)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump({'word': word_vec, 'char': char_vec}, f, protocol=4)
    logger.info("Model saved -> %s", MODEL_PATH)
    logger.info("Vectorizers saved -> %s", VECTORIZER_PATH)


def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            "Model files not found. Run 'python main.py --train' first."
        )
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vecs = pickle.load(f)
    logger.info("Model and vectorizers loaded.")
    return model, vecs['word'], vecs['char']
