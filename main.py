"""
main.py - Advanced CLI Entry Point
====================================
Usage:
    python main.py --train                    # Train ensemble model
    python main.py --train --model svm        # Train SVM only
    python main.py --predict                  # Interactive CLI prediction
    python main.py --predict --threshold 0.3  # Lower spam threshold
    python main.py --batch emails.csv         # Batch predict from CSV
    python main.py --stats                    # Show DB prediction stats
    python main.py --train --predict          # Train then immediately predict
"""

import argparse
import sys
import logging

from utils import (
    setup_logging, ensure_dataset, print_banner,
    print_prediction_result, batch_predict, get_stats
)
from preprocess import load_and_preprocess_data
from model import (
    split_data, train_model, evaluate_model,
    plot_all, save_model, load_model
)


def run_training(model_type: str = 'ensemble'):
    logger = logging.getLogger(__name__)

    logger.info("=== STEP 1: Datasets ===")
    # ensure_dataset() now returns a dict of paths
    dataset_paths = ensure_dataset()

    logger.info("=== STEP 2: Preprocessing ===")
    df = load_and_preprocess_data(dataset_paths)

    logger.info("=== STEP 3: Train/Test Split ===")
    train_df, test_df, _, _ = split_data(df)
    logger.info("Train: %d  |  Test: %d", len(train_df), len(test_df))

    logger.info("=== STEP 4-7: Feature Extraction + Training + CV ===")
    word_vec, char_vec, model, cv_scores = train_model(train_df, model_type=model_type)

    logger.info("=== STEP 8: Evaluation ===")
    metrics, y_pred, y_prob, y_test = evaluate_model(model, word_vec, char_vec, test_df)

    logger.info("=== STEP 9: Visualizations ===")
    plot_all(df, y_test, y_pred, y_prob, cv_scores, model, word_vec)

    logger.info("=== STEP 10: Saving Model ===")
    save_model(model, word_vec, char_vec)

    print("\n Training complete!")
    print(f"   Model type   : {model_type}")
    print(f"   Total records: {len(df)}")
    print(f"   Test Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"   Test F1-Score: {metrics['f1_score']*100:.2f}%")
    print(f"   ROC-AUC      : {metrics['roc_auc']:.4f}")
    print(f"   Model saved  : spam_model.pkl\n")

    return model, word_vec, char_vec


def run_prediction(model=None, word_vec=None, char_vec=None, threshold: float = 0.5):
    """Interactive CLI prediction loop with feedback collection."""
    from utils import predict_message, save_feedback

    logger = logging.getLogger(__name__)

    if model is None:
        try:
            model, word_vec, char_vec = load_model()
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)

    print(f"\n📬 Interactive Spam Predictor  (threshold={threshold})")
    print("   Type a message → get spam/ham verdict.")
    print("   Type 'stats' to see prediction history stats.")
    print("   Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("Message: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Goodbye!")
            break

        if user_input.lower() in ('quit', 'exit', 'q'):
            print("[INFO] Goodbye!")
            break

        if user_input.lower() == 'stats':
            stats = get_stats()
            print("\n📊 Prediction Statistics:")
            for k, v in stats.items():
                print(f"   {k}: {v}")
            print()
            continue

        if not user_input:
            continue

        try:
            result = predict_message(
                user_input, model, word_vec, char_vec,
                threshold=threshold, source='cli'
            )
            print_prediction_result(result, user_input)

            # Collect feedback
            if result['db_id']:
                fb = input("  Was this correct? [y/n/skip]: ").strip().lower()
                if fb == 'y':
                    save_feedback(result['db_id'], correct=True)
                elif fb == 'n':
                    true = input("  True label [ham/spam]: ").strip().lower()
                    save_feedback(result['db_id'], correct=False,
                                  true_label=true if true in ('ham','spam') else None)
                print()

        except Exception as e:
            logger.error("Prediction failed: %s", e)


def run_batch(csv_path: str, threshold: float = 0.5):
    """Batch predict all messages in a CSV file and save results."""
    import pandas as pd
    logger = logging.getLogger(__name__)

    if not csv_path or not __import__('os').path.exists(csv_path):
        logger.error("CSV file not found: %s", csv_path)
        sys.exit(1)

    try:
        model, word_vec, char_vec = load_model()
    except FileNotFoundError as e:
        logger.error(str(e)); sys.exit(1)

    df = pd.read_csv(csv_path)
    # Expect a column named 'message' (or take the first column)
    msg_col = 'message' if 'message' in df.columns else df.columns[0]
    messages = df[msg_col].astype(str).tolist()

    logger.info("Running batch prediction on %d messages...", len(messages))
    results = batch_predict(messages, model, word_vec, char_vec, threshold=threshold)

    out_path = csv_path.replace('.csv', '_predictions.csv')
    results.to_csv(out_path, index=False)
    print(f"\n✅ Batch complete! Results saved to: {out_path}")
    print(f"   Spam detected: {(results['label']=='SPAM').sum()} / {len(results)}")


def show_stats():
    stats = get_stats()
    print("\n📊 Spam Detector — Prediction History Statistics")
    print("─" * 45)
    for k, v in stats.items():
        print(f"  {k:<25}: {v}")
    print()


def main():
    print_banner()
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Email Spam Detector v2 — Advanced ML Pipeline",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--predict', action='store_true',
                        help='Interactive prediction mode')
    parser.add_argument('--batch', type=str, metavar='CSV',
                        help='Batch predict from a CSV file')
    parser.add_argument('--stats', action='store_true',
                        help='Show prediction history statistics')
    parser.add_argument('--model', type=str, default='ensemble',
                        choices=['ensemble','svm','logistic_regression','naive_bayes'],
                        help='Model type for training (default: ensemble)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Spam probability threshold 0-1 (default: 0.5)')

    args = parser.parse_args()

    if not any([args.train, args.predict, args.batch, args.stats]):
        parser.print_help()
        print("\n[TIP] Quick start: python main.py --train --predict")
        sys.exit(0)

    model = word_vec = char_vec = None

    if args.train:
        model, word_vec, char_vec = run_training(model_type=args.model)

    if args.predict:
        run_prediction(model, word_vec, char_vec, threshold=args.threshold)

    if args.batch:
        run_batch(args.batch, threshold=args.threshold)

    if args.stats:
        show_stats()


if __name__ == '__main__':
    main()
