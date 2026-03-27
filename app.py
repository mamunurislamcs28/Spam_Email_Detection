"""
app.py - Advanced Flask REST API
==================================
Endpoints:
    GET  /                  → Web UI
    POST /predict           → Single message prediction (JSON)
    POST /predict/batch     → Batch prediction (JSON list or CSV upload)
    POST /feedback/<id>     → Submit prediction feedback
    GET  /history           → Recent predictions from DB
    GET  /stats             → Aggregate statistics
    GET  /health            → Health check

Authentication:
    Optional API key via X-API-Key header (set API_KEY env var to enable).
    If API_KEY not set, all requests are accepted (dev mode).
"""

import os
import io
import csv
import logging
from functools import wraps
from flask import Flask, request, jsonify, render_template, g

from model import load_model
from utils import (
    setup_logging, predict_message, batch_predict,
    save_feedback, get_prediction_history, get_stats, init_db
)

# ── App setup ───────────────────────────────────────────────────────────────────
setup_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB upload limit

# ── Model (loaded once at startup) ─────────────────────────────────────────────
model = word_vec = char_vec = None

def initialize_model():
    global model, word_vec, char_vec
    try:
        model, word_vec, char_vec = load_model()
        logger.info("Model loaded and ready.")
    except FileNotFoundError as e:
        logger.warning("%s", e)

# ── Optional API Key Auth ───────────────────────────────────────────────────────
API_KEY = os.environ.get('API_KEY')  # set in env to enable auth

def require_api_key(f):
    """
    Decorator that checks for a valid API key in X-API-Key header.
    Only enforced when API_KEY environment variable is set.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        if API_KEY:
            key = request.headers.get('X-API-Key')
            if key != API_KEY:
                return jsonify({'error': 'Unauthorized. Provide valid X-API-Key header.'}), 401
        return f(*args, **kwargs)
    return decorated


# ── Routes ──────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    """
    Single message prediction.

    Request JSON:
        { "message": "...", "threshold": 0.5 }   ← threshold is optional

    Response JSON:
        {
          "label": "SPAM",
          "confidence": 97.42,
          "spam_prob": 0.9742,
          "ham_prob": 0.0258,
          "clean_text": "...",
          "features": { "num_exclamations": 3, ... },
          "db_id": 42
        }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded. Run python main.py --train first.'}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Request body must be JSON with a "message" field.'}), 400

    message   = data.get('message', '').strip()
    threshold = float(data.get('threshold', 0.5))

    if not message:
        return jsonify({'error': 'Message field is empty.'}), 400
    if not (0.0 < threshold < 1.0):
        return jsonify({'error': 'threshold must be between 0 and 1 (exclusive).'}), 400
    if len(message) > 10_000:
        return jsonify({'error': 'Message too long (max 10,000 chars).'}), 400

    try:
        result = predict_message(
            message, model, word_vec, char_vec,
            threshold=threshold, source='web'
        )
        return jsonify(result)
    except Exception as e:
        logger.exception("Prediction error")
        return jsonify({'error': str(e)}), 500


@app.route('/predict/batch', methods=['POST'])
@require_api_key
def predict_batch():
    """
    Batch prediction — accepts JSON list OR CSV file upload.

    JSON mode:
        { "messages": ["msg1", "msg2", ...], "threshold": 0.5 }

    CSV mode:
        POST multipart/form-data with file field 'file'
        CSV must have a 'message' column (or first column is used).

    Response: array of prediction objects.
    """
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 503

    threshold = float(request.form.get('threshold', 0.5) or
                      (request.get_json(silent=True) or {}).get('threshold', 0.5))

    # ── CSV upload ──────────────────────────────────────────────────────
    if 'file' in request.files:
        f = request.files['file']
        if not f.filename.endswith('.csv'):
            return jsonify({'error': 'Only .csv files are supported.'}), 400
        content = f.read().decode('utf-8', errors='replace')
        reader  = csv.DictReader(io.StringIO(content))
        rows    = list(reader)
        if not rows:
            return jsonify({'error': 'CSV file is empty.'}), 400
        col = 'message' if 'message' in rows[0] else list(rows[0].keys())[0]
        messages = [r[col] for r in rows if r.get(col)]

    # ── JSON list ───────────────────────────────────────────────────────
    else:
        data = request.get_json(silent=True)
        if not data or 'messages' not in data:
            return jsonify({'error': 'Provide JSON {"messages": [...]} or a CSV file upload.'}), 400
        messages = data['messages']

    if len(messages) > 1000:
        return jsonify({'error': 'Max 1,000 messages per batch request.'}), 400

    try:
        df = batch_predict(messages, model, word_vec, char_vec, threshold=threshold)
        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        logger.exception("Batch prediction error")
        return jsonify({'error': str(e)}), 500


@app.route('/feedback/<int:prediction_id>', methods=['POST'])
def feedback(prediction_id: int):
    """
    Submit feedback for a prediction.

    Request JSON:
        { "correct": true }                    ← prediction was right
        { "correct": false, "true_label": "ham" }  ← correction
    """
    data = request.get_json(silent=True)
    if not data or 'correct' not in data:
        return jsonify({'error': 'Provide {"correct": true|false, "true_label": "ham"|"spam"}'}), 400

    correct    = bool(data['correct'])
    true_label = data.get('true_label')

    if true_label and true_label not in ('ham', 'spam'):
        return jsonify({'error': 'true_label must be "ham" or "spam"'}), 400

    try:
        save_feedback(prediction_id, correct=correct, true_label=true_label)
        return jsonify({'status': 'ok', 'prediction_id': prediction_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/history')
def history():
    """Return recent prediction history from the database."""
    limit = min(int(request.args.get('limit', 20)), 100)
    try:
        rows = get_prediction_history(limit=limit)
        return jsonify(rows)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stats')
def stats():
    """Return aggregate prediction statistics."""
    try:
        return jsonify(get_stats())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({
        'status'      : 'ok',
        'model_loaded': model is not None,
        'version'     : '2.0',
    })


# ── Error Handlers ──────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found.'}), 404

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Max 5 MB.'}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error.'}), 500


# ── Startup ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    init_db()
    initialize_model()
    port = int(os.environ.get('PORT', 5000))
    logger.info("Flask app starting on http://localhost:%d", port)
    app.run(debug=True, port=port)
