"""
tests/test_pipeline.py - Automated Test Suite
=============================================
Run with:  pytest tests/ -v

Tests cover:
  - Text preprocessing correctness
  - Custom feature extraction
  - Model prediction sanity checks
  - API endpoint responses
  - Edge cases (empty input, unicode, very long text)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np

from preprocess import clean_text, extract_custom_features


# ── Preprocessing Tests ─────────────────────────────────────────────────────────

class TestCleanText:

    def test_lowercasing(self):
        result = clean_text("FREE PRIZE WIN NOW")
        assert result == result.lower()

    def test_url_removal(self):
        result = clean_text("Click here http://spam.com to win")
        assert 'http' not in result
        assert 'spam.com' not in result

    def test_email_removal(self):
        result = clean_text("Contact us at prize@win.com now")
        assert '@' not in result

    def test_number_removal(self):
        result = clean_text("Call 07912345678 to claim your 1000 prize")
        assert not any(c.isdigit() for c in result)

    def test_punctuation_removal(self):
        result = clean_text("WIN!!!! FREE $$$")
        assert '!' not in result
        assert '$' not in result

    def test_stopword_removal(self):
        result = clean_text("the cat is on the mat")
        tokens = result.split()
        stopwords_present = [t for t in tokens if t in ('the', 'is', 'on')]
        assert len(stopwords_present) == 0

    def test_empty_string(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""

    def test_unicode_input(self):
        # Should not crash on unicode
        result = clean_text("Hello wörld! Ça va?")
        assert isinstance(result, str)

    def test_all_numbers(self):
        result = clean_text("12345 67890")
        assert result == ""

    def test_known_spam_words_survive(self):
        result = clean_text("You have won a free prize claim now")
        # After stemming/lemmatizing, spam signal words should remain
        assert len(result) > 0


class TestCustomFeatures:

    def test_exclamation_count(self):
        feats = extract_custom_features("WIN NOW!!!")
        assert feats['num_exclamations'] == 3.0

    def test_url_detection(self):
        feats = extract_custom_features("Click http://spam.com and https://win.com")
        assert feats['num_urls'] == 2.0

    def test_currency_detection(self):
        feats = extract_custom_features("You won $1000 and £500!")
        assert feats['num_currency'] == 2.0

    def test_caps_ratio(self):
        feats = extract_custom_features("FREE PRIZE")
        assert feats['caps_ratio'] > 0.8

    def test_normal_text_low_caps(self):
        feats = extract_custom_features("hello how are you doing today")
        assert feats['caps_ratio'] < 0.1

    def test_empty_features(self):
        feats = extract_custom_features("")
        assert all(v == 0.0 for v in feats.values())

    def test_none_features(self):
        feats = extract_custom_features(None)
        assert all(v == 0.0 for v in feats.values())

    def test_msg_length(self):
        msg   = "Hello world"
        feats = extract_custom_features(msg)
        assert feats['msg_length'] == float(len(msg))

    def test_html_detection(self):
        feats_html = extract_custom_features("<b>FREE</b> <a href='x'>click</a>")
        feats_plain = extract_custom_features("FREE click here")
        assert feats_html['has_html'] == 1.0
        assert feats_plain['has_html'] == 0.0

    def test_repeated_chars(self):
        feats = extract_custom_features("freeeeee moneyyyy!!!!")
        assert feats['repeated_chars'] > 0


# ── Integration Tests (require trained model) ───────────────────────────────────

class TestPrediction:
    """
    These tests require a trained model (spam_model.pkl).
    Skip gracefully if model not yet trained.
    """

    @pytest.fixture(autouse=True)
    def load_model_fixture(self):
        try:
            from model import load_model
            self.model, self.word_vec, self.char_vec = load_model()
        except FileNotFoundError:
            pytest.skip("Model not trained yet. Run: python main.py --train")

    def test_obvious_spam(self):
        from utils import predict_message
        result = predict_message(
            "CONGRATULATIONS! You've WON £1000! Click http://win.com NOW to CLAIM FREE PRIZE!!!",
            self.model, self.word_vec, self.char_vec, save_to_db=False
        )
        assert result['label'] == 'SPAM'
        assert result['spam_prob'] > 0.7

    def test_obvious_ham(self):
        from utils import predict_message
        result = predict_message(
            "Hey, are you coming to the team meeting at 3pm on Thursday?",
            self.model, self.word_vec, self.char_vec, save_to_db=False
        )
        assert result['label'] == 'HAM'
        assert result['ham_prob'] > 0.7

    def test_result_keys(self):
        from utils import predict_message
        result = predict_message(
            "Test message", self.model, self.word_vec,
            self.char_vec, save_to_db=False
        )
        for key in ('label', 'confidence', 'spam_prob', 'ham_prob', 'clean_text', 'features'):
            assert key in result

    def test_probabilities_sum_to_one(self):
        from utils import predict_message
        result = predict_message(
            "Hello there", self.model, self.word_vec,
            self.char_vec, save_to_db=False
        )
        total = result['spam_prob'] + result['ham_prob']
        assert abs(total - 1.0) < 0.01

    def test_confidence_range(self):
        from utils import predict_message
        result = predict_message(
            "Test", self.model, self.word_vec,
            self.char_vec, save_to_db=False
        )
        assert 0.0 <= result['confidence'] <= 100.0

    def test_empty_raises(self):
        from utils import predict_message
        with pytest.raises(ValueError):
            predict_message("", self.model, self.word_vec, self.char_vec, save_to_db=False)

    def test_threshold_effect(self):
        from utils import predict_message
        msg = "You might have won something"
        r_low  = predict_message(msg, self.model, self.word_vec, self.char_vec,
                                  threshold=0.1, save_to_db=False)
        r_high = predict_message(msg, self.model, self.word_vec, self.char_vec,
                                  threshold=0.9, save_to_db=False)
        # Same message, low threshold → more likely SPAM, high → more likely HAM
        assert r_low['spam_prob'] == r_high['spam_prob']  # probs unchanged
        # labels may differ based on threshold


# ── Flask API Tests ─────────────────────────────────────────────────────────────

class TestFlaskAPI:

    @pytest.fixture(autouse=True)
    def client(self):
        try:
            from model import load_model
            import app as flask_app
            flask_app.model, flask_app.word_vec, flask_app.char_vec = load_model()
        except FileNotFoundError:
            pytest.skip("Model not trained yet.")
        from app import app
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_health_endpoint(self):
        resp = self.client.get('/health')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'ok'
        assert data['model_loaded'] is True

    def test_predict_spam(self):
        resp = self.client.post('/predict',
            json={'message': 'FREE PRIZE WIN CLICK NOW http://spam.com !!!'})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['label'] in ('SPAM', 'HAM')
        assert 'confidence' in data

    def test_predict_empty_message(self):
        resp = self.client.post('/predict', json={'message': ''})
        assert resp.status_code == 400

    def test_predict_no_body(self):
        resp = self.client.post('/predict', data='not json',
                                content_type='text/plain')
        assert resp.status_code == 400

    def test_stats_endpoint(self):
        resp = self.client.get('/stats')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'total_predictions' in data

    def test_history_endpoint(self):
        resp = self.client.get('/history?limit=5')
        assert resp.status_code == 200
        assert isinstance(resp.get_json(), list)
