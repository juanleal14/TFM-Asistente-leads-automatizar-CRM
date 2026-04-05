"""
tests/test_feature_engineering.py
──────────────────────────────────
Unit tests for src/feature_engineering.py

Run with:
    pytest tests/test_feature_engineering.py -v
"""
import numpy as np
import pandas as pd
import pytest

# TODO: implement full test suite once dataset generation is complete.
# The stubs below document expected behaviour and can be fleshed out with
# real fixtures once `data/raw/moveup_crm_dataset.csv` exists.


class TestLoadAndClean:
    """Tests for load_and_clean()."""

    def test_returns_dataframe(self, tmp_path):
        # TODO: write a minimal CSV fixture to tmp_path and assert
        # that load_and_clean(csv_path) returns a non-empty DataFrame.
        pytest.skip("Fixture CSV not yet available")

    def test_null_fill_applied(self, tmp_path):
        # TODO: assert that rows with empty prev_outcome / prev_next_step
        # are filled with CONFIG["null_fill_value"] = "PRIMERA_LLAMADA".
        pytest.skip("Fixture CSV not yet available")

    def test_numeric_columns_are_numeric(self, tmp_path):
        # TODO: verify that all CONFIG["numeric_features"] columns are float/int
        # after calling load_and_clean().
        pytest.skip("Fixture CSV not yet available")


class TestGenerateEmbeddings:
    """Tests for generate_embeddings()."""

    def test_output_shape(self):
        # TODO: pass a small DataFrame (3 rows) and assert that the returned
        # numpy array has shape (3, 768) — 2 × 384 dimensions.
        pytest.skip("Model download required in CI")

    def test_cache_hit(self, tmp_path):
        # TODO: run generate_embeddings twice with the same cache_path,
        # assert that the second call returns identical results and does not
        # re-instantiate the SentenceTransformer (mock the constructor).
        pytest.skip("Model download required in CI")


class TestBuildFeatureMatrix:
    """Tests for build_feature_matrix()."""

    def test_feature_count_matches_names(self):
        # TODO: build a minimal DataFrame + dummy embeddings and assert that
        # X.shape[1] == len(feature_names).
        pytest.skip("End-to-end fixture not yet available")

    def test_fit_false_uses_provided_encoders(self):
        # TODO: fit on a small training set, then call with fit=False on a
        # test row and assert no refitting occurs (check scaler.mean_ unchanged).
        pytest.skip("End-to-end fixture not yet available")

    def test_label_encoding_roundtrip(self):
        # TODO: verify that label_encoder.inverse_transform(y) returns the
        # original next_step strings from the DataFrame.
        pytest.skip("End-to-end fixture not yet available")
