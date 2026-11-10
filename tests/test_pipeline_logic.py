"""
Unit tests for pipeline_logic.py
=================================
Tests cover every cleaning/transformation function individually (happy path +
edge cases + expected errors) as well as run_pipeline end-to-end integration.

Run with:
    pytest tests/test_pipeline_logic.py -v
"""

import sys
import os

import pytest
import pandas as pd
import numpy as np

# Make the project root importable when running from any working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline_logic import (
    sanitize_for_csv,
    impute_missing,
    normalize,
    scale,
    binarize,
    remove_outliers,
    remove_nan_cols,
    remove_nan_rows,
    dataset_multiplier,
    add_noise,
    one_hot_encode,
    oversample_data,
    undersample_data,
    run_pipeline,
    ImputeMissingValuesParams,
    NormalizeParams,
    ScaleParams,
    BinarizeParams,
    RemoveOutliersParams,
    RemoveNaNColsParams,
    RemoveNaNRowsParams,
    DatasetMultiplierParams,
    AddNoiseParams,
    OversampleParams,
    UndersampleParams,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def numeric_df():
    """Numeric DataFrame containing NaNs for imputation tests."""
    return pd.DataFrame({
        "a": [1.0, 2.0, np.nan, 4.0, 5.0],
        "b": [10.0, 20.0, 30.0, np.nan, 50.0],
    })


@pytest.fixture
def clean_numeric_df():
    """Numeric DataFrame with no missing values."""
    return pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [10.0, 20.0, 30.0, 40.0, 50.0],
    })


@pytest.fixture
def categorical_df():
    """Mixed DataFrame with a categorical string column and a numeric column."""
    return pd.DataFrame({
        "cat": ["apple", "banana", "apple", "cherry", "banana"],
        "val": [1.0, 2.0, 3.0, 4.0, 5.0],
    })


@pytest.fixture
def imbalanced_df():
    """DataFrame with a heavily imbalanced class label, suitable for
    over-/under-sampling tests."""
    return pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feature2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        "label":    ["A", "A", "A", "A", "A", "A", "A", "B", "B", "B"],
    })


# ---------------------------------------------------------------------------
# impute_missing
# ---------------------------------------------------------------------------

class TestImputeMissing:
    """Tests for impute_missing()."""

    def test_mean_strategy_removes_all_nans(self, numeric_df):
        params = ImputeMissingValuesParams(strategy="mean")
        result = impute_missing(numeric_df.copy(), params)
        assert result.isnull().sum().sum() == 0

    def test_median_strategy_removes_all_nans(self, numeric_df):
        params = ImputeMissingValuesParams(strategy="median")
        result = impute_missing(numeric_df.copy(), params)
        assert result.isnull().sum().sum() == 0

    def test_constant_strategy_fills_with_specified_value(self, numeric_df):
        params = ImputeMissingValuesParams(strategy="constant", fill_value=99.0)
        result = impute_missing(numeric_df.copy(), params)
        assert result.isnull().sum().sum() == 0
        # The NaN in column 'a' (row 2) should be filled with 99.0
        assert result.loc[2, "a"] == 99.0

    def test_most_frequent_strategy_fills_numeric_nans(self):
        # most_frequent uses SimpleImputer over all columns; numeric NaNs
        # are reliably filled.  Object columns may retain NaN due to dtype
        # coercion behaviour in some pandas/sklearn versions.
        df = pd.DataFrame({
            "a": [1.0, 1.0, 2.0, None],
            "b": [10.0, 10.0, 10.0, None],
        })
        params = ImputeMissingValuesParams(strategy="most_frequent")
        result = impute_missing(df.copy(), params)
        assert result.isnull().sum().sum() == 0

    def test_invalid_strategy_raises_value_error(self, numeric_df):
        params = ImputeMissingValuesParams(strategy="bogus_strategy")
        with pytest.raises(ValueError, match="Unsupported imputation strategy"):
            impute_missing(numeric_df.copy(), params)

    def test_no_numeric_columns_raises_for_mean(self):
        df = pd.DataFrame({"text": ["hello", "world", None]})
        params = ImputeMissingValuesParams(strategy="mean")
        with pytest.raises(ValueError, match="No numeric columns"):
            impute_missing(df.copy(), params)

    def test_output_shape_unchanged(self, numeric_df):
        params = ImputeMissingValuesParams(strategy="mean")
        result = impute_missing(numeric_df.copy(), params)
        assert result.shape == numeric_df.shape


# ---------------------------------------------------------------------------
# normalize (StandardScaler)
# ---------------------------------------------------------------------------

class TestNormalize:
    """Tests for normalize() — applies StandardScaler to all numeric columns."""

    def test_output_shape_unchanged(self, clean_numeric_df):
        result = normalize(clean_numeric_df.copy(), NormalizeParams())
        assert result.shape == clean_numeric_df.shape

    def test_mean_is_approximately_zero(self, clean_numeric_df):
        result = normalize(clean_numeric_df.copy(), NormalizeParams())
        assert abs(result["a"].mean()) < 1e-9
        assert abs(result["b"].mean()) < 1e-9

    def test_population_std_is_approximately_one(self, clean_numeric_df):
        result = normalize(clean_numeric_df.copy(), NormalizeParams())
        # sklearn StandardScaler uses population std (ddof=0)
        assert abs(result["a"].std(ddof=0) - 1.0) < 1e-6
        assert abs(result["b"].std(ddof=0) - 1.0) < 1e-6

    def test_non_numeric_columns_are_untouched(self):
        df = pd.DataFrame({"num": [1.0, 2.0, 3.0], "label": ["x", "y", "z"]})
        result = normalize(df.copy(), NormalizeParams())
        assert list(result["label"]) == ["x", "y", "z"]


# ---------------------------------------------------------------------------
# scale (MinMaxScaler)
# ---------------------------------------------------------------------------

class TestScale:
    """Tests for scale() — applies MinMaxScaler to all numeric columns."""

    def test_default_range_0_to_1(self, clean_numeric_df):
        result = scale(clean_numeric_df.copy(), ScaleParams(min_value=0, max_value=1))
        assert result["a"].min() >= 0.0
        assert result["a"].max() <= 1.0
        assert result["b"].min() >= 0.0
        assert result["b"].max() <= 1.0

    def test_custom_range_minus1_to_1(self, clean_numeric_df):
        result = scale(clean_numeric_df.copy(), ScaleParams(min_value=-1, max_value=1))
        assert result["a"].min() >= -1.0
        assert result["a"].max() <= 1.0

    def test_output_shape_unchanged(self, clean_numeric_df):
        result = scale(clean_numeric_df.copy(), ScaleParams())
        assert result.shape == clean_numeric_df.shape


# ---------------------------------------------------------------------------
# binarize
# ---------------------------------------------------------------------------

class TestBinarize:
    """Tests for binarize() — applies sklearn Binarizer to numeric columns."""

    def test_output_values_are_binary(self, clean_numeric_df):
        result = binarize(clean_numeric_df.copy(), BinarizeParams(threshold=2.5))
        unique_a = set(result["a"].unique())
        assert unique_a.issubset({0, 1, 0.0, 1.0})

    def test_threshold_zero_converts_positives_to_one(self):
        df = pd.DataFrame({"x": [0.0, 0.5, 1.0, -1.0]})
        result = binarize(df.copy(), BinarizeParams(threshold=0.0))
        # Values > 0 become 1; 0 and below become 0
        assert result.loc[1, "x"] == 1.0
        assert result.loc[3, "x"] == 0.0

    def test_raises_on_nan_in_numeric_columns(self, numeric_df):
        with pytest.raises(ValueError, match="NaN"):
            binarize(numeric_df.copy(), BinarizeParams(threshold=0.0))


# ---------------------------------------------------------------------------
# remove_outliers
# ---------------------------------------------------------------------------

class TestRemoveOutliers:
    """Tests for remove_outliers() — IQR-based outlier removal."""

    def test_extreme_value_is_removed(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 1000.0]})
        result = remove_outliers(df.copy(), RemoveOutliersParams(iqr_multiplier=1.5))
        assert 1000.0 not in result["x"].values

    def test_clean_data_is_not_filtered(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = remove_outliers(df.copy(), RemoveOutliersParams(iqr_multiplier=1.5))
        assert len(result) == len(df)

    def test_larger_multiplier_retains_more_rows(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})
        result_tight = remove_outliers(df.copy(), RemoveOutliersParams(iqr_multiplier=0.5))
        result_loose = remove_outliers(df.copy(), RemoveOutliersParams(iqr_multiplier=10.0))
        assert len(result_loose) >= len(result_tight)


# ---------------------------------------------------------------------------
# remove_nan_cols
# ---------------------------------------------------------------------------

class TestRemoveNanCols:
    """Tests for remove_nan_cols() — drops columns exceeding NaN threshold."""

    def test_drops_column_above_threshold(self):
        df = pd.DataFrame({
            "good": [1, 2, 3, 4, 5],
            "bad":  [np.nan, np.nan, np.nan, np.nan, 1],  # 80% NaN
        })
        result = remove_nan_cols(df.copy(), RemoveNaNColsParams(nan_threshold=0.5))
        assert "bad" not in result.columns
        assert "good" in result.columns

    def test_keeps_column_below_threshold(self):
        df = pd.DataFrame({
            "ok": [1, np.nan, 3, 4, 5],  # 20% NaN — below 0.5 threshold
        })
        result = remove_nan_cols(df.copy(), RemoveNaNColsParams(nan_threshold=0.5))
        assert "ok" in result.columns

    def test_all_good_columns_unchanged(self, clean_numeric_df):
        result = remove_nan_cols(clean_numeric_df.copy(), RemoveNaNColsParams(nan_threshold=0.5))
        assert list(result.columns) == list(clean_numeric_df.columns)


# ---------------------------------------------------------------------------
# remove_nan_rows
# ---------------------------------------------------------------------------

class TestRemoveNanRows:
    """Tests for remove_nan_rows() — drops rows exceeding NaN threshold."""

    def test_drops_row_where_all_values_are_nan(self):
        df = pd.DataFrame({
            "a": [1, np.nan, 3],
            "b": [4, np.nan, 6],
            "c": [7, np.nan, 9],
        })
        result = remove_nan_rows(df.copy(), RemoveNaNRowsParams(nan_threshold=0.5))
        assert len(result) == 2

    def test_keeps_rows_with_few_nans(self):
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        })
        result = remove_nan_rows(df.copy(), RemoveNaNRowsParams(nan_threshold=0.5))
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# dataset_multiplier
# ---------------------------------------------------------------------------

class TestDatasetMultiplier:
    """Tests for dataset_multiplier() — repeats the DataFrame N times."""

    def test_doubles_row_count(self, clean_numeric_df):
        result = dataset_multiplier(clean_numeric_df.copy(), DatasetMultiplierParams(size_multiplier=2))
        assert len(result) == len(clean_numeric_df) * 2

    def test_triples_row_count(self, clean_numeric_df):
        result = dataset_multiplier(clean_numeric_df.copy(), DatasetMultiplierParams(size_multiplier=3))
        assert len(result) == len(clean_numeric_df) * 3

    def test_multiplier_of_one_is_identity(self, clean_numeric_df):
        result = dataset_multiplier(clean_numeric_df.copy(), DatasetMultiplierParams(size_multiplier=1))
        assert len(result) == len(clean_numeric_df)

    def test_invalid_string_multiplier_raises(self, clean_numeric_df):
        params = DatasetMultiplierParams(size_multiplier="xyz")
        with pytest.raises(ValueError, match="Invalid multiplier"):
            dataset_multiplier(clean_numeric_df.copy(), params)

    def test_index_is_reset(self, clean_numeric_df):
        result = dataset_multiplier(clean_numeric_df.copy(), DatasetMultiplierParams(size_multiplier=2))
        assert list(result.index) == list(range(len(result)))


# ---------------------------------------------------------------------------
# add_noise
# ---------------------------------------------------------------------------

class TestAddNoise:
    """Tests for add_noise() — adds Gaussian noise to numeric columns."""

    def test_output_shape_unchanged(self, clean_numeric_df):
        result = add_noise(clean_numeric_df.copy(), AddNoiseParams(noise_factor=0.01))
        assert result.shape == clean_numeric_df.shape

    def test_large_noise_changes_values(self, clean_numeric_df):
        result = add_noise(clean_numeric_df.copy(), AddNoiseParams(noise_factor=100.0))
        assert not result["a"].equals(clean_numeric_df["a"])

    def test_zero_noise_preserves_values(self, clean_numeric_df):
        # noise_factor=0 → noise array is all zeros
        result = add_noise(clean_numeric_df.copy(), AddNoiseParams(noise_factor=0.0))
        pd.testing.assert_frame_equal(result, clean_numeric_df)

    def test_non_numeric_columns_untouched(self):
        df = pd.DataFrame({"num": [1.0, 2.0, 3.0], "label": ["x", "y", "z"]})
        result = add_noise(df.copy(), AddNoiseParams(noise_factor=1.0))
        assert list(result["label"]) == ["x", "y", "z"]


# ---------------------------------------------------------------------------
# one_hot_encode
# ---------------------------------------------------------------------------

class TestOneHotEncode:
    """Tests for one_hot_encode() — expands categorical columns."""

    def test_categorical_column_is_expanded(self, categorical_df):
        result = one_hot_encode(categorical_df.copy(), {})
        assert "cat" not in result.columns
        assert any("apple" in c for c in result.columns)

    def test_numeric_column_is_retained(self, categorical_df):
        result = one_hot_encode(categorical_df.copy(), {})
        assert "val" in result.columns

    def test_no_object_columns_leaves_df_unchanged(self, clean_numeric_df):
        result = one_hot_encode(clean_numeric_df.copy(), {})
        assert list(result.columns) == list(clean_numeric_df.columns)

    def test_all_original_categories_get_own_column(self, categorical_df):
        result = one_hot_encode(categorical_df.copy(), {})
        for cat in ["apple", "banana", "cherry"]:
            assert any(cat in c for c in result.columns)


# ---------------------------------------------------------------------------
# oversample_data
# ---------------------------------------------------------------------------

class TestOversampleData:
    """Tests for oversample_data() — RandomOverSampler on the minority class."""

    def test_total_row_count_increases(self, imbalanced_df):
        params = OversampleParams(target_column="label")
        result = oversample_data(imbalanced_df.copy(), params)
        assert len(result) >= len(imbalanced_df)

    def test_invalid_target_column_raises(self, imbalanced_df):
        params = OversampleParams(target_column="nonexistent_col")
        with pytest.raises(ValueError, match="Oversampling: Target column"):
            oversample_data(imbalanced_df.copy(), params)

    def test_nan_in_target_raises(self):
        df = pd.DataFrame({
            "x": [1, 2, 3],
            "label": ["A", None, "B"],
        })
        with pytest.raises(ValueError, match="missing values"):
            oversample_data(df.copy(), OversampleParams(target_column="label"))

    def test_columns_are_preserved(self, imbalanced_df):
        params = OversampleParams(target_column="label")
        result = oversample_data(imbalanced_df.copy(), params)
        assert set(result.columns) == set(imbalanced_df.columns)


# ---------------------------------------------------------------------------
# undersample_data
# ---------------------------------------------------------------------------

class TestUndersampleData:
    """Tests for undersample_data() — RandomUnderSampler on the majority class."""

    def test_total_row_count_decreases_or_stays_equal(self, imbalanced_df):
        params = UndersampleParams(target_column="label")
        result = undersample_data(imbalanced_df.copy(), params)
        assert len(result) <= len(imbalanced_df)

    def test_invalid_target_column_raises(self, imbalanced_df):
        params = UndersampleParams(target_column="nonexistent_col")
        with pytest.raises(ValueError, match="Undersampling: Target column"):
            undersample_data(imbalanced_df.copy(), params)

    def test_nan_in_target_raises(self):
        df = pd.DataFrame({
            "x": [1, 2, 3],
            "label": ["A", None, "B"],
        })
        with pytest.raises(ValueError, match="missing values"):
            undersample_data(df.copy(), UndersampleParams(target_column="label"))

    def test_columns_are_preserved(self, imbalanced_df):
        params = UndersampleParams(target_column="label")
        result = undersample_data(imbalanced_df.copy(), params)
        assert set(result.columns) == set(imbalanced_df.columns)


# ---------------------------------------------------------------------------
# run_pipeline  (integration-style)
# ---------------------------------------------------------------------------

class TestRunPipeline:
    """Integration tests for run_pipeline() — chains multiple steps."""

    def test_empty_steps_returns_original_data(self, clean_numeric_df):
        result = run_pipeline(clean_numeric_df.copy(), [])
        pd.testing.assert_frame_equal(result, clean_numeric_df)

    def test_single_normalize_step(self, clean_numeric_df):
        steps = [{"name": "Normalize", "params": {}}]
        result = run_pipeline(clean_numeric_df.copy(), steps)
        assert abs(result["a"].mean()) < 1e-9

    def test_single_scale_step(self, clean_numeric_df):
        steps = [{"name": "Scale", "params": {"min_value": 0, "max_value": 1}}]
        result = run_pipeline(clean_numeric_df.copy(), steps)
        assert result["a"].min() >= 0.0
        assert result["a"].max() <= 1.0

    def test_impute_then_normalize_chained(self, numeric_df):
        steps = [
            {"name": "Impute Missing Values", "params": {"strategy": "mean", "fill_value": 0.0}},
            {"name": "Normalize", "params": {}},
        ]
        result = run_pipeline(numeric_df.copy(), steps)
        assert result.isnull().sum().sum() == 0
        assert abs(result["a"].mean()) < 1e-9

    def test_unknown_step_name_is_silently_skipped(self, clean_numeric_df):
        steps = [{"name": "CompletelyFakeStep", "params": {}}]
        result = run_pipeline(clean_numeric_df.copy(), steps)
        pd.testing.assert_frame_equal(result, clean_numeric_df)

    def test_remove_nan_cols_step(self):
        df = pd.DataFrame({
            "good": [1, 2, 3, 4, 5],
            "bad":  [np.nan, np.nan, np.nan, np.nan, 1],
        })
        steps = [{"name": "Remove Columns with Excessive NaNs", "params": {"nan_threshold": 0.5}}]
        result = run_pipeline(df.copy(), steps)
        assert "bad" not in result.columns

    def test_remove_nan_rows_step(self):
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0],
            "b": [4.0, np.nan, 6.0],
            "c": [7.0, np.nan, 9.0],
        })
        steps = [{"name": "Remove Rows with Excessive NaNs", "params": {"nan_threshold": 0.5}}]
        result = run_pipeline(df.copy(), steps)
        assert len(result) == 2

    def test_dataset_multiplier_step(self, clean_numeric_df):
        steps = [{"name": "Dataset Multiplier", "params": {"size_multiplier": 2}}]
        result = run_pipeline(clean_numeric_df.copy(), steps)
        assert len(result) == len(clean_numeric_df) * 2

    def test_binarize_step_after_impute(self, numeric_df):
        steps = [
            {"name": "Impute Missing Values", "params": {"strategy": "mean", "fill_value": 0.0}},
            {"name": "Binarize", "params": {"threshold": 2.5}},
        ]
        result = run_pipeline(numeric_df.copy(), steps)
        unique_vals = set(result["a"].unique())
        assert unique_vals.issubset({0, 1, 0.0, 1.0})

    def test_one_hot_encoding_step(self, categorical_df):
        steps = [{"name": "One-Hot Encoding", "params": {}}]
        result = run_pipeline(categorical_df.copy(), steps)
        assert "cat" not in result.columns
        assert "val" in result.columns


class TestSanitizeForCsv:
    """Unit tests for sanitize_for_csv() mitigating CSV/DDE formula injection."""

    def test_escapes_formula_trigger_prefixes(self):
        formulas = [
            "=SUM(A1:A10)",
            "+cmd|'/c calc'!A0",
            "-2+3",
            "@SUM(1,2)",
            "\tcmd",
            "\rcalc",
        ]
        safe = ["normal text", "hello world", "test_123", "SAFE", "UPPER", "lower"]
        df = pd.DataFrame({"formulas": formulas, "safe": safe})
        result = sanitize_for_csv(df)

        assert result.loc[0, "formulas"] == "'=SUM(A1:A10)"
        assert result.loc[1, "formulas"] == "'+cmd|'/c calc'!A0"
        assert result.loc[2, "formulas"] == "'-2+3"
        assert result.loc[3, "formulas"] == "'@SUM(1,2)"
        assert result.loc[4, "formulas"] == "'\tcmd"
        assert result.loc[5, "formulas"] == "'\rcalc"

        # Safe values must not be modified
        for idx in range(6):
            assert result.loc[idx, "safe"] == df.loc[idx, "safe"]

    def test_preserves_numeric_and_null_values(self):
        df = pd.DataFrame({
            "ints": [1, -5, 42],
            "floats": [1.5, -2.5, np.nan],
            "bools": [True, False, True],
        })
        result = sanitize_for_csv(df)

        assert result.loc[0, "ints"] == 1
        assert result.loc[1, "ints"] == -5
        assert result.loc[2, "ints"] == 42
        assert result.loc[0, "floats"] == 1.5
        assert result.loc[1, "floats"] == -2.5
        assert pd.isna(result.loc[2, "floats"])
        assert bool(result.loc[0, "bools"]) is True
        assert bool(result.loc[1, "bools"]) is False

    def test_handles_empty_dataframe(self):
        df = pd.DataFrame(columns=["a", "b"])
        result = sanitize_for_csv(df)
        assert result.empty
        assert list(result.columns) == ["a", "b"]

    def test_does_not_mutate_original_dataframe(self):
        df = pd.DataFrame({"col": ["=1+1", "safe"]})
        result = sanitize_for_csv(df)
        assert df.loc[0, "col"] == "=1+1"
        assert result.loc[0, "col"] == "'=1+1"

    def test_works_when_map_is_used_or_fallback(self, monkeypatch):
        df = pd.DataFrame({"col": ["=test", "safe"]})

        # Test with standard map
        res1 = sanitize_for_csv(df)
        assert res1.loc[0, "col"] == "'=test"

        # Test fallback branch when hasattr(df, 'map') is False
        orig_hasattr = hasattr
        def fake_hasattr(obj, name):
            if name == "map" and isinstance(obj, pd.DataFrame):
                return False
            return orig_hasattr(obj, name)

        # Simulate older pandas having applymap
        if not hasattr(df, "applymap"):
            df.applymap = df.map

        monkeypatch.setattr("builtins.hasattr", fake_hasattr)
        res2 = sanitize_for_csv(df)
        assert res2.loc[0, "col"] == "'=test"
