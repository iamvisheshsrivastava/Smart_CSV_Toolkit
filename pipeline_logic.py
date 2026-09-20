"""
pipeline_logic.py
=================
Defines every data-cleaning / transformation operation that can be applied
to a pandas DataFrame, plus the `run_pipeline` orchestrator that chains them.

Each operation follows the same contract:
    func(data: pd.DataFrame, params: <ParamsDataclass>) -> pd.DataFrame

A matching ``OPERATION_MAP`` dictionary maps step names (as they appear in
``table_steps.json``) to their implementation functions so that
``run_pipeline`` can dispatch dynamically.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
from safe_exec import safe_exec
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Binarizer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

@dataclass
class SMOTEParams:
    target_column: str
    sampling_strategy: str = "auto"
    k_neighbors: int = 5

@dataclass
class OversampleParams:
    target_column: str
    sampling_strategy: str = "minority"

@dataclass
class UndersampleParams:
    target_column: str
    sampling_strategy: str = "majority"

@dataclass
class ImputeMissingValuesParams:
    strategy: str = "mean"
    fill_value: float = 0.0

@dataclass
class NormalizeParams:
    pass

@dataclass
class ScaleParams:
    min_value: int = 0
    max_value: int = 1

@dataclass
class LLMCleaningParams:
    code: str

@dataclass
class BinarizeParams:
    threshold: float = 0.0

@dataclass
class RemoveOutliersParams:
    iqr_multiplier: float = 1.5

@dataclass
class RemoveNaNColsParams:
    nan_threshold: float = 0.5

@dataclass
class RemoveNaNRowsParams:
    nan_threshold: float = 0.5

@dataclass
class DatasetMultiplierParams:
    size_multiplier: int = 1

@dataclass
class AddNoiseParams:
    noise_factor: float = 0.1

def impute_missing(data: pd.DataFrame, params: ImputeMissingValuesParams) -> pd.DataFrame:
    """Fill missing values in numeric (and optionally all) columns.

    Strategies:
        - ``"mean"`` / ``"median"`` / ``"constant"`` – applied to numeric
          columns only via :class:`sklearn.impute.SimpleImputer`.
        - ``"most_frequent"`` – applied across *all* columns (works for both
          numeric and categorical/object dtypes).

    Args:
        data: Input DataFrame.  Modified in place and returned.
        params: Configuration dataclass with ``strategy`` and ``fill_value``.

    Returns:
        DataFrame with NaNs filled.

    Raises:
        ValueError: If the strategy is invalid, or if mean/median/constant is
            requested but no numeric columns are present.
    """
    strategy = params.strategy
    fill_value = params.fill_value
    numeric_cols = data.select_dtypes(include=np.number).columns
    if strategy in ["mean", "median", "constant"]:
        if numeric_cols.empty:
            raise ValueError(f"No numeric columns found for strategy '{strategy}'.")
        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
    elif strategy == "most_frequent":
        # Impute numeric and non-numeric columns as two separate groups
        # rather than the whole frame at once. SimpleImputer.fit_transform
        # on a DataFrame with mixed dtypes returns a single homogeneous
        # numpy array (dtype=object), and assigning that back into
        # data[data.columns] silently casts EVERY column — including
        # originally-numeric ones — to object dtype. That corrupts any
        # later pipeline step relying on select_dtypes(include=np.number)
        # (Normalize, Scale, Binarize, Remove Outliers, Add Noise, SMOTE,
        # Oversample, Undersample all use it): numeric_cols comes back
        # empty and the step silently no-ops or raises a confusing sklearn
        # error instead of doing what the user asked.
        other_cols = [c for c in data.columns if c not in numeric_cols]
        if len(numeric_cols) > 0:
            data[numeric_cols] = SimpleImputer(strategy=strategy).fit_transform(data[numeric_cols])
        if other_cols:
            data[other_cols] = SimpleImputer(strategy=strategy).fit_transform(data[other_cols])
    else:
        raise ValueError(f"Unsupported imputation strategy: {strategy}")
    return data


def normalize(data: pd.DataFrame, params: NormalizeParams) -> pd.DataFrame:
    """Standardise all numeric columns to zero mean and unit variance (Z-score).

    Uses :class:`sklearn.preprocessing.StandardScaler` which divides by the
    population standard deviation (``ddof=0``).

    Args:
        data: Input DataFrame.
        params: ``NormalizeParams`` (no configurable options).

    Returns:
        DataFrame with numeric columns standardised.
    """
    scaler = StandardScaler()
    numeric_cols = data.select_dtypes(include=np.number).columns
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    return data

def scale(data: pd.DataFrame, params: ScaleParams) -> pd.DataFrame:
    """Rescale all numeric columns to a specified [min, max] range (MinMax scaling).

    Uses :class:`sklearn.preprocessing.MinMaxScaler`.

    Args:
        data: Input DataFrame.
        params: ``ScaleParams`` with ``min_value`` and ``max_value``.

    Returns:
        DataFrame with numeric columns rescaled.
    """
    scaler = MinMaxScaler(feature_range=(params.min_value, params.max_value))
    numeric_cols = data.select_dtypes(include=np.number).columns
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    return data

def llm_data_cleaning(data: pd.DataFrame, params: Any) -> pd.DataFrame:
    """
    Applies LLM-generated dynamic cleaning code to the given DataFrame.
    Expects 'code' inside params, which should be a string of Python code
    that modifies the variable `df` in-place.
    """
    code = getattr(params, "code", None)
    if not code:
        raise ValueError("LLM Data Cleaning failed: No code provided in step parameters.")

    try:
        local_vars = {"df": data.copy()}
        global_vars = {
            "pd": pd,
            "np": np,
            "SimpleImputer": SimpleImputer,
            "StandardScaler": StandardScaler,
            "MinMaxScaler": MinMaxScaler,
            "Binarizer": Binarizer,
            "SMOTE": SMOTE,
            "RandomOverSampler": RandomOverSampler,
            "RandomUnderSampler": RandomUnderSampler
        }
        safe_exec(code, global_vars, local_vars)
        return local_vars["df"]
    except Exception as e:
        raise RuntimeError(
            f"LLM Data Cleaning failed to execute.\n\n"
            f"Please verify the generated code or revise the cleaning suggestion.\n\n"
            f"Error: {str(e)}"
        )

def one_hot_encode(data: pd.DataFrame, params: Any) -> pd.DataFrame:
    """One-hot encode every object/categorical column using ``pd.get_dummies``.

    All string columns are expanded into binary indicator columns; numeric
    columns are left unchanged.

    Args:
        data: Input DataFrame.
        params: Not used (accepted for interface consistency).

    Returns:
        DataFrame with categorical columns replaced by binary dummies.
    """
    categorical_cols = data.select_dtypes(include='object').columns
    return pd.get_dummies(data, columns=categorical_cols)

def binarize(data: pd.DataFrame, params: BinarizeParams) -> pd.DataFrame:
    """Threshold numeric columns to binary 0/1 values.

    Values strictly greater than ``threshold`` become 1; all others become 0.
    NaNs in numeric columns cause an early failure with a helpful message —
    run ``impute_missing`` first if needed.

    Args:
        data: Input DataFrame.
        params: ``BinarizeParams`` with a ``threshold`` float.

    Returns:
        DataFrame with numeric columns binarized.

    Raises:
        ValueError: If any numeric column still contains NaN values.
    """
    binarizer = Binarizer(threshold=params.threshold)
    numeric_cols = data.select_dtypes(include=np.number).columns

    if data[numeric_cols].isnull().any().any():
        raise ValueError(
            f"Binarization failed: Numeric columns contain NaNs. "
            f"Please impute missing values first. Columns: "
            f"{data[numeric_cols].columns[data[numeric_cols].isnull().any()].tolist()}"
        )
    data[numeric_cols] = binarizer.fit_transform(data[numeric_cols])
    return data

def remove_outliers(data: pd.DataFrame, params: RemoveOutliersParams) -> pd.DataFrame:
    """Remove rows where any numeric value falls outside the IQR-based fence.

    For each numeric column, the lower fence is ``Q1 - k*IQR`` and the upper
    fence is ``Q3 + k*IQR``, where ``k = iqr_multiplier`` (default 1.5).

    Args:
        data: Input DataFrame.
        params: ``RemoveOutliersParams`` with ``iqr_multiplier``.

    Returns:
        DataFrame with outlier rows removed.
    """
    numeric_cols = data.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - params.iqr_multiplier * IQR
        upper = Q3 + params.iqr_multiplier * IQR
        data = data[(data[col] >= lower) & (data[col] <= upper)]
    return data

def remove_nan_cols(data: pd.DataFrame, params: RemoveNaNColsParams) -> pd.DataFrame:
    """Drop columns whose fraction of NaN values exceeds the given threshold.

    Args:
        data: Input DataFrame.
        params: ``RemoveNaNColsParams`` with ``nan_threshold`` (0–1).

    Returns:
        DataFrame with high-NaN columns removed.
    """
    return data.loc[:, data.isnull().mean() < params.nan_threshold]

def remove_nan_rows(data: pd.DataFrame, params: RemoveNaNRowsParams) -> pd.DataFrame:
    """Drop rows whose fraction of NaN values exceeds the given threshold.

    Args:
        data: Input DataFrame.
        params: ``RemoveNaNRowsParams`` with ``nan_threshold`` (0–1).

    Returns:
        DataFrame with high-NaN rows removed.
    """
    return data.loc[data.isnull().mean(axis=1) < params.nan_threshold]

def dataset_multiplier(data: pd.DataFrame, params: DatasetMultiplierParams) -> pd.DataFrame:
    """Duplicate the dataset N times by stacking copies of the DataFrame.

    Useful for augmenting small datasets before training.  The integer
    multiplier can be supplied as a plain int or as a string like ``"2x"``.

    Args:
        data: Input DataFrame.
        params: ``DatasetMultiplierParams`` with ``size_multiplier``.

    Returns:
        DataFrame with ``len(data) * multiplier`` rows and a reset index.

    Raises:
        ValueError: If ``size_multiplier`` cannot be parsed as an integer.
    """
    try:
        multiplier = int(str(params.size_multiplier).replace("x", ""))
    except Exception:
        raise ValueError(
            f"Invalid multiplier value: '{params.size_multiplier}'. "
            "Expected format like '1x', '2x', ..., '5x'."
        )
    return pd.concat([data] * multiplier, ignore_index=True)

def add_noise(data: pd.DataFrame, params: AddNoiseParams) -> pd.DataFrame:
    """Add Gaussian noise to every numeric column.

    Draws noise from ``N(0, noise_factor)`` and adds it element-wise.  A
    ``noise_factor`` of 0 is a no-op.

    Args:
        data: Input DataFrame.
        params: ``AddNoiseParams`` with ``noise_factor`` (standard deviation).

    Returns:
        DataFrame with noise injected into numeric columns.
    """
    numeric_cols = data.select_dtypes(include=np.number).columns
    noise = np.random.normal(loc=0.0, scale=params.noise_factor, size=data[numeric_cols].shape)
    data[numeric_cols] = data[numeric_cols] + noise
    return data

def apply_smote(data: pd.DataFrame, params: SMOTEParams) -> pd.DataFrame:
    if params.target_column not in data.columns:
        raise ValueError(f"SMOTE: Target column '{params.target_column}' not found.")
    X = data.drop(columns=[params.target_column])
    y = data[params.target_column]
    smote = SMOTE(sampling_strategy=params.sampling_strategy, k_neighbors=params.k_neighbors)
    X_res, y_res = smote.fit_resample(X, y)
    return pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=params.target_column)], axis=1)

def oversample_data(data: pd.DataFrame, params: OversampleParams) -> pd.DataFrame:
    if params.target_column not in data.columns:
        raise ValueError(f"Oversampling: Target column '{params.target_column}' not found.")
    y = data[params.target_column]
    X = data.drop(columns=[params.target_column])
    if y.isnull().any():
        raise ValueError(
            f"Oversampling skipped: Target column '{params.target_column}' contains missing values (NaNs). "
            "Please impute or remove them before applying oversampling."
        )
    try:
        y = y.astype(str)
        sampler = RandomOverSampler(sampling_strategy=params.sampling_strategy)
        X_res, y_res = sampler.fit_resample(X, y)
        return pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=params.target_column)], axis=1)
    except Exception as e:
        raise ValueError(
            f"Oversampling failed. Ensure the target column contains consistent class labels (no mixed types).\n\n"
            f"Details: {str(e)}"
        )

def undersample_data(data: pd.DataFrame, params: UndersampleParams) -> pd.DataFrame:
    if params.target_column not in data.columns:
        raise ValueError(f"Undersampling: Target column '{params.target_column}' not found.")
    y = data[params.target_column]
    X = data.drop(columns=[params.target_column])
    if y.isnull().any():
        raise ValueError(
            f"Undersampling skipped: Target column '{params.target_column}' contains missing values (NaNs). "
            "Please clean or impute the target column before applying undersampling."
        )
    try:
        y = y.astype(str)
        sampler = RandomUnderSampler(sampling_strategy=params.sampling_strategy)
        X_res, y_res = sampler.fit_resample(X, y)
        return pd.concat(
            [pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=params.target_column)],
            axis=1
        )
    except Exception as e:
        raise ValueError(
            f"Undersampling failed. Make sure the target column contains clean and consistent class labels.\n\n"
            f"Details: {str(e)}"
        )

OPERATION_MAP = {
    "Impute Missing Values": impute_missing,
    "Normalize": normalize,
    "Scale": scale,
    "LLM Data Cleaning": llm_data_cleaning,
    "One-Hot Encoding": one_hot_encode,
    "Binarize": binarize,
    "Remove Outliers": remove_outliers,
    "Remove Columns with Excessive NaNs": remove_nan_cols,
    "Remove Rows with Excessive NaNs": remove_nan_rows,
    "SMOTE": apply_smote,
    "Dataset Multiplier": dataset_multiplier,
    "Add Noise": add_noise,
    "Oversample": oversample_data,
    "Undersample": undersample_data
}

def run_pipeline(data: pd.DataFrame, steps: List[Dict[str, Any]]) -> pd.DataFrame:
    """Execute a sequence of named cleaning steps on a DataFrame.

    Each element of ``steps`` is a dict like::

        {"name": "Impute Missing Values", "params": {"strategy": "mean", "fill_value": 0.0}}

    Step names must match the keys in ``OPERATION_MAP``.  Unknown names are
    silently skipped so the pipeline can continue even when a step is
    misconfigured.

    Args:
        data: The DataFrame to process (a copy is typically passed by the
            caller; this function mutates and returns it).
        steps: Ordered list of step descriptors.

    Returns:
        Cleaned / transformed DataFrame after all recognised steps are applied.
    """
    for step in steps:
        name = step["name"]
        params_dict = step.get("params", {})

        if name == "Impute Missing Values":
            params = ImputeMissingValuesParams(**params_dict)
        elif name == "Normalize":
            params = NormalizeParams()
        elif name == "Scale":
            params = ScaleParams(**params_dict)
        elif name == "LLM Data Cleaning":
            params = LLMCleaningParams(**params_dict)
        elif name == "One-Hot Encoding":
            params = {}
        elif name == "Binarize":
            params = BinarizeParams(**params_dict)
        elif name == "Remove Outliers":
            params = RemoveOutliersParams(**params_dict)
        elif name == "Remove Columns with Excessive NaNs":
            params = RemoveNaNColsParams(**params_dict)
        elif name == "Remove Rows with Excessive NaNs":
            params = RemoveNaNRowsParams(**params_dict)
        elif name == "SMOTE":
            params = SMOTEParams(**params_dict)
        elif name == "Dataset Multiplier":
            params = DatasetMultiplierParams(**params_dict)
        elif name == "Add Noise":
            params = AddNoiseParams(**params_dict)
        elif name == "Oversample":
            params = OversampleParams(**params_dict)
        elif name == "Undersample":
            params = UndersampleParams(**params_dict)
        else:
            # Unknown step name — skip gracefully so the rest of the pipeline
            # can still run.  The UI surfaces a warning separately if needed.
            continue
        data = OPERATION_MAP[name](data, params)

    return data
