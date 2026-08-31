import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Optional, Union
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
import xgboost as xgb
import matplotlib.pyplot as plt


# ==============================================================================
# 1. Custom Loss / Evaluation Metrics
# ==============================================================================

def asymmetric_prognostics_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Standard PHM / NASA Prognostics Metric (Asymmetric Penalty):
    Penalizes late predictions (overestimating RUL) more heavily than early predictions.
    d_i = y_pred - y_true
    score = sum(exp(d_i / 10) - 1) if d_i >= 0 else sum(exp(-d_i / 13) - 1)
    """
    d = np.array(y_pred) - np.array(y_true)
    score = 0.0
    for diff in d:
        if diff < 0:
            score += np.exp(-diff / 13.0) - 1.0
        else:
            score += np.exp(diff / 10.0) - 1.0
    return float(score)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, custom_loss_fn: Optional[Callable] = None) -> Dict[str, float]:
    """Calculate standard and optional custom evaluation metrics."""
    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "PHM_Score": asymmetric_prognostics_score(y_true, y_pred)
    }
    if custom_loss_fn is not None:
        metrics["Custom_Loss"] = float(custom_loss_fn(y_true, y_pred))
    return metrics


# ==============================================================================
# 2. Dataset Preparation & Feature Engineering
# ==============================================================================

def prepare_rul_dataset(
    loader,
    battery_ids: List[str],
    eol_threshold: float = 1.40
) -> pd.DataFrame:
    """
    Extracts cycle summaries and calculates ground-truth RUL for specified batteries.
    """
    all_records = []
    for b_id in battery_ids:
        df_summary = loader.extract_cycle_summary(b_id)
        
        # Filter valid discharge cycles
        dis = df_summary[(df_summary["type"] == "discharge") & (df_summary["capacity"].notna())].copy()
        dis["cycle_num"] = np.arange(1, len(dis) + 1)
        
        # Determine EOL cycle
        eol_candidates = dis[dis["capacity"] <= eol_threshold]["cycle_num"]
        eol_idx = eol_candidates.iloc[0] if len(eol_candidates) > 0 else dis["cycle_num"].max()
        
        # Ground truth RUL
        dis["RUL"] = (eol_idx - dis["cycle_num"]).clip(lower=0)
        
        # State of Health (SoH) = Current Capacity / Initial Nominal 2.0 Ah
        dis["SoH"] = dis["capacity"] / 2.0
        
        all_records.append(dis)
        
    return pd.concat(all_records, ignore_index=True)


# ==============================================================================
# 3. Model Factory
# ==============================================================================

def build_ml_model(model_type: str, hyperparams: Optional[Dict] = None):
    """
    Instantiates an ML regression model with user-provided hyperparameters.
    """
    params = hyperparams or {}
    model_type = model_type.lower()
    
    if model_type == "xgboost":
        return xgb.XGBRegressor(**params)
    elif model_type == "random_forest":
        return RandomForestRegressor(**params)
    elif model_type == "svr":
        return SVR(**params)
    elif model_type == "gradient_boosting":
        return GradientBoostingRegressor(**params)
    elif model_type == "ridge":
        return Ridge(**params)
    elif model_type == "elastic_net":
        return ElasticNet(**params)
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Choose from: 'xgboost', 'random_forest', 'svr', 'gradient_boosting', 'ridge', 'elastic_net'")


# ==============================================================================
# 4. Pipeline Runner
# ==============================================================================

class BatteryMLPipeline:
    """
    End-to-end training and evaluation pipeline.
    """

    def __init__(
        self,
        features: List[str],
        target: str = "RUL",
        scale_features: bool = True
    ):
        self.features = features
        self.target = target
        self.scale_features = scale_features
        self.scaler = StandardScaler() if scale_features else None
        self.model = None

    def fit(self, train_df: pd.DataFrame, model_type: str, hyperparams: Optional[Dict] = None):
        """Train the model on the training dataframe."""
        df_clean = train_df.dropna(subset=self.features + [self.target])
        X = df_clean[self.features].values
        y = df_clean[self.target].values

        if self.scale_features:
            X = self.scaler.fit_transform(X)

        self.model = build_ml_model(model_type, hyperparams)
        self.model.fit(X, y)
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """Generate predictions for a test dataframe."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call .fit() first.")
            
        df_clean = test_df.dropna(subset=self.features)
        X = df_clean[self.features].values
        if self.scale_features:
            X = self.scaler.transform(X)
            
        return self.model.predict(X)

    def evaluate(
        self,
        test_df: pd.DataFrame,
        custom_loss_fn: Optional[Callable] = None
    ) -> Tuple[Dict[str, float], np.ndarray]:
        """Evaluate the model and return metrics dict and predictions."""
        df_clean = test_df.dropna(subset=self.features + [self.target])
        y_true = df_clean[self.target].values
        y_pred = self.predict(df_clean)
        
        metrics = calculate_metrics(y_true, y_pred, custom_loss_fn=custom_loss_fn)
        return metrics, y_pred

