import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def regression_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
    }

def compare_results(results):
    return pd.DataFrame(results).sort_values("MAE")
