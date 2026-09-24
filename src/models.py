from dataclasses import dataclass
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, BayesianRidge, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

@dataclass
class ModelBundle:
    name: str
    model: object

def build_models():
    return [
        ModelBundle(
            "Polynomial Regression",
            Pipeline([
                ("poly", PolynomialFeatures(degree=3, include_bias=False)),
                ("model", LinearRegression())
            ])
        ),
        ModelBundle(
            "Bayesian Ridge",
            Pipeline([
                ("poly", PolynomialFeatures(degree=3, include_bias=False)),
                ("model", BayesianRidge())
            ])
        ),
        ModelBundle(
            "Random Forest",
            Pipeline([
                ("scale", StandardScaler()),
                ("model", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
            ])
        ),
        ModelBundle(
            "Gradient Boosting",
            Pipeline([
                ("scale", StandardScaler()),
                ("model", GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42))
            ])
        ),
        ModelBundle(
            "Ridge Regression",
            Pipeline([
                ("poly", PolynomialFeatures(degree=3, include_bias=False)),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=1.0))
            ])
        ),
        ModelBundle(
            "SVR",
            Pipeline([
                ("scale", StandardScaler()),
                ("model", SVR(kernel="rbf", C=1e5, gamma=0.1, epsilon=1))
            ])
        ),
    ]

