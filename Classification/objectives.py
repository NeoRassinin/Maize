import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split


def _evaluate_model(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return np.mean((y_test - y_pred) ** 2)

def objective_linear_regression(trial, X, y):
    """
    Optuna objective-функция для подбора гиперпараметра модели Ridge.
    """
    alpha = trial.suggest_float("alpha", 1e-5, 1e-1)

    model = Ridge(alpha=alpha)

    score = _evaluate_model(model,X,y)
    return score


def objective_random_forest(trial, X, y):
    """
    Optuna objective-функция для подбора гиперпараметров RandomForest.
    """
    n_estimators = trial.suggest_int("n_estimators", 10, 300)
    max_depth = trial.suggest_int("max_depth", 1, 6)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    score = _evaluate_model(model,X,y)

    return score


def objective_svr(trial, X, y):
    """
    Optuna objective-функция для подбора гиперпараметров SVR.
    """
    C = trial.suggest_float("C", 1e-2, 1e2, log=True)
    gamma = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
    epsilon = trial.suggest_float("epsilon", 0.01, 0.5)

    model = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon)

    score = _evaluate_model(model,X,y)

    return score


def objective_adaboost(trial, X, y):
    """
    Optuna objective-функция для подбора гиперпараметров AdaBoost.
    """
    n_estimators = trial.suggest_int("n_estimators", 10, 300)
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 1.0)
    max_depth = trial.suggest_int("max_depth", 1, 6)

    base_estimator = DecisionTreeRegressor(max_depth=max_depth)

    model = AdaBoostRegressor(
        estimator=base_estimator,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42
    )
    score = _evaluate_model(model,X,y)
    return score


