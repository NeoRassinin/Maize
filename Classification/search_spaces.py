import optuna

space_linear_regression = {
    'alpha': optuna.distributions.LogUniformDistribution(1e-5, 1e-1)
}

space_random_forest = {
    'n_estimators': optuna.distributions.IntUniformDistribution(10, 1000),
    'max_depth': optuna.distributions.IntUniformDistribution(1, 100),
    'min_samples_split': optuna.distributions.IntUniformDistribution(2, 20),
    'min_samples_leaf': optuna.distributions.IntUniformDistribution(1, 10)
}

space_svr = {
    'C': optuna.distributions.IntUniformDistribution(1e-2, 1e2),
    'gamma': optuna.distributions.IntUniformDistribution(1e-4, 1.0),
    'epsilon': optuna.distributions.IntUniformDistribution(0.01, 0.5)
}


space_adaboost = {
    'n_estimators': optuna.distributions.IntUniformDistribution(10, 300),
    'learning_rate': optuna.distributions.FloatDistribution(1e-3, 1.0, log=True),
    'max_depth': optuna.distributions.IntUniformDistribution(1, 6)
}