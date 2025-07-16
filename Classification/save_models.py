from data_loader import load_data, prepare_features_targets
from objectives import objective_svr,objective_adaboost,objective_random_forest,objective_linear_regression
from search_spaces import space_svr, space_adaboost, space_random_forest, space_linear_regression
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import optuna
from io_utils import save_models


df_lee = load_data("data_lee.csv")
X_lee, y_lee = prepare_features_targets(df_lee)

models = [
    ('linear', objective_linear_regression, space_linear_regression),
    ('svr', objective_svr, space_svr),
    ('adaboost', objective_adaboost, space_adaboost),
    ('rf', objective_random_forest, space_random_forest)
]

# Запускаем оптимизацию для каждой модели и для каждой выборки
list_of_best_params_lee = []
for i in range(len(models)):
    model_name, objective_func, space = models[i]
    print(f'Optimizing {model_name} for Lee data...')
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective_func(trial, X_lee, y_lee), n_trials=20)
    best_params = {name: value for name, value in study.best_trial.params.items()}
    print(f'Best params for {model_name}: {best_params}')
    list_of_best_params_lee.append(best_params)

lr = Ridge(**list_of_best_params_lee[0])
lr.fit(X_lee, y_lee)

svr = SVR(**list_of_best_params_lee[1])
svr.fit(X_lee, y_lee)

adaboost = AdaBoostRegressor(**list_of_best_params_lee[2])
adaboost.fit(X_lee, y_lee)

rf = RandomForestRegressor(**list_of_best_params_lee[3])
rf.fit(X_lee, y_lee)

models_dict = {
    "random_forest": rf,
    "linear_regression": lr,
    "svr": svr,
    "adaboost": adaboost
}

save_models(models_dict, dir_path="saved_models")
