# Загружаем данные из CSV-файла
df_lee = pd.read_csv(WORK_DIR + '/data_zhang.csv')
df_zhang = pd.read_csv(WORK_DIR + '/data_lee.csv')
df_lee['max-box-side'] = df_lee[['width', 'height']].max(axis=1)
df_zhang['max-box-side'] = df_zhang[['width', 'height']].max(axis=1)
df_lee = df_lee.fillna(0)
df_zhang = df_zhang.fillna(0)


train_cols = cols[1:-1] + ['max-box-side'] + ['avg_angle']

X_lee = df_lee[train_cols]
y_lee = df_lee['target']


def objective_linear_regression(trial, X, y):
    # Определяем гиперпараметры
    alpha = trial.suggest_float('alpha', 1e-5, 1e-1)

    # Определяем модель
    model = Ridge(alpha=alpha)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучаем модель
    model.fit(X_train, y_train)

    # Вычисляем значение метрики качества на тестовой выборке
    y_pred = model.predict(X_test)
    score = np.mean((y_test - y_pred) ** 2)

    return score

def objective_random_forest(trial, X, y):
    # Определяем гиперпараметры
    n_estimators = trial.suggest_int('n_estimators', 10, 300)
    max_depth = trial.suggest_int('max_depth', 1, 6)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

    # Определяем модель
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучаем модель
    model.fit(X_train, y_train)

    # Вычисляем значение метрики качества на тестовой выборке
    y_pred = model.predict(X_test)
    score = np.mean((y_test - y_pred) ** 2)

    return score

def objective_polynomial_regression(trial, X, y):
    # Определяем гиперпараметры
    degree = trial.suggest_int('degree', 1, 3)

    # Определяем модель
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = Ridge()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучаем модель
    model.fit(X_train, y_train)

    # Вычисляем значение метрики качества на тестовой выборке
    y_pred = model.predict(X_test)
    score = np.mean((y_test - y_pred) ** 2)

    return score

def objective_gradient_boosting(trial, X, y):
    # Определяем гиперпараметры
    n_estimators = trial.suggest_int('n_estimators', 10, 300)
    max_depth = trial.suggest_int('max_depth', 1, 6)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-4)

    # Определяем модель
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        learning_rate=learning_rate
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучаем модель
    model.fit(X_train, y_train)

    # Вычисляем значение метрики качества на тестовой выборке
    y_pred = model.predict(X_test)
    score = np.mean((y_test - y_pred) ** 2)

    return score

space_linear_regression = {
    'alpha': optuna.distributions.LogUniformDistribution(1e-5, 1e-1)
}

space_random_forest = {
    'n_estimators': optuna.distributions.IntUniformDistribution(10, 1000),
    'max_depth': optuna.distributions.IntUniformDistribution(1, 100),
    'min_samples_split': optuna.distributions.IntUniformDistribution(2, 20),
    'min_samples_leaf': optuna.distributions.IntUniformDistribution(1, 10)
}

space_polynomial_regression = {
    'degree': optuna.distributions.IntUniformDistribution(1, 3)
}

space_gradient_boosting = {
    'n_estimators': optuna.distributions.IntUniformDistribution(10, 1000),
    'max_depth': optuna.distributions.IntUniformDistribution(1, 100),
    'min_samples_split': optuna.distributions.IntUniformDistribution(2, 20),
    'min_samples_leaf': optuna.distributions.IntUniformDistribution(1, 10),
    'learning_rate': optuna.distributions.LogUniformDistribution(1e-5, 1e-1)
}

# Запускаем оптимизацию для каждой модели и для каждой выборки
list_of_best_params_lee = []
list_of_best_params_zhang = []
models = [
    ('linear_regression', objective_linear_regression, space_linear_regression),
    ('polynomial_regression', objective_polynomial_regression, space_polynomial_regression),
    ('gradient_boosting', objective_gradient_boosting, space_gradient_boosting),
    ('random_forest', objective_random_forest, space_random_forest)
]

# Запускаем оптимизацию для каждой модели и для каждой выборки
for i in range(len(models)):
    model_name, objective_func, space = models[i]
    print(f'Optimizing {model_name} for Lee data...')
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective_func(trial, X_best_lee, y_best_lee), n_trials=20)
    best_params = {name: value for name, value in study.best_trial.params.items()}
    print(f'Best params for {model_name}: {best_params}')
    list_of_best_params_lee.append(best_params)

    print(f'Optimizing {model_name} for Zhang data...')
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective_func(trial, X_best_zhang, y_best_zhang), n_trials=20)
    best_params = {name: value for name, value in study.best_trial.params.items()}
    print(f'Best params for {model_name}: {best_params}')
    list_of_best_params_zhang.append(best_params)

lr_lee = Ridge(**list_of_best_params_lee[0])
lr_lee.fit(X_best_lee, y_best_lee)

gb_lee = GradientBoostingRegressor(**list_of_best_params_lee[2])
gb_lee.fit(X_best_lee, y_best_lee)

rf_lee = RandomForestRegressor(**list_of_best_params_lee[3])
rf_lee.fit(X_best_lee, y_best_lee)

# Сохраняем модель в файл
def save_model():
    with open("model.pkl", "wb") as f:
        pickle.dump(save_models(), f)