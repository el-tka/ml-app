"""
This module provides functionality to build a machine learning model pipeline 
for predicting rental prices.

It includes functions to load and prepare data, identify the best model, 
perform hyperparameter tuning using Optuna,
train the best model, evaluate its performance, and save the trained model 
along with relevant metrics and results.
"""

from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from loguru import logger
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from xgboost import XGBRegressor

from config import model_settings
from model.pipeline.preparation import feature_cols, prepare_data


def build_model() -> None:
    """
    Builds the machine learning model pipeline for predicting rental prices.

    It performs the following steps:
    1. Loads and prepares the data using the `prepare_data` function.
    2. Identifies the best model using cross-validation.
    3. Performs hyperparameter tuning using Optuna for the best model.
    4. Trains the best model on the training data.
    5. Evaluates the model's performance on the test data.
    6. Saves the trained model, best hyperparameters, evaluation metrics, 
    and results to the specified output directory.

    Returns:
        None
    """
    logger.info('Starting building the model pipeline')
    df = prepare_data()
    x, y = _get_x_y(df)
    x_train, x_test, y_train, y_test = _split_train_test(x, y)
    model_results, winner_name = _search_best_model(x_train, y_train)
    study = _tune_optuna(x_train, y_train, winner_name)
    best_model = _suggest_model(
        optuna.trial.FixedTrial(study.best_params),
        winner_name
    )
    best_model = _train_best_model(x_train, y_train, best_model)
    _predictions, metrics = _evaluate_model(best_model, x_test, y_test)
    _save_model(
        best_model, winner_name, study, metrics, model_results, feature_cols=feature_cols
    )


def _get_x_y(dataframe: pd.DataFrame,
             col_x: list = feature_cols,
             col_y: str = 'rent',
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extracts the feature columns (X) and target variable (y) from 
    the DataFrame.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing the features 
        and target variable.
        col_x (list): The list of feature column names to be used as 
        input features.
        col_y (str): The name of the target variable column.

    Returns:
        tuple: A tuple containing the feature DataFrame (X) and target Series (y).
    """
    logger.info(
        f'Extracting features and target variable. \n Features: {col_x} \n Target: {col_y}'
    )
    x = dataframe[col_x].apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(dataframe[col_y], errors='coerce')
    valid = y.notna()
    x, y = x.loc[valid], y.loc[valid]
    return x, y


def _split_train_test(x: pd.DataFrame,
                      y: pd.Series,
                      test_size: float = 0.2,
                      random_state: int = model_settings.random_state,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the data into training and testing sets.

    Args:
        x (pd.DataFrame): The feature DataFrame.
        y (pd.Series): The target variable Series.
        test_size (float): The proportion of the dataset to include in the 
        test split.
        random_state (int): Random state for reproducibility.
    
    Returns:
        tuple: A tuple containing the training and testing sets for 
        features (X_train, X_test) 
        and target variable (y_train, y_test).
    """
    logger.info('Splitting data into train and test sets.')
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    train_medians = x_train.median().fillna(0)
    x_train = x_train.fillna(train_medians)
    x_test = x_test.fillna(train_medians)
    return x_train, x_test, y_train, y_test


def _search_best_model(x_train: pd.DataFrame, 
                       y_train: pd.Series, 
                       n_splits: int = 5, 
                       random_state: int = model_settings.random_state,
) -> tuple[pd.DataFrame, str]:
    """
    Identifies the best model among a set of candidate models using 
    cross-validation.

    Args:
        x_train (pd.DataFrame): The training feature DataFrame.
        y_train (pd.Series): The training target variable Series.
        n_splits (int): The number of splits for cross-validation.
        random_state (int): Random state for reproducibility.
    
    Returns:
        tuple: A tuple containing a DataFrame with cross-validation results 
        for each model and the name of the best-performing model.
    """
    logger.info('Starting best model search using cross-validation')
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=1),
        'Gradient Boosting': GradientBoostingRegressor(random_state=random_state),
        'XGBoost': XGBRegressor(n_estimators=300, random_state=random_state, n_jobs=1, objective='reg:squarederror'),
        'LightGBM': LGBMRegressor(n_estimators=300, random_state=random_state, n_jobs=1, verbosity=-1),
        'CatBoost': CatBoostRegressor(iterations=300, random_seed=random_state, verbose=False, thread_count=1),
    }
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {'mae': 'neg_mean_absolute_error', 
               'rmse': 'neg_root_mean_squared_error', 
               'r2': 'r2'}
    rows = []
    for name, model in models.items():
        scores = cross_validate(model, x_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        rows.append({'model': name, 'CV_MAE': -scores['test_mae'].mean(),
                    'CV_RMSE': -scores['test_rmse'].mean(), 'CV_R2': scores['test_r2'].mean(),
                    'CV_RMSE_std': scores['test_rmse'].std()})
    model_results = pd.DataFrame(rows).sort_values('CV_RMSE').reset_index(drop=True)
    winner_name = model_results.loc[0, 'model']
    logger.debug(f"\nModel Comparison: {model_results.to_string(index=False)}")
    return model_results, winner_name


def _suggest_model(trial: optuna.trial.Trial, 
                   name: str, 
                   random_state: int = model_settings.random_state,
) -> object:
    """
    Suggests a model with hyperparameters based on the provided Optuna trial.

    Args:
        trial (optuna.trial.Trial): An Optuna trial object for hyperparameter optimization.
        name (str): The name of the model to suggest.
        random_state (int): Random state for reproducibility.

    Returns:
        object: An instance of the suggested model with optimized hyperparameters.
    """
    logger.info(f'Suggesting model: {name} with hyperparameters from Optuna trial')
    if name == 'Random Forest':
        return RandomForestRegressor(
            n_estimators=trial.suggest_int('n_estimators', 100, 800, step=100),
            max_depth=trial.suggest_int('max_depth', 3, 25),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
            max_features=trial.suggest_float('max_features', 0.5, 1.0),
            random_state=random_state, n_jobs=1)
    if name == 'Gradient Boosting':
        return GradientBoostingRegressor(
            n_estimators=trial.suggest_int('n_estimators', 100, 700, step=100),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            max_depth=trial.suggest_int('max_depth', 2, 7),
            subsample=trial.suggest_float('subsample', 0.6, 1.0),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
            random_state=random_state)
    if name == 'XGBoost':
        return XGBRegressor(
            n_estimators=trial.suggest_int('n_estimators', 100, 800, step=100),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            max_depth=trial.suggest_int('max_depth', 2, 10),
            subsample=trial.suggest_float('subsample', 0.6, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
            reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 20, log=True),
            objective='reg:squarederror', random_state=random_state, n_jobs=1)
    if name == 'LightGBM':
        return LGBMRegressor(
            n_estimators=trial.suggest_int('n_estimators', 100, 800, step=100),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            num_leaves=trial.suggest_int('num_leaves', 7, 63),
            min_child_samples=trial.suggest_int('min_child_samples', 5, 50),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
            reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 20, log=True),
            random_state=random_state, n_jobs=1, verbosity=-1)
    if name == 'CatBoost':
        return CatBoostRegressor(
            iterations=trial.suggest_int('iterations', 100, 800, step=100),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            depth=trial.suggest_int('depth', 3, 9),
            l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1, 20, log=True),
            random_seed=random_state, verbose=False, thread_count=1)
    raise ValueError(name)


def _define_objective(trial: optuna.trial.Trial, 
                      x_train: pd.DataFrame, 
                      y_train: pd.Series, 
                      winner_name: str,
) -> float:
    """
    Defines the objective function for Optuna hyperparameter tuning.
    """
    model = _suggest_model(trial, winner_name)
    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=model_settings.random_state
    )
    scores = cross_val_score(model, x_train, y_train, cv=cv,
                             scoring='neg_root_mean_squared_error', n_jobs=1)
    
    logger.debug(f"Cross-validation scores for {winner_name}: {-scores.mean()}")
    return -scores.mean()


def _tune_optuna(x_train: pd.DataFrame, 
                  y_train: pd.Series, 
                  winner_name: str,
) -> optuna.study.Study:
    """
    Performs hyperparameter tuning using Optuna for the best model identified.

    Args:
        x_train (pd.DataFrame): The training feature DataFrame.
        y_train (pd.Series): The training target variable Series.
        winner_name (str): The name of the best-performing model.

    Returns:
        optuna.study.Study: An Optuna study object containing the results of the hyperparameter tuning.
    """
    logger.info(f'Starting hyperparameter tuning for {winner_name} using Optuna')
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
    lambda trial: _define_objective(
        trial,
        x_train,
        y_train,
        winner_name,
    ),
    n_trials=10
)
    logger.debug(f'Winner: {winner_name}')
    logger.debug(f'Optimal CV RMSE: {study.best_value}')
    logger.debug(f'Best Parameters: {study.best_params}')
    return study


def _train_best_model(x_train: pd.DataFrame, 
                      y_train: pd.Series, 
                      best_model: object,
) -> object:
    """
    Trains the best model on the training data.

    Args:
        x_train (pd.DataFrame): The training feature DataFrame. 
        y_train (pd.Series): The training target variable Series.
        best_model (object): The best model instance with optimized hyperparameters.

    Returns:
        object: The trained best model.
    """
    logger.info(f'Training the best model: {best_model.__class__.__name__}')
    best_model.fit(x_train, y_train)
    return best_model


def _evaluate_model(best_model: object, 
                    x_test: pd.DataFrame, 
                    y_test: pd.Series,
) -> tuple[np.ndarray, dict]:
    """
    Evaluates the trained model on the test data and calculates performance metrics.

    Args:
        best_model (object): The trained best model.
        x_test (pd.DataFrame): The test feature DataFrame.
        y_test (pd.Series): The test target variable Series.

    Returns:
        tuple: A tuple containing the predictions made by the model and a dictionary of evaluation 
        metrics (MAE, RMSE, R2).
    """
    logger.info(f'Evaluating the model: {best_model.__class__.__name__}')
    predictions = best_model.predict(x_test)
    
    metrics = {
        "Test_MAE": mean_absolute_error(
            y_test, predictions
        ),
        "Test_RMSE": mean_squared_error(
            y_test, predictions
        ) ** 0.5,
        "Test_R2": r2_score(
            y_test, predictions
        )
    }

    logger.debug(f"\nTest Metrics: {pd.Series(metrics)}")
    return predictions, metrics


def _save_model(best_model: object, 
                winner_name: str, 
                study: optuna.Study, 
                metrics: dict, 
                model_results: pd.DataFrame, 
                feature_cols: list = feature_cols,
) -> None:
    """
    Saves the trained model, best hyperparameters, evaluation metrics, and results 
    to the specified output directory.

    Args:
        best_model (object): The trained best model.
        winner_name (str): The name of the best-performing model.
        study (optuna.Study): An Optuna study object containing the results of hyperparameter tuning.
        metrics (dict): A dictionary containing evaluation metrics (MAE, RMSE, R2).
        model_results (pd.DataFrame): A DataFrame containing cross-validation results for each model.
        feature_cols (list): A list of feature column names used in the model.

    Returns:
        None
    """
    logger.info(f'Saving the model: {best_model.__class__.__name__}')
    
    output_dir = Path(model_settings.model_output_path)   # model_output_path
    results_dir = output_dir / model_settings.model_results_path   # model_results_path

    results_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": best_model,
        "model_name": winner_name,
        "features": feature_cols,
        "best_params": study.best_params,
        "test_metrics": metrics
    }
 
    with open(output_dir / model_settings.model_name, 'wb') as model_file:  # model_name
        joblib.dump(artifact, model_file)

    with open(results_dir / model_settings.model_results_name, 'w') as results_file:  # model_results_name
        model_results.to_csv(results_file, index=False)

    logger.info(f'Model and results saved successfully to the directory: {output_dir} and {results_dir}')


if __name__ == "__main__":
    build_model()
