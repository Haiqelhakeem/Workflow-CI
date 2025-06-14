import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import mlflow
import mlflow.sklearn
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import argparse
import os
import dagshub
dagshub.init(repo_owner='Haiqelhakeem', repo_name='msml-haiqel-aziizul-hakeem', mlflow=True)

def load_and_prepare_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Data file '{file_path}' not found.")
        exit()

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Split data before scaling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scaling is optional for RF but retained for consistency
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def objective(params):
    with mlflow.start_run(nested=True):
        # Hyperopt passes float values, so we must cast integers
        params['n_estimators'] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])
        params['min_samples_split'] = int(params['min_samples_split'])
        params['min_samples_leaf'] = int(params['min_samples_leaf'])

        mlflow.log_params(params)
        print(f"\n--- Trying params: {params} ---")

        model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,  # Use all available cores
            **params
        )
        
        model.fit(X_train, y_train)

        # Predict probabilities for the positive class (Churn)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        mlflow.log_metric("test_auc", auc)
        print(f"AUC: {auc:.4f}")
        
        # Log the model from this trial for potential future inspection
        mlflow.sklearn.log_model(model, "rf-model")

        # Hyperopt aims to MINIMIZE the loss, so we return 1 - AUC.
        return {'loss': 1 - auc, 'status': STATUS_OK, 'auc': auc}

def main(max_evals, data_path):
    # Load data once to be globally available to the objective function
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = load_and_prepare_data(data_path)
    
    mlflow.set_experiment("Churn Prediction RF Tuning")
    with mlflow.start_run(run_name="Hyperopt_RF_Tuning_Parent_Run") as parent_run:
        mlflow.set_tag("Model Type", "RandomForestClassifier")
        mlflow.set_tag("Tuning Method", "Hyperopt with TPE")
        print(f"Parent MLflow Run ID: {parent_run.info.run_id}")

        # --- Define Hyperparameter Search Space for Random Forest ---
        space = {
            'n_estimators': hp.quniform('n_estimators', 50, 500, 25),
            'max_depth': hp.quniform('max_depth', 3, 20, 1),
            'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
            'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
            'max_features': hp.choice('max_features', ['sqrt', 'log2', None])
        }

        print(f"Starting hyperparameter tuning with a maximum of {max_evals} evaluations...")
        trials = Trials()
        fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials
        )

        # --- Post-Tuning Analysis ---
        best_trial = trials.best_trial
        best_params_raw = best_trial['misc']['vals']
        
        # Clean up the best parameters for logging
        best_params_cleaned = {key: val[0] for key, val in best_params_raw.items()}
        # Map choice indices back to their string values
        max_features_options = ['sqrt', 'log2', None]
        best_params_cleaned['max_features'] = max_features_options[int(best_params_cleaned['max_features'])]
        
        best_auc = best_trial['result']['auc']
        
        print("\n--- Hyperparameter Tuning Complete ---")
        print(f"Best Test AUC: {best_auc:.4f}")
        print("Best parameters found:")
        print(best_params_cleaned)
        
        # Log the final results to the parent run
        mlflow.log_metric("best_test_auc_from_tuning", best_auc)
        mlflow.log_params(best_params_cleaned)
        mlflow.set_tag("Best Trial ID", best_trial['tid'])
        
        print("\nTuning process finished. Check the 'Churn Prediction RF Tuning' experiment in the MLflow UI.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for Churn Prediction model using RandomForest, Hyperopt, and MLflow.')
    parser.add_argument('--max_evals', type=int, default=50, help='Number of hyperparameter tuning iterations.')
    parser.add_argument('--data_path', type=str, default='membangun_model_sml/telco_preprocessed.csv', help='Path to the preprocessed data file.')
    args = parser.parse_args()
    main(args.max_evals, args.data_path)