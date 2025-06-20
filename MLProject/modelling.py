import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import mlflow
import mlflow.keras
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class MlflowMetricLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            mlflow.log_metric("epoch_loss", logs.get('loss'), step=epoch)
            mlflow.log_metric("epoch_accuracy", logs.get('accuracy'), step=epoch)
            mlflow.log_metric("epoch_val_loss", logs.get('val_loss'), step=epoch)
            mlflow.log_metric("epoch_val_accuracy", logs.get('val_accuracy'), step=epoch)

def build_model(input_dim, n_hidden, n_units, dropout_rate, learning_rate):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    for _ in range(n_hidden):
        model.add(Dense(n_units, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(model, X_test, y_test):
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype("int32")
    metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall": recall_score(y_test, y_pred),
        "test_f1_score": f1_score(y_test, y_pred),
        "test_auc": roc_auc_score(y_test, y_pred_proba)
    }
    return metrics

def main(epochs, batch_size, learning_rate, n_hidden, n_units, dropout_rate):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")   # Important for CI workflow
    mlflow.set_experiment("Churn Prediction DL")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Starting MLflow Run: {run_id}")

        # Log parameters
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("n_hidden_layers", n_hidden)
        mlflow.log_param("n_units_per_layer", n_units)
        mlflow.log_param("dropout_rate", dropout_rate)

        # Load data
        print("Loading data...")
        try:
            df = pd.read_csv('telco_preprocessed.csv')
        except FileNotFoundError:
            print("File 'telco_preprocessed.csv' not found!")
            return

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        mlflow.set_tag("scaler", "StandardScaler")
        mlflow.log_param("training_samples", len(X_train))
        mlflow.log_param("testing_samples", len(X_test))

        print("Building model...")
        model = build_model(X_train.shape[1], n_hidden, n_units, dropout_rate, learning_rate)

        # Log model summary
        summary_list = []
        model.summary(print_fn=lambda x: summary_list.append(x))
        summary_str = "\n".join(summary_list)
        mlflow.log_text(summary_str, "model_summary.txt")

        print("Training model...")
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        mlflow_logger = MlflowMetricLogger()

        model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, mlflow_logger],
            verbose=2
        )
        mlflow.log_param("stopped_epoch", early_stopping.stopped_epoch if early_stopping.stopped_epoch > 0 else epochs)

        print("Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        # Save artifacts
        print("Saving artifacts...")
        os.makedirs("saved_model", exist_ok=True)
        model_h5_path = "saved_model/sequential_model.h5"
        model.save(model_h5_path)
        scaler_path = "saved_model/scaler.pkl"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(model_h5_path, artifact_path="model")
        mlflow.log_artifact(scaler_path, artifact_path="scaler")

        # Log model directly to MLflow
        mlflow.keras.log_model(
            model,
            "keras-model",
            registered_model_name="deep-churn-predictor"
        )

        print("\nRun completed successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DL model with MLflow CI-style logging.')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--n_hidden', type=int, default=2)
    parser.add_argument('--n_units', type=int, default=64)
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    args = parser.parse_args()
    main(**vars(args))
