import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.keras
import argparse
import os

# Suppress TensorFlow logging for cleaner output
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

def run_training_logic(epochs, batch_size, learning_rate, n_hidden, n_units, dropout_rate):
    print("Logging parameters...")
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("n_hidden_layers", n_hidden)
    mlflow.log_param("n_units_per_layer", n_units)
    mlflow.log_param("dropout_rate", dropout_rate)

    # --- 2. Load and Prepare Data ---
    print("Loading and preparing data...")
    try:
        # Assuming the MLProject file structure places data correctly
        df = pd.read_csv('telco_preprocessed.csv')
    except FileNotFoundError:
        print("Error: 'telco_preprocessed.csv' not found.")
        return

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    mlflow.set_tag("scaler", "StandardScaler")
    mlflow.log_param("training_samples", len(X_train_scaled))
    mlflow.log_param("testing_samples", len(X_test_scaled))

    # --- 3. Build Model ---
    print("Building the model...")
    model = build_model(X_train_scaled.shape[1], n_hidden, n_units, dropout_rate, learning_rate)

    # Log model summary as a text artifact
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    summary_str = "\n".join(summary_list)
    mlflow.log_text(summary_str, "model_summary.txt")

    # --- 4. Train Model ---
    print("Training the model...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    mlflow_logger = MlflowMetricLogger()

    model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, mlflow_logger],
        verbose=2
    )
    mlflow.log_param("stopped_epoch", early_stopping.stopped_epoch if early_stopping.stopped_epoch > 0 else epochs)

    # --- 5. Evaluate and Log ---
    print("Evaluating the model and logging final metrics...")
    metrics = evaluate_model(model, X_test_scaled, y_test)
    mlflow.log_metrics(metrics)
    print("\nTest Metrics:", metrics)

    # --- 6. Log Model ---
    print("Logging the Keras model...")
    mlflow.keras.log_model(
        model, 
        "keras-model", 
        registered_model_name="deep-churn-predictor"
    )

def main(epochs, batch_size, learning_rate, n_hidden, n_units, dropout_rate):
    mlflow.set_experiment("Churn Prediction DL")

    # This logic prevents the MlflowException. It checks if a run is already
    # active. If not, it creates one. Otherwise, it uses the existing run.
    if mlflow.active_run() is None:
        # Case 1: Script is run directly (e.g., `python modelling.py`)
        print("No active MLflow run detected. Creating a new run.")
        with mlflow.start_run(run_name="DL_Manual_Run") as run:
            print(f"Started new run: {run.info.run_id}")
            run_training_logic(epochs, batch_size, learning_rate, n_hidden, n_units, dropout_rate)
    else:
        # Case 2: Script is run via `mlflow run` (e.g., in GitHub Actions)
        print("Detected an existing MLflow run. Using it.")
        run_training_logic(epochs, batch_size, learning_rate, n_hidden, n_units, dropout_rate)

    print("\nMLflow Run completed successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Deep Learning model for Churn Prediction with MLflow tracking.')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the Adam optimizer.')
    parser.add_argument('--n_hidden', type=int, default=2, help='Number of hidden layers.')
    parser.add_argument('--n_units', type=int, default=64, help='Number of units in each hidden layer.')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate after each hidden layer.')
    
    args = parser.parse_args()
    main(**vars(args))
