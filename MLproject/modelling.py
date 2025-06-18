# MLProject/modelling.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub

dagshub.init(repo_owner='kkokonatsu', repo_name='Workflow-CI', mlflow=True)

LE_CLASSES = ['Extrovert', 'Introvert']

def run_modelling_lr(
    train_x_path, train_y_path, test_x_path, test_y_path,
    lr_c=1.0, lr_penalty='l2', 
    random_state=42
):
    """
    Trains and logs a Logistic Regression model using MLflow.
    Args:
        train_x_path (str): Path to the processed training features.
        train_y_path (str): Path to the processed training labels.
        test_x_path (str): Path to the processed testing features.
        test_y_path (str): Path to the processed testing labels.
        lr_c (float): Inverse of regularization strength for Logistic Regression.
        lr_penalty (str): Regularization type for Logistic Regression.
        random_state (int): Random state for reproducibility.
    """
    # --- MLflow Configuration ---
    mlflow.set_experiment("Automated_CI_Logistic_Regression_Final_Model")
    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)
    print("MLflow Autolog for Scikit-learn activated.\n")

    # --- Load Data ---
    try:
        X_train = pd.read_csv(train_x_path)
        X_test = pd.read_csv(test_x_path)
        y_train = pd.read_csv(train_y_path)
        y_test = pd.read_csv(test_y_path)

        print("Data loaded successfully:")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}\n")

    except FileNotFoundError as e:
        print(f"Error: Data file not found. Please ensure paths are correct. Error: {e}")
        exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)
        
    # --- Train Logistic Regression Model ---
    print(f"Training Logistic Regression Model with C={lr_c}, penalty={lr_penalty}...")

    model = LogisticRegression(C=lr_c, penalty=lr_penalty, random_state=random_state, solver='liblinear')

    model.fit(X_train, y_train)
    print("Logistic Regression model trained successfully.")

    # --- Evaluate Model ---
    y_pred = model.predict(X_test)

    report_str = classification_report(y_test, y_pred, target_names=LE_CLASSES)
    mlflow.log_text(report_str, "classification_report.txt")
    print("\nClassification Report:\n", report_str)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=LE_CLASSES, yticklabels=LE_CLASSES)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - Logistic Regression (C={lr_c}, P={lr_penalty})')
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    # Export (Log) the model as an MLflow artifact
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="logistic_regression_model",
        registered_model_name="PersonalityClassifier_LogisticRegression_Final"
    )
    print("Logistic Regression model exported (logged) to MLflow.")

    print(f"--- Model Training and Logging for Logistic Regression Completed ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLflow Project for Logistic Regression Model Training.")
    
    parser.add_argument("--train_x", type=str, default="personality_preprocessing/X_train_personality_processing.csv",
                        help="Path to training features.")
    parser.add_argument("--train_y", type=str, default="personality_preprocessing/y_train.csv",
                        help="Path to training labels.")
    parser.add_argument("--test_x", type=str, default="personality_preprocessing/X_test_personality_processing.csv",
                        help="Path to testing features.")
    parser.add_argument("--test_y", type=str, default="personality_preprocessing/y_test.csv",
                        help="Path to testing labels.")
    
    parser.add_argument("--lr_c", type=float, default=0.1, 
                        help="Inverse of regularization strength for Logistic Regression (C).")
    parser.add_argument("--lr_penalty", type=str, default="l2", 
                        choices=["l1", "l2"],
                        help="Regularization type for Logistic Regression (penalty).")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random state for reproducibility.")
    
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_x_abs_path = os.path.join(current_dir, args.train_x)
    train_y_abs_path = os.path.join(current_dir, args.train_y)
    test_x_abs_path = os.path.join(current_dir, args.test_x)
    test_y_abs_path = os.path.join(current_dir, args.test_y)

    run_modelling_lr(
        train_x_path=train_x_abs_path,
        train_y_path=train_y_abs_path,
        test_x_path=test_x_abs_path,
        test_y_path=test_y_abs_path,
        lr_c=args.lr_c,
        lr_penalty=args.lr_penalty,
        random_state=args.random_state
    )