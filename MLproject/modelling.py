# MLProject/modelling.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


LE_CLASSES = ['Extrovert', 'Introvert'] 

def run_modelling(train_x_path, test_x_path, train_y_path, test_y_path, lr_c, lr_penalty, random_state):
    with mlflow.start_run(run_name=f"CI_Automated_LR_Deploy_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"):
        
        run_id = mlflow.active_run().info.run_id
        print(f"MLflow Run ID: {run_id}")
        with open("run_id.txt", "w") as f:
            f.write(run_id)
            
        mlflow.set_experiment("Automated_MLflow_Project_Run")
        mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)
        print("MLflow Autolog for Scikit-learn diaktifkan.\n")

        # --- Load Data ---
        try:
            print("Memuat data pelatihan dan pengujian...")
            X_train = pd.read_csv(train_x_path) 
            X_test = pd.read_csv(test_x_path)   
            y_train = pd.read_csv(train_y_path) 
            y_test = pd.read_csv(test_y_path)   

            print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

            # Gabungkan data training dan testing untuk melatih model final
            X_full = pd.concat([X_train, X_test], ignore_index=True)
            y_full = pd.concat([y_train, y_test], ignore_index=True)
            
            print(f"\nData gabungan (full) shape: X_full={X_full.shape}, y_full={y_full.shape}\n")

        except FileNotFoundError as e:
            print(f"Error: Salah satu file data tidak ditemukan. Pastikan path benar. Error: {e}")
            exit(1)
        except KeyError:
            print("Error: Pastikan nama kolom di file CSV/Parquet Anda sudah benar.")
            exit(1)
        except Exception as e:
            print(f"Error saat memuat atau memproses data: {e}")
            exit(1)

        print(f"Melatih Logistic Regression dengan C={lr_c}, penalty={lr_penalty} pada data penuh...")

        model = LogisticRegression(C=lr_c, penalty=lr_penalty, random_state=random_state, solver='liblinear')
        model.fit(X_full, y_full) # Latih pada data gabungan (full)
        print("Model Logistic Regression berhasil dilatih pada data penuh.")

        # Evaluasi pada data gabungan (ini lebih ke indikator kecocokan, bukan generalisasi)
        y_pred_full = model.predict(X_full)
        accuracy_full = accuracy_score(y_full, y_pred_full)
        f1_full = f1_score(y_full, y_pred_full, average='binary')
        mlflow.log_metric("full_data_accuracy", accuracy_full) # Metrik untuk data gabungan
        mlflow.log_metric("full_data_f1_score", f1_full)
        print(f"Akurasi pada data penuh: {accuracy_full:.4f}, F1-Score: {f1_full:.4f}")

        # Log Classification Report sebagai artefak
        report_str = classification_report(y_full, y_pred_full, target_names=LE_CLASSES)
        mlflow.log_text(report_str, "classification_report_full_data.txt")
        
        # Log Confusion Matrix sebagai gambar
        cm_full = confusion_matrix(y_full, y_pred_full)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_full, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=LE_CLASSES, yticklabels=LE_CLASSES)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - LR (Full Data)')
        plt.close()

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="logistic_regression_model",
            registered_model_name="PersonalityClassifier_LR_Final"
        )
        print("Model Logistic Regression berhasil diekspor ke MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Menjalankan training Logistic Regression sebagai MLflow Project.")
    
    parser.add_argument("--train_x", type=str, default="personality_preprocessing/X_train_personality_processing.csv", help="Path relatif ke X_train.")
    parser.add_argument("--test_x", type=str, default="personality_preprocessing/X_test_personality_processing.csv", help="Path relatif ke X_test.")
    parser.add_argument("--train_y", type=str, default="personality_preprocessing/y_train.csv", help="Path relatif ke y_train.")
    parser.add_argument("--test_y", type=str, default="personality_preprocessing/y_test.csv", help="Path relatif ke y_test.")
    
    parser.add_argument("--lr_c", type=float, default=0.1, help="Parameter C untuk Logistic Regression.")
    parser.add_argument("--lr_penalty", type=str, default="l2", choices=["l1", "l2"], help="Parameter penalty untuk Logistic Regression.")
    parser.add_argument("--random_state", type=int, default=42, help="Random state.")
    
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_x_abs_path = os.path.join(current_dir, args.train_x)
    test_x_abs_path = os.path.join(current_dir, args.test_x)
    train_y_abs_path = os.path.join(current_dir, args.train_y)
    test_y_abs_path = os.path.join(current_dir, args.test_y)

    run_modelling(
        train_x_path=train_x_abs_path,
        test_x_path=test_x_abs_path,
        train_y_path=train_y_abs_path,
        test_y_path=test_y_abs_path,
        lr_c=args.lr_c,
        lr_penalty=args.lr_penalty,
        random_state=args.random_state
    )