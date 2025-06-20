# MLProject/modelling.py (Versi Perbaikan)
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score # Dipertahankan untuk print ke konsol
import mlflow
import mlflow.sklearn
import os
import argparse
import sys

LE_CLASSES = ['Extrovert', 'Introvert'] 

def run_modelling(train_x_path, test_x_path, train_y_path, test_y_path, lr_c, lr_penalty, random_state):
    """
    Fungsi memulai modelling
    """
    
    mlflow.sklearn.autolog(
        log_model_signatures=True, 
        log_input_examples=True,
        registered_model_name="PersonalityClassifier_LR_Final"
    )
    
    with mlflow.start_run(run_name=f"CI_Automated_LR_Deploy_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"):
        
        run_id = mlflow.active_run().info.run_id
        print(f"MLflow Run ID: {run_id}")
        with open("run_id.txt", "w") as f:
            f.write(run_id)
            
        mlflow.set_experiment("Automated_MLflow_Project_Run")
        print("MLflow Autolog for Scikit-learn diaktifkan.\n")

        try:
            print("Memuat data pelatihan dan pengujian...")
            X_train = pd.read_csv(train_x_path) 
            X_test = pd.read_csv(test_x_path)   
            y_train = pd.read_csv(train_y_path) 
            y_test = pd.read_csv(test_y_path)   

            X_full = pd.concat([X_train, X_test], ignore_index=True)
            y_full = pd.concat([y_train, y_test], ignore_index=True)
            print("Data berhasil dimuat.")

        except FileNotFoundError as e:
            print(f"Error: Salah satu file data tidak ditemukan. Pastikan path benar. Error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error saat memuat atau memproses data: {e}", file=sys.stderr)
            sys.exit(1)

        print(f"Melatih Logistic Regression dengan C={lr_c}, penalty={lr_penalty} pada data penuh...")
        model = LogisticRegression(C=lr_c, penalty=lr_penalty, random_state=random_state, solver='liblinear')
        
        model.fit(X_full, y_full)
        print("Model Logistic Regression berhasil dilatih pada data penuh.")

        print("Mengevaluasi model pada data test...")
        y_pred_test = model.predict(X_test)
        
        accuracy_test = accuracy_score(y_test, y_pred_test)
        f1_test = f1_score(y_test, y_pred_test, average='binary')
        print(f"  -> Akurasi pada data test (di konsol): {accuracy_test:.4f}")
        print(f"  -> F1-Score pada data test (di konsol): {f1_test:.4f}")
        print("\nSemua metrik, parameter, dan artefak (model, confusion matrix) dicatat secara otomatis oleh MLflow Autolog.")

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