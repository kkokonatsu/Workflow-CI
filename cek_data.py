# cek_data.py
import pandas as pd
import json

file_path = 'MLProject/personality_preprocessing/X_train_personality_processing.csv'

try:
    df = pd.read_csv(file_path)

    column_names = df.columns.tolist()

    sample_data_row = df.iloc[0].tolist()

    print("--- NAMA KOLOM (COPY SEMUA DI BAWAH INI) ---")
    print(json.dumps(column_names))
    print("\n" + "="*50 + "\n")

    print("--- CONTOH DATA BARIS PERTAMA (COPY SEMUA DI BAWAH INI) ---")
    print(json.dumps(sample_data_row))
    print("\n" + "="*50 + "\n")

    print("Silakan copy-paste kedua output di atas (termasuk tanda kurung []) ke dalam chat.")

except FileNotFoundError:
    print(f"Error: File tidak ditemukan di '{file_path}'.")
    print("Pastikan Anda menjalankan skrip ini dari direktori root 'Workflow-CI/' Anda.")
except Exception as e:
    print(f"Terjadi error: {e}")