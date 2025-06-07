# model.py

import pandas as pd
from xgboost import XGBClassifier
import joblib

# Список категориальных признаков
cat_cols = ['P_Gender', 'P_Education', 'P_Home', 'L_Intent', 'L_Defaults']

def load_data():
    """Загружает данные из CSV"""
    df = pd.read_csv('L_Score.csv')
    return df

def preprocess_data(df):
    """Предобрабатывает данные: кодирует категориальные признаки"""
    df_encoded = pd.get_dummies(df, columns=cat_cols)
    X = df_encoded.drop('L_Status', axis=1)
    y = df_encoded['L_Status']
    return X, y

def train_final_model(X, y):
    """Обучает финальную модель на всех данных"""
    model = XGBClassifier(eval_metric='logloss', random_state=42)
    model.fit(X, y)
    return model

def save_model(model):
    """Сохраняет модель в файл .pkl"""
    joblib.dump(model, 'credit_scoring_model.pkl')
    print("✅ Модель сохранена как credit_scoring_model.pkl")

if __name__ == "__main__":
    df = load_data()
    X, y = preprocess_data(df)
    model = train_final_model(X, y)
    save_model(model)