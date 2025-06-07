# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Заголовок
st.set_page_config(page_title="🏦 Кредитный скоринг", layout="centered")
st.title("🏦 Система кредитного скоринга")
st.markdown("Введите данные клиента, чтобы получить прогноз одобрения кредита.")

# Загрузка модели
@st.cache_resource
def load_model():
    return joblib.load('credit_scoring_model.pkl')

model = load_model()

# Восстановление списка колонок из файла или на основе данных
try:
    model_columns = joblib.load('credit_scoring_columns.pkl')
except FileNotFoundError:
    # Если файл не найден, можно создать шаблон колонок (заменить на твои реальные колонки)
    model_columns = [
        'P_Age', 'P_Income', 'P_Emp_Exp', 'Credit_History',
        'L_Amount', 'L_Rate', 'L_Pers_Income',
        'P_Gender_female', 'P_Gender_male',
        'P_Education_Associate', 'P_Education_Bachelor', 'P_Education_Doctorate',
        'P_Education_High School', 'P_Education_Master',
        'P_Home_MORTGAGE', 'P_Home_OWN', 'P_Home_RENT',
        'L_Intent_DEBTCONSOLIDATION', 'L_Intent_EDUCATION', 'L_Intent_HOMEIMPROVEMENT',
        'L_Intent_MEDICAL', 'L_Intent_PERSONAL', 'L_Intent_VENTURE',
        'L_Defaults_No', 'L_Defaults_Yes'
    ]
    joblib.dump(model_columns, 'credit_scoring_columns.pkl')

# Форма ввода данных
with st.form(key='credit_form'):
    col1, col2 = st.columns(2)

    with col1:
        P_Age = st.slider("Возраст", 18, 70, 30)
        P_Income = st.number_input("Годовой доход", min_value=10000, max_value=1000000, value=50000)
        P_Emp_Exp = st.slider("Стаж работы (лет)", 0, 30, 5)
        Credit_History = st.slider("Длина кредитной истории (лет)", 0, 30, 5)
        L_Amount = st.number_input("Сумма кредита", min_value=1000, max_value=500000, value=10000)
        L_Rate = st.number_input("Процентная ставка (%)", min_value=5.0, max_value=20.0, value=10.0, step=0.1)

    with col2:
        P_Gender = st.selectbox("Пол", ["male", "female"])
        P_Education = st.selectbox("Уровень образования", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
        P_Home = st.selectbox("Тип жилья", ["RENT", "OWN", "MORTGAGE"])
        L_Intent = st.selectbox("Цель кредита", [
            "DEBTCONSOLIDATION", "MEDICAL", "VENTURE", "PERSONAL", "EDUCATION", "HOMEIMPROVEMENT"
        ])
        L_Defaults = st.selectbox("Были ли дефолты ранее", ["No", "Yes"])

    submit_button = st.form_submit_button(label='Проверить заявку')

if submit_button:
    # Вычисление доли кредита от дохода
    L_Pers_Income = round(L_Amount / P_Income, 2)

    # Создание DataFrame
    input_data = pd.DataFrame({
        "P_Age": [P_Age],
        "P_Gender": [P_Gender],
        "P_Education": [P_Education],
        "P_Income": [P_Income],
        "P_Emp_Exp": [P_Emp_Exp],
        "P_Home": [P_Home],
        "L_Amount": [L_Amount],
        "L_Intent": [L_Intent],
        "L_Rate": [L_Rate],
        "L_Pers_Income": [L_Pers_Income],
        "Credit_History": [Credit_History],
        "L_Defaults": [L_Defaults]
    })

    # One-Hot кодирование
    input_data = pd.get_dummies(input_data)

    # Выравнивание столбцов
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    # Предсказание
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Вывод результата
    st.markdown("### 🧾 Результат")

    if prediction == 1:
        st.success(f"✅ Кредит **одобрен**")
    else:
        st.error(f"❌ Кредит **отклонён**")

    st.info(f"📊 Вероятность одобрения: **{probability:.2f}**")