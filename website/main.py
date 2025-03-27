from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import io, base64
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer


app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = CatBoostClassifier()
model.load_model("catboostmodel.cbm")

young_age = 40
poor_income = 20000
revolvingUtilization_threshold = 1


def preprocess_df(df, young_age, poor_income, revolvingUtilization):
    revolvingInFraudRange = list(np.logical_and(df['RevolvingUtilizationOfUnsecuredLines'] < 2.7,
                                                 df['RevolvingUtilizationOfUnsecuredLines'] > 0.8))
    isYoung = list(df['age'] < young_age)
    isPoorIncome = list(df['MonthlyIncome'] < poor_income)
    doesHaveRevolving = list(df['RevolvingUtilizationOfUnsecuredLines'] > revolvingUtilization)
    
    df['revolvingInFraudRange'] = revolvingInFraudRange
    df['isYoung'] = isYoung
    df['isPoorIncome'] = isPoorIncome 
    df['doesHaveRevolving'] = doesHaveRevolving
    
    df['age'] = [np.log(age) if age != 0 else 0 for age in df['age']]
    df['MonthlyIncome'] = [np.log(income) if income != 0 else 0 for income in df['MonthlyIncome']]
    df['RevolvingUtilizationOfUnsecuredLines'] = [np.log(rev) if rev != 0 else 0 for rev in df['RevolvingUtilizationOfUnsecuredLines']]
    
    df['revolvingAndIncome'] = [
        df['RevolvingUtilizationOfUnsecuredLines'][i] ** 2 / df['MonthlyIncome'][i] if df['MonthlyIncome'][i] != 0 else 0 
        for i in range(len(df))
    ]
    df['squaredNumber'] = df['NumberOfTime30-59DaysPastDueNotWorse'] ** 2
    df['squaredRevolving'] = df['RevolvingUtilizationOfUnsecuredLines'] ** 2
    df['squaredDependents'] = df['NumberOfDependents'] ** 2
    df['daysLate'] = (
        df['NumberOfTime30-59DaysPastDueNotWorse'] +
        df['NumberOfTime60-89DaysPastDueNotWorse'] +
        df['NumberOfTimes90DaysLate']
    )
    
    df['isYoung'] = df['isYoung'].astype('int')
    df['isPoorIncome'] = df['isPoorIncome'].astype('int')
    df['doesHaveRevolving'] = df['doesHaveRevolving'].astype('int')
    
    return df

feature_names = [
    'RevolvingUtilizationOfUnsecuredLines', 'age',
    'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents', 'revolvingInFraudRange', 'isYoung',
    'isPoorIncome', 'doesHaveRevolving', 'revolvingAndIncome',
    'squaredNumber', 'squaredRevolving', 'squaredDependents', 'daysLate'
]

def generate_feature_importance_plot(model, feature_names):
    importances = model.get_feature_importance()
    plt.figure(figsize=(10,8))
    plt.bar(feature_names, importances)
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.title("Feature Importance")
    plt.xticks(rotation=90)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return encoded


@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "features": feature_names})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request,
    RevolvingUtilizationOfUnsecuredLines: float = Form(...),
    age: float = Form(...),
    NumberOfTime30_59DaysPastDueNotWorse: float = Form(...),
    DebtRatio: float = Form(...),
    MonthlyIncome: float = Form(...),
    NumberOfOpenCreditLinesAndLoans: float = Form(...),
    NumberOfTimes90DaysLate: float = Form(...),
    NumberRealEstateLoansOrLines: float = Form(...),
    NumberOfTime60_89DaysPastDueNotWorse: float = Form(...),
    NumberOfDependents: float = Form(...)):
    
    input_data = {
        "RevolvingUtilizationOfUnsecuredLines": RevolvingUtilizationOfUnsecuredLines,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": NumberOfTime30_59DaysPastDueNotWorse,
        "DebtRatio": DebtRatio,
        "MonthlyIncome": MonthlyIncome,
        "NumberOfOpenCreditLinesAndLoans": NumberOfOpenCreditLinesAndLoans,
        "NumberOfTimes90DaysLate": NumberOfTimes90DaysLate,
        "NumberRealEstateLoansOrLines": NumberRealEstateLoansOrLines,
        "NumberOfTime60-89DaysPastDueNotWorse": NumberOfTime60_89DaysPastDueNotWorse,
        "NumberOfDependents": NumberOfDependents
    }
    input_df = pd.DataFrame([input_data])
    
    input_df = preprocess_df(input_df, young_age, poor_income, revolvingUtilization_threshold)
    input_df = input_df[feature_names]
    
    prediction_proba = model.predict_proba(input_df)[:, 1][0]
    prediction = model.predict(input_df)[0]
    
    feature_importance_img = generate_feature_importance_plot(model, feature_names)
    
    return templates.TemplateResponse("result.html", {
        "request": request,
        "prediction": prediction,
        "prediction_proba": round(prediction_proba, 2),
        "feature_importance_img": feature_importance_img
    })
