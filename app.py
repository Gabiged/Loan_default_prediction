import joblib
from flask import Flask, request, render_template
import lightgbm
import sklearn
import xgboost
import pandas as pd
import numpy as np
import myfunctions

app = Flask(__name__, template_folder='templates')
feature_names = ['CREDIT_TERM', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'AMT_ANNUITY', 'POS_ACTIVE_CNT_INSTALMENT_FUTURE_MEAN',
                 'AMT_CREDIT', 'EMPLOY_YEAR', 'PREV_CNT_PAYMENT_MEAN', 'INS_AMT_PAYMENT_SUM', 'AMT_GOODS_PRICE',
                 'CB_ACTIVE_DAYS_CREDIT_MAX', 'INS_NEW_PAYMENT_DIFF', 'CB_ACTIVE_DAYS_CREDIT_ENDDATE_MAX',
                 'DAYS_BIRTH', 'INS_NEW_PAYMENT_PERCENT', 'CB_ACTIVE_AMT_CREDIT_SUM_MEAN', 'TOTALAREA_MODE',
                 'PREV_AMT_DOWN_PAYMENT_MEAN', 'INS_NUM_INSTALMENT_NUMBER_MAX', 'PREV_NAME_CONTRACT_STATUS_Refused_SUM',
                 'CODE_GENDER_F', 'PREV_AMT_ANNUITY_MEAN', 'DAYS_ID_PUBLISH', 'CB_CLOSED_AMT_CREDIT_SUM_SUM',
                 'CB_ACTIVE_DAYS_CREDIT_ENDDATE_MIN', 'AMT_INCOME_TOTAL', 'CC_ACTIVE_AMT_DRAWINGS_ATM_CURRENT_MEAN',
                 'AGE_YEAR', 'DAYS_LAST_PHONE_CHANGE', 'PREV_NAME_YIELD_GROUP_high_SUM', 'PREV_SELLERPLACE_AREA_MEAN',
                 'INS_DAYS_ENTRY_PAYMENT_MAX', 'CB_CLOSED_AMT_CREDIT_SUM_MEAN',
                 'POS_ACTIVE_NAME_CONTRACT_STATUS_Active_SUM', 'ANNUAL_INCOME_PER_FAM']


@app.route('/', methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    """For rendering results on HTML GUI"""
    model = joblib.load("classifier_model.joblib")
    if request.method == "POST":
        input_values = request.form.get("input_values")
        int_features = [float(x) for x in input_values.split(',')]
        data = {feature: value for feature, value in zip(feature_names, int_features)}
        df = pd.DataFrame([data])
        probabilities = model.predict_proba(df)
        prediction_percent = probabilities[0][1]
        threshold = 0.5
        binary_prediction = 1 if prediction_percent >= threshold else 0
        if binary_prediction == 1:
            result = f"Client is likely to default {int(prediction_percent * 100)}%."
        else:
            result = f"Client is not likely to default {int((1 - prediction_percent) * 100)}%."
    else:
        result = f"No data found."

    return render_template('index.html', prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)

