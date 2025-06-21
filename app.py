
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("credit_model.pkl")

app = FastAPI()

class CreditInput(BaseModel):
    person_age: int
    person_income: float
    person_emp_length: float
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_cred_hist_length: int
    person_home_ownership_OTHER: int
    person_home_ownership_OWN: int
    person_home_ownership_RENT: int
    loan_intent_EDUCATION: int
    loan_intent_HOMEIMPROVEMENT: int
    loan_intent_MEDICAL: int
    loan_intent_PERSONAL: int
    loan_intent_VENTURE: int
    loan_grade_B: int
    loan_grade_C: int
    loan_grade_D: int
    loan_grade_E: int
    loan_grade_F: int
    loan_grade_G: int
    cb_person_default_on_file_Y: int

@app.post("/predict")
def predict_credit(data: CreditInput):
    try:
        input_data = np.array([[
            data.person_age,
            data.person_income,
            data.person_emp_length,
            data.loan_amnt,
            data.loan_int_rate,
            data.loan_percent_income,
            data.cb_person_cred_hist_length,
            data.person_home_ownership_OTHER,
            data.person_home_ownership_OWN,
            data.person_home_ownership_RENT,
            data.loan_intent_EDUCATION,
            data.loan_intent_HOMEIMPROVEMENT,
            data.loan_intent_MEDICAL,
            data.loan_intent_PERSONAL,
            data.loan_intent_VENTURE,
            data.loan_grade_B,
            data.loan_grade_C,
            data.loan_grade_D,
            data.loan_grade_E,
            data.loan_grade_F,
            data.loan_grade_G,
            data.cb_person_default_on_file_Y
        ]])
        prediction = model.predict(input_data)
        result = "Default risk" if prediction[0] == 1 else "Low risk"
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
