from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib

app = FastAPI()

# ===== CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====  MODELS =====
rf_model = joblib.load("rf_model.pkl")
lr_model = joblib.load("lr_model.pkl")
columns = joblib.load("columns.pkl")

# ===== HELPER FUNCTION =====
def predict_row(data: dict):
    import pandas as pd

    input_data = pd.DataFrame(columns=columns)
    input_data.loc[0] = 0

    # Numerical
    for col in ["Age","Income","LoanAmount","CreditScore","MonthsEmployed",
                "NumCreditLines","InterestRate","LoanTerm","DTIRatio"]:
        input_data[col] = data.get(col, 0)

    # Categorical
    for cat_col in ["Education","EmploymentType","MaritalStatus","LoanPurpose"]:
        col_name = f"{cat_col}_{data.get(cat_col,'')}"
        if col_name in columns:
            input_data[col_name] = 1

    for yes_col in ["HasMortgage","HasDependents","HasCoSigner"]:
        if data.get(yes_col,"No") == "Yes":
            input_data[f"{yes_col}_Yes"] = 1

    # Probability
    prob = lr_model.predict_proba(input_data)[0][1]

    # Adjust for demo
    if data.get("CreditScore",0) < 500:
        prob += 0.15
    prob = min(prob, 1.0)

    # Risk level
    if prob < 0.3:
        risk = "NO RISK 🟢"
        decision = "Loan Approved ✅"
    elif prob < 0.6:
        risk = "MEDIUM RISK 🟠"
        decision = "Further Review ⚠️"
    else:
        risk = "HIGH RISK 🔴"
        decision = "Loan Rejected ❌"

    # Active factors
    key_factors = {
        "low_credit_score": data.get("CreditScore",0) < 500,
        "high_dti": data.get("DTIRatio",0) > 0.6,
        "low_income": data.get("Income",0) < 30000,
        "high_loan_amount": data.get("LoanAmount",0) > 70000,
        "high_interest_rate": data.get("InterestRate",0) > 12,
        "many_credit_lines": data.get("NumCreditLines",0) > 5,
        "unstable_employment": data.get("EmploymentType","") != "Full-time"
    }
    active = [k for k,v in key_factors.items() if v]

    return {
        "risk_level": risk,
        "decision": decision,
        "probability_of_default": round(prob,3),
        "active_risk_factors": active
    }

# ===== SINGLE PREDICT =====
from pydantic import BaseModel

class PredictRequest(BaseModel):
    Age: int
    Income: int
    LoanAmount: int
    CreditScore: int
    MonthsEmployed: int
    NumCreditLines: int
    InterestRate: float
    LoanTerm: int
    DTIRatio: float
    Education: str
    EmploymentType: str
    MaritalStatus: str
    HasMortgage: str
    HasDependents: str
    LoanPurpose: str
    HasCoSigner: str

@app.post("/predict")
def predict(data: PredictRequest):
    return predict_row(data.dict())

# ===== BATCH PREDICT =====
@app.post("/batch_predict")
async def batch_predict(file: UploadFile = File(...)):
    # Only accept CSV
    if not file.filename.endswith(".csv"):
        return {"error": "Only CSV files supported"}

    # Read CSV into pandas
    df = pd.read_csv(file.file)

    # Process each row
    results = []
    for _, row in df.iterrows():
        results.append(predict_row(row.to_dict()))

    return {"results": results}