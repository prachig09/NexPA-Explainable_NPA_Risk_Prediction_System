from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import io
from pydantic import BaseModel
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from backend.database import get_db, EvaluationAuditLog

app = FastAPI(title="NexPA - Multi-Agent Asset Quality Workspace")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== SYSTEM REGISTRY ASSETS =====
try:
    production_model = joblib.load("rf_production_model.pkl")
    production_scaler = joblib.load("scaler.pkl")
    production_columns = joblib.load("columns.pkl")
except Exception as e:
    print(f"⚠️ Core artifacts missing. Run generation and training scripts first: {e}")

# =====================================================================
#                          MULTI-AGENT CORES
# =====================================================================

class DataScoutAgent:
    """
    Agent 1: Specialized in financial data preparation and ratio engineering.
    Cleans up incoming chaos so the models get perfect features.
    """
    def __init__(self, target_columns: List[str]):
        self.columns = target_columns

    def execute(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        frame = pd.DataFrame(columns=self.columns)
        frame.loc[0] = 0
        
        # Parse numerical variables
        for col in ["Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed", 
                    "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio"]:
            frame[col] = float(raw_data.get(col, 0))
            
        # Advanced Multi-Variant Engineering
        frame['EMI_Proxy'] = (frame['LoanAmount'] * (frame['InterestRate'] / 100) / 12)
        frame['FOIR'] = frame['EMI_Proxy'] / (frame['Income'] / 12)
        
        # Categorical structural one-hot flags mapping
        for cat in ["Education", "EmploymentType", "MaritalStatus", "LoanPurpose"]:
            target_flag = f"{cat}_{raw_data.get(cat, '')}"
            if target_flag in frame.columns:
                frame[target_flag] = 1
                
        for binary in ["HasMortgage", "HasDependents", "HasCoSigner"]:
            if raw_data.get(binary) == "Yes":
                frame[binary] = 1
                
        return frame


class RiskOracleAgent:
    """
    Agent 2: Specialized in statistical risk forecasting.
    Maintains the predictive machine learning engine to see defaults before they happen.
    """
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def execute(self, prepared_frame: pd.DataFrame) -> float:
        scaled_features = self.scaler.transform(prepared_frame)
        risk_probability = self.model.predict_proba(scaled_features)[0][1]
        return float(risk_probability)


class LegalScribeAgent:
    """
    Agent 3: Specialized in regulatory enforcement and compliance rules.
    Maps real account statuses against RBI rules and Indian recovery law frameworks.
    """
    def extract_remediation(self, dpd: int, purpose: str, asset_class: str, tag: str, loan_amt: float, has_mortgage: str) -> Dict[str, str]:
        if tag == "GREEN":
            return {"action": "Maintain routine automated tracking. Push standard payment alerts.", "provision": "0.40%"}
        
        if tag == "AMBER":
            return {"action": "Transfer file to Early Resolution Desk. Execute soft calling outreach verification.", "provision": "1.00%"}
            
        if "SMA-2" in asset_class:
            action = "Extend formal restructuring window under specialized RBI MSME frameworks." if purpose == "Business" else "Issue pre-compromise legal settlement offer notice directly."
            return {"action": action, "provision": "5.00%"}
            
        # Hard Default Asset Recovery Matrix (NPA Zone)
        if purpose == "Home" or has_mortgage == "Yes":
            action = "Initiate statutory property asset seizure protocol under Section 13(2) of the SARFAESI Act, 2002."
        elif purpose == "Business" and loan_amt > 10000000:
            action = "File emergency summary corporate collection application via NCLT under the IBC framework."
        else:
            action = "Institute litigation recovery suit via nearest Debt Recovery Tribunal (DRT) / regional Lok Adalat."
            
        return {"action": action, "provision": "15.00%"}

    def execute(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        dpd = int(raw_data.get("DPD", 0))
        purpose = raw_data.get("LoanPurpose", "Other")
        loan_amount = float(raw_data.get("LoanAmount", 0))
        has_mortgage = raw_data.get("HasMortgage", "No")
        
        # RBI Timeline Classification Rules Engine
        if dpd == 0:
            asset_class, tag = "STANDARD (Performing) 🟢", "GREEN"
        elif dpd <= 30:
            asset_class, tag = "SMA-0 (Incipient Stress) 🟡", "AMBER"
        elif dpd <= 60:
            asset_class, tag = "SMA-1 (Friction Detected) 🟠", "AMBER"
        elif dpd <= 90:
            asset_class, tag = "SMA-2 (High Risk Overdue) 🔴", "RED"
        else:
            asset_class, tag = "NPA (Sub-Standard Asset) ❌", "CRITICAL_RED"
            
        remediation = self.extract_remediation(dpd, purpose, asset_class, tag, loan_amount, has_mortgage)
        
        return {
            "classification": asset_class,
            "color_tag": tag,
            "action": remediation["action"],
            "provision": remediation["provision"]
        }


class OrchestratorAgent:
    """
    The System Brain. Collects operational data, commands sub-agents, 
    assembles their intelligence, and outputs the uniform dashboard payload.
    """
    def __init__(self):
        self.scout = DataScoutAgent(production_columns)
        self.oracle = RiskOracleAgent(production_model, production_scaler)
        self.scribe = LegalScribeAgent()

    def process_lifecycle(self, client_payload: Dict[str, Any]) -> Dict[str, Any]:
        # Step A: Data Scout transforms raw input inputs into analytical frames
        clean_frame = self.scout.execute(client_payload)
        
        # Step B: Risk Oracle reads the engineered vectors to forecast probabilities
        ml_score = self.oracle.execute(clean_frame)
        
        # Step C: Legal Scribe cross-references parameters against financial law books
        legal_profile = self.scribe.execute(client_payload)
        
        # Step D: Consolidate intelligence outputs
        foir_percentage = round(float(clean_frame['FOIR'].iloc[0] * 100), 1)
        driver = f"High Commitment Load (FOIR: {foir_percentage}%)" if foir_percentage > 50 else "Stable Leverage Balance"
        
        return {
            "account_dpd": int(client_payload.get("DPD", 0)),
            "rbi_asset_classification": legal_profile["classification"],
            "color_profile": legal_profile["color_tag"],
            "ml_default_probability": round(ml_score, 3),
            "regulatory_provisioning": legal_profile["provision"],
            "actionable_measure": legal_profile["action"],
            "risk_drivers": [driver]
        }

# =====================================================================
#                          API ROUTER SETUP
# =====================================================================

system_orchestrator = OrchestratorAgent()

class AssetDataPayload(BaseModel):
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
    DPD: int = 0

@app.post("/predict")
def single_inference_endpoint(data: AssetDataPayload, db: Session = Depends(get_db)):
    # 1. Run our multi-agent pipeline logic
    result = system_orchestrator.process_lifecycle(data.dict())
    
    # 2. Persist the results into our audit ledger
    audit_record = EvaluationAuditLog(
        account_dpd=result["account_dpd"],
        rbi_asset_classification=result["rbi_asset_classification"],
        color_profile=result["color_profile"],
        ml_default_probability=result["ml_default_probability"],
        regulatory_provisioning=result["regulatory_provisioning"],
        actionable_measure=result["actionable_measure"],
        primary_risk_driver=result["risk_drivers"][0] if result["risk_drivers"] else "Unknown"
    )
    db.add(audit_record)
    db.commit()
    
    return result

@app.post("/batch_predict")
async def batch_inference_endpoint(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith(".csv"):
        return {"error": "Invalid format. Process requires clean CSV datasets."}
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    batch_results = []
    for _, row in df.iterrows():
        result = system_orchestrator.process_lifecycle(row.to_dict())
        batch_results.append(result)
        
        # Log each batch record to the database
        audit_record = EvaluationAuditLog(
            account_dpd=result["account_dpd"],
            rbi_asset_classification=result["rbi_asset_classification"],
            color_profile=result["color_profile"],
            ml_default_probability=result["ml_default_probability"],
            regulatory_provisioning=result["regulatory_provisioning"],
            actionable_measure=result["actionable_measure"],
            primary_risk_driver=result["risk_drivers"][0] if result["risk_drivers"] else "Unknown"
        )
        db.add(audit_record)
        
    db.commit() # Save everything in one transaction block
    return {"results": batch_results}

# New complementary endpoint to display audit trail inside your dashboard
@app.get("/audit_logs")
def fetch_historical_audit_logs(limit: int = 100, db: Session = Depends(get_db)):
    logs = db.query(EvaluationAuditLog).order_by(EvaluationAuditLog.id.desc()).limit(limit).all()
    return logs