import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

def train_production_pipeline(data_path="NexPA_Dynamic_Live_Data.csv"):
    print(f"⏳ Step 2: Extracting data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 1. Advanced Core Feature Engineering
    df['EMI_Proxy'] = (df['LoanAmount'] * (df['InterestRate'] / 100) / 12)
    df['FOIR'] = df['EMI_Proxy'] / (df['Income'] / 12) # Fixed Obligation to Income Ratio
    
    # 2. Handling Categorical Strings cleanly via One-Hot Encoding
    categorical_cols = ["Education", "EmploymentType", "MaritalStatus", "LoanPurpose"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    binary_map = {"Yes": 1, "No": 0}
    for col in ["HasMortgage", "HasDependents", "HasCoSigner"]:
        df[col] = df[col].map(binary_map)

    # 3. Isolating Features (Exclude DPD and Default to prevent structural data leakage)
    X = df.drop(columns=["Default", "DPD"])
    y = df["Default"]
    
    # Cache structural feature format for online schema validation mapping
    joblib.dump(X.columns.tolist(), "columns.pkl")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Fit and Store standard data scalers
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "scaler.pkl")
    
    # 5. Execute Regularized Gradient Boosting
    print("🚀 Training Regularized Gradient Boosting Pipeline Core...")
    model = GradientBoostingClassifier(
        n_estimators=120, 
        learning_rate=0.08, 
        max_depth=5, 
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # 6. Performance Evaluation Check
    probs = model.predict_proba(X_test_scaled)[:, 1]
    print(f"\n✨ System Verified. ROC-AUC Stable Performance: {roc_auc_score(y_test, probs):.4f}")
    
    # Save optimized model artifact
    joblib.dump(model, "rf_production_model.pkl")
    print("✅ Step 2 Complete: Operational inference artifacts stored successfully.")

if __name__ == "__main__":
    train_production_pipeline()