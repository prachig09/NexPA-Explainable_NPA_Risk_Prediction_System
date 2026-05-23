import pandas as pd
import numpy as np

def generate_indian_banking_data(filename="NexPA_Dynamic_Live_Data.csv", num_records=50000):
    print(f"⏳ Step 1: Simulating {num_records} Indian banking operational records...")
    np.random.seed(42)

    # 1. Base Demographic & Income Generation (INR)
    age = np.random.randint(22, 65, size=num_records)
    income = np.random.randint(300000, 3600000, size=num_records) 
    cibil_score = np.random.randint(300, 850, size=num_records)
    months_employed = np.random.randint(6, 240, size=num_records)
    num_credit_lines = np.random.randint(1, 10, size=num_records)
    
    # 2. Credit Product Mapping
    loan_purpose = np.random.choice(
        ["Home", "Auto", "Business", "Education", "Personal"], 
        size=num_records, 
        p=[0.3, 0.2, 0.2, 0.1, 0.2]
    )
    
    # Enforce realistic multi-multiplier scaling for loan sizes relative to annual earnings
    loan_amount = []
    for inc, purp in zip(income, loan_purpose):
        if purp == "Home": loan_amount.append(inc * np.random.uniform(3.0, 5.5))
        elif purp == "Business": loan_amount.append(inc * np.random.uniform(1.5, 3.5))
        else: loan_amount.append(inc * np.random.uniform(0.3, 1.2))
    loan_amount = np.array(loan_amount).astype(int)

    interest_rate = np.random.uniform(8.5, 18.0, size=num_records) 
    loan_term = np.random.choice([12, 36, 60, 120, 240], size=num_records) 
    dti_ratio = np.random.uniform(0.15, 0.75, size=num_records)

    # 3. Qualitative Operations Metrics
    education = np.random.choice(["Graduate", "PostGraduate", "Undergraduate", "Professional"], size=num_records)
    employment_type = np.random.choice(["Salaried", "Self-Employed", "Unemployed"], size=num_records, p=[0.7, 0.25, 0.05])
    marital_status = np.random.choice(["Single", "Married", "Divorced"], size=num_records)
    has_mortgage = np.random.choice(["Yes", "No"], size=num_records, p=[0.4, 0.6])
    has_dependents = np.random.choice(["Yes", "No"], size=num_records)
    has_cosigner = np.random.choice(["Yes", "No"], size=num_records, p=[0.3, 0.7])

    # 4. Inject Behavioral Stress Risk & Operational Days Past Due (DPD) 
    dpd = []
    for i in range(num_records):
        stress_factor = 0.0
        if cibil_score[i] < 600: stress_factor += 0.4
        if dti_ratio[i] > 0.55: stress_factor += 0.3
        if employment_type[i] == "Unemployed": stress_factor += 0.3
        if age[i] < 26: stress_factor += 0.1
        
        rand_draw = np.random.uniform(0, 1)
        if stress_factor > 0.7 or (stress_factor > 0.4 and rand_draw > 0.5):
            dpd.append(np.random.choice([61, 75, 91, 120, 180])) # SMA-2 / Hard NPA Range
        elif stress_factor > 0.3 or rand_draw > 0.85:
            dpd.append(np.random.choice([5, 15, 32, 45]))       # SMA-0 / SMA-1 Range
        else:
            dpd.append(0)                                       # Clean Standard Asset
            
    dpd = np.array(dpd)

    # 5. Define Ground Truth Target (Default = DPD > 30 Days)
    default_target = np.where(dpd > 30, 1, 0)

    # Compile structured dataframe
    df = pd.DataFrame({
        "Age": age, "Income": income, "LoanAmount": loan_amount, "CreditScore": cibil_score,
        "MonthsEmployed": months_employed, "NumCreditLines": num_credit_lines, "InterestRate": interest_rate,
        "LoanTerm": loan_term, "DTIRatio": dti_ratio, "Education": education, "EmploymentType": employment_type,
        "MaritalStatus": marital_status, "HasMortgage": has_mortgage, "HasDependents": has_dependents,
        "LoanPurpose": loan_purpose, "HasCoSigner": has_cosigner, "DPD": dpd, "Default": default_target
    })

    df.to_csv(filename, index=False)
    print(f"✅ Step 1 Complete: Dataset saved as '{filename}'.\nTarget Split:\n{df['Default'].value_counts(normalize=True)}")

if __name__ == "__main__":
    generate_indian_banking_data()