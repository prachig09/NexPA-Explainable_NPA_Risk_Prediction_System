# backend/database.py
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

DATABASE_URL = "sqlite:///./asset_quality_audit.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class EvaluationAuditLog(Base):
    """
    SQLAlchemy Database Schema tracking full Multi-Agent processing outputs.
    Acts as the bank's internal compliance ledger.
    """
    __tablename__ = "evaluation_audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Core Account Context
    account_dpd = Column(Integer)
    rbi_asset_classification = Column(String)
    color_profile = Column(String)
    ml_default_probability = Column(Float)
    regulatory_provisioning = Column(String)
    actionable_measure = Column(String)
    primary_risk_driver = Column(String)

# Create tables instantly if they do not exist
Base.metadata.create_all(bind=engine)

def get_db():
    """Dependency injection tool to manage database connections cleanly."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()