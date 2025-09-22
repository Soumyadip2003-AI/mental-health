"""
Database models for the Mental Health Crisis Detection System
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy import create_engine
from datetime import datetime
from typing import Optional, Dict, Any
import json

Base = declarative_base()


class User(Base):
    """User model for authentication and tracking"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    
    # Relationships
    analyses = relationship("Analysis", back_populates="user")
    alerts = relationship("Alert", back_populates="user")


class Analysis(Base):
    """Analysis model for storing text analysis results"""
    __tablename__ = "analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    text_content = Column(Text, nullable=False)
    text_length = Column(Integer)
    
    # Analysis results
    crisis_probability = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)  # low, medium, high, critical
    confidence_score = Column(Float, nullable=False)
    
    # Model information
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(20))
    processing_time = Column(Float)  # in seconds
    
    # Features and explanations
    key_features = Column(JSON)  # Important words/phrases
    explanation = Column(JSON)  # LIME/SHAP explanations
    feature_importance = Column(JSON)  # Feature importance scores
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    
    # Relationships
    user = relationship("User", back_populates="analyses")
    alerts = relationship("Alert", back_populates="analysis")


class Alert(Base):
    """Alert model for crisis notifications"""
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id"), nullable=False)
    
    # Alert details
    alert_type = Column(String(50), nullable=False)  # crisis, risk, warning
    severity = Column(String(20), nullable=False)  # low, medium, high, critical
    message = Column(Text, nullable=False)
    
    # Status tracking
    status = Column(String(20), default="pending")  # pending, sent, acknowledged, resolved
    acknowledged_at = Column(DateTime)
    resolved_at = Column(DateTime)
    
    # Response information
    response_actions = Column(JSON)  # Actions taken
    emergency_contacts_notified = Column(JSON)  # Who was contacted
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="alerts")
    analysis = relationship("Analysis", back_populates="alerts")


class ModelPerformance(Base):
    """Model performance tracking"""
    __tablename__ = "model_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(20), nullable=False)
    
    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    auc_roc = Column(Float)
    
    # Dataset information
    dataset_size = Column(Integer)
    training_time = Column(Float)  # in seconds
    evaluation_time = Column(Float)  # in seconds
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    evaluation_date = Column(DateTime, default=datetime.utcnow)


class SystemMetrics(Base):
    """System performance and usage metrics"""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Usage metrics
    total_analyses = Column(Integer, default=0)
    crisis_detections = Column(Integer, default=0)
    false_positives = Column(Integer, default=0)
    false_negatives = Column(Integer, default=0)
    
    # Performance metrics
    avg_processing_time = Column(Float)
    max_processing_time = Column(Float)
    min_processing_time = Column(Float)
    
    # System health
    cpu_usage = Column(Float)
    memory_usage = Column(Float)
    disk_usage = Column(Float)
    
    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    """Audit log for security and compliance"""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # User information
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    username = Column(String(50))
    
    # Action details
    action = Column(String(100), nullable=False)  # login, analysis, alert, etc.
    resource = Column(String(100))  # what was accessed
    result = Column(String(20))  # success, failure, error
    
    # Request details
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    request_data = Column(JSON)
    
    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow)


# Database connection and session management
class DatabaseManager:
    """Database connection and session management"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def close_session(self, session):
        """Close database session"""
        session.close()


# Database utility functions
def get_db_manager(database_url: str) -> DatabaseManager:
    """Get database manager instance"""
    return DatabaseManager(database_url)


def create_analysis_record(
    session,
    text_content: str,
    crisis_probability: float,
    risk_level: str,
    confidence_score: float,
    model_name: str,
    user_id: Optional[int] = None,
    key_features: Optional[Dict[str, Any]] = None,
    explanation: Optional[Dict[str, Any]] = None,
    processing_time: Optional[float] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None
) -> Analysis:
    """Create a new analysis record"""
    
    analysis = Analysis(
        user_id=user_id,
        text_content=text_content,
        text_length=len(text_content),
        crisis_probability=crisis_probability,
        risk_level=risk_level,
        confidence_score=confidence_score,
        model_name=model_name,
        key_features=key_features,
        explanation=explanation,
        processing_time=processing_time,
        ip_address=ip_address,
        user_agent=user_agent
    )
    
    session.add(analysis)
    session.commit()
    session.refresh(analysis)
    
    return analysis


def create_alert_record(
    session,
    analysis_id: int,
    alert_type: str,
    severity: str,
    message: str,
    user_id: Optional[int] = None
) -> Alert:
    """Create a new alert record"""
    
    alert = Alert(
        user_id=user_id,
        analysis_id=analysis_id,
        alert_type=alert_type,
        severity=severity,
        message=message
    )
    
    session.add(alert)
    session.commit()
    session.refresh(alert)
    
    return alert


def get_user_analyses(session, user_id: int, limit: int = 100):
    """Get analyses for a specific user"""
    return session.query(Analysis).filter(
        Analysis.user_id == user_id
    ).order_by(Analysis.created_at.desc()).limit(limit).all()


def get_crisis_analyses(session, limit: int = 100):
    """Get recent crisis analyses"""
    return session.query(Analysis).filter(
        Analysis.risk_level.in_(["high", "critical"])
    ).order_by(Analysis.created_at.desc()).limit(limit).all()


def get_system_stats(session):
    """Get system statistics"""
    total_analyses = session.query(Analysis).count()
    crisis_analyses = session.query(Analysis).filter(
        Analysis.risk_level.in_(["high", "critical"])
    ).count()
    
    return {
        "total_analyses": total_analyses,
        "crisis_analyses": crisis_analyses,
        "crisis_rate": crisis_analyses / total_analyses if total_analyses > 0 else 0
    }
