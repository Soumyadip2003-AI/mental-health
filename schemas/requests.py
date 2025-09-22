"""
Request schemas for the Mental Health Crisis Detection System
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class AnalysisRequest(BaseModel):
    """Request schema for single text analysis"""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")
    confidence_threshold: Optional[float] = Field(
        default=None, 
        ge=0.0, 
        le=1.0, 
        description="Confidence threshold for crisis detection"
    )
    include_explanation: bool = Field(default=True, description="Include explanation of the analysis")
    client_ip: Optional[str] = Field(default=None, description="Client IP address")
    user_agent: Optional[str] = Field(default=None, description="User agent string")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()
    
    @validator('confidence_threshold')
    def validate_confidence_threshold(cls, v):
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError('Confidence threshold must be between 0.0 and 1.0')
        return v


class BatchAnalysisRequest(BaseModel):
    """Request schema for batch text analysis"""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to analyze")
    confidence_threshold: Optional[float] = Field(
        default=None, 
        ge=0.0, 
        le=1.0, 
        description="Confidence threshold for crisis detection"
    )
    include_explanation: bool = Field(default=True, description="Include explanation of the analysis")
    client_ip: Optional[str] = Field(default=None, description="Client IP address")
    user_agent: Optional[str] = Field(default=None, description="User agent string")
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError('Texts list cannot be empty')
        
        for i, text in enumerate(v):
            if not text.strip():
                raise ValueError(f'Text at index {i} cannot be empty')
            if len(text) > 10000:
                raise ValueError(f'Text at index {i} is too long (max 10000 characters)')
        
        return [text.strip() for text in v]
    
    @validator('confidence_threshold')
    def validate_confidence_threshold(cls, v):
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError('Confidence threshold must be between 0.0 and 1.0')
        return v


class UserRegistrationRequest(BaseModel):
    """Request schema for user registration"""
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: str = Field(..., description="Email address")
    password: str = Field(..., min_length=8, description="Password")
    
    @validator('username')
    def validate_username(cls, v):
        if not v.isalnum():
            raise ValueError('Username must contain only alphanumeric characters')
        return v.lower()
    
    @validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v.lower()


class UserLoginRequest(BaseModel):
    """Request schema for user login"""
    username: str = Field(..., description="Username or email")
    password: str = Field(..., description="Password")


class ModelTrainingRequest(BaseModel):
    """Request schema for model training"""
    model_type: str = Field(..., description="Type of model to train")
    training_data: List[Dict[str, Any]] = Field(..., description="Training data")
    validation_data: Optional[List[Dict[str, Any]]] = Field(default=None, description="Validation data")
    hyperparameters: Optional[Dict[str, Any]] = Field(default=None, description="Model hyperparameters")
    
    @validator('model_type')
    def validate_model_type(cls, v):
        allowed_types = ['text', 'multimodal', 'ensemble', 'bert', 'roberta']
        if v not in allowed_types:
            raise ValueError(f'Model type must be one of: {allowed_types}')
        return v


class AlertConfigurationRequest(BaseModel):
    """Request schema for alert configuration"""
    risk_level: str = Field(..., description="Risk level to configure alerts for")
    enabled: bool = Field(default=True, description="Whether alerts are enabled")
    notification_methods: List[str] = Field(default=["email"], description="Notification methods")
    emergency_contacts: Optional[List[str]] = Field(default=None, description="Emergency contacts")
    
    @validator('risk_level')
    def validate_risk_level(cls, v):
        allowed_levels = ['low', 'medium', 'high', 'critical']
        if v not in allowed_levels:
            raise ValueError(f'Risk level must be one of: {allowed_levels}')
        return v
    
    @validator('notification_methods')
    def validate_notification_methods(cls, v):
        allowed_methods = ['email', 'sms', 'webhook', 'push']
        for method in v:
            if method not in allowed_methods:
                raise ValueError(f'Notification method must be one of: {allowed_methods}')
        return v


class SystemConfigurationRequest(BaseModel):
    """Request schema for system configuration"""
    confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_text_length: Optional[int] = Field(default=None, ge=1, le=50000)
    batch_size: Optional[int] = Field(default=None, ge=1, le=1000)
    enable_monitoring: Optional[bool] = Field(default=None)
    enable_alerts: Optional[bool] = Field(default=None)
    
    @validator('confidence_threshold')
    def validate_confidence_threshold(cls, v):
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError('Confidence threshold must be between 0.0 and 1.0')
        return v


class FeedbackRequest(BaseModel):
    """Request schema for user feedback"""
    analysis_id: int = Field(..., description="ID of the analysis to provide feedback for")
    feedback_type: str = Field(..., description="Type of feedback")
    rating: Optional[int] = Field(default=None, ge=1, le=5, description="Rating from 1 to 5")
    comment: Optional[str] = Field(default=None, max_length=1000, description="Additional comment")
    
    @validator('feedback_type')
    def validate_feedback_type(cls, v):
        allowed_types = ['accurate', 'inaccurate', 'helpful', 'not_helpful', 'false_positive', 'false_negative']
        if v not in allowed_types:
            raise ValueError(f'Feedback type must be one of: {allowed_types}')
        return v


class SearchRequest(BaseModel):
    """Request schema for searching analyses"""
    query: Optional[str] = Field(default=None, description="Search query")
    risk_level: Optional[str] = Field(default=None, description="Filter by risk level")
    date_from: Optional[datetime] = Field(default=None, description="Start date filter")
    date_to: Optional[datetime] = Field(default=None, description="End date filter")
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum number of results")
    offset: int = Field(default=0, ge=0, description="Number of results to skip")
    
    @validator('risk_level')
    def validate_risk_level(cls, v):
        if v is not None:
            allowed_levels = ['low', 'medium', 'high', 'critical']
            if v not in allowed_levels:
                raise ValueError(f'Risk level must be one of: {allowed_levels}')
        return v
