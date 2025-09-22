"""
Response schemas for the Mental Health Crisis Detection System
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class AnalysisResponse(BaseModel):
    """Response schema for text analysis"""
    analysis_id: int = Field(..., description="Unique identifier for the analysis")
    crisis_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of crisis (0-1)")
    risk_level: str = Field(..., description="Risk level: low, medium, high, critical")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in the prediction")
    key_features: Optional[Dict[str, Any]] = Field(default=None, description="Key features that influenced the decision")
    explanation: Optional[Dict[str, Any]] = Field(default=None, description="Explanation of the analysis")
    processing_time: float = Field(..., description="Time taken to process the analysis (seconds)")
    timestamp: str = Field(..., description="ISO timestamp of the analysis")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BatchAnalysisResponse(BaseModel):
    """Response schema for batch analysis"""
    analyses: List[AnalysisResponse] = Field(..., description="List of analysis results")
    total_processed: int = Field(..., description="Total number of texts processed")
    processing_time: float = Field(..., description="Total time taken for batch processing (seconds)")


class UserResponse(BaseModel):
    """Response schema for user information"""
    id: int = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: str = Field(..., description="Email address")
    is_active: bool = Field(..., description="Whether user account is active")
    is_admin: bool = Field(..., description="Whether user has admin privileges")
    created_at: str = Field(..., description="Account creation timestamp")
    last_login: Optional[str] = Field(default=None, description="Last login timestamp")


class AlertResponse(BaseModel):
    """Response schema for alerts"""
    id: int = Field(..., description="Alert ID")
    analysis_id: int = Field(..., description="Associated analysis ID")
    alert_type: str = Field(..., description="Type of alert")
    severity: str = Field(..., description="Alert severity level")
    message: str = Field(..., description="Alert message")
    status: str = Field(..., description="Alert status")
    created_at: str = Field(..., description="Alert creation timestamp")
    acknowledged_at: Optional[str] = Field(default=None, description="When alert was acknowledged")
    resolved_at: Optional[str] = Field(default=None, description="When alert was resolved")


class SystemStatsResponse(BaseModel):
    """Response schema for system statistics"""
    total_analyses: int = Field(..., description="Total number of analyses performed")
    crisis_analyses: int = Field(..., description="Number of crisis analyses detected")
    crisis_rate: float = Field(..., description="Rate of crisis detection (0-1)")
    avg_processing_time: float = Field(..., description="Average processing time in seconds")
    system_uptime: float = Field(..., description="System uptime in seconds")
    active_users: int = Field(..., description="Number of active users")
    pending_alerts: int = Field(..., description="Number of pending alerts")


class ModelPerformanceResponse(BaseModel):
    """Response schema for model performance metrics"""
    model_name: str = Field(..., description="Name of the model")
    model_version: str = Field(..., description="Version of the model")
    accuracy: float = Field(..., description="Model accuracy")
    precision: float = Field(..., description="Model precision")
    recall: float = Field(..., description="Model recall")
    f1_score: float = Field(..., description="Model F1 score")
    auc_roc: float = Field(..., description="ROC AUC score")
    dataset_size: int = Field(..., description="Size of the training dataset")
    training_time: float = Field(..., description="Training time in seconds")
    evaluation_time: float = Field(..., description="Evaluation time in seconds")
    created_at: str = Field(..., description="Model creation timestamp")


class HealthCheckResponse(BaseModel):
    """Response schema for health check"""
    status: str = Field(..., description="System status")
    timestamp: str = Field(..., description="Check timestamp")
    version: str = Field(..., description="System version")
    uptime: float = Field(..., description="System uptime in seconds")
    database: str = Field(..., description="Database status")
    models: str = Field(..., description="Models status")
    memory_usage: float = Field(..., description="Memory usage percentage")
    cpu_usage: float = Field(..., description="CPU usage percentage")


class ErrorResponse(BaseModel):
    """Response schema for errors"""
    detail: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")
    timestamp: str = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracking")


class SuccessResponse(BaseModel):
    """Response schema for successful operations"""
    message: str = Field(..., description="Success message")
    timestamp: str = Field(..., description="Operation timestamp")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Additional data")


class AuthenticationResponse(BaseModel):
    """Response schema for authentication"""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    user: UserResponse = Field(..., description="User information")


class SearchResponse(BaseModel):
    """Response schema for search results"""
    results: List[AnalysisResponse] = Field(..., description="Search results")
    total_count: int = Field(..., description="Total number of matching results")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of results per page")
    has_next: bool = Field(..., description="Whether there are more results")
    has_previous: bool = Field(..., description="Whether there are previous results")


class FeedbackResponse(BaseModel):
    """Response schema for feedback submission"""
    feedback_id: int = Field(..., description="Feedback ID")
    analysis_id: int = Field(..., description="Associated analysis ID")
    feedback_type: str = Field(..., description="Type of feedback")
    rating: Optional[int] = Field(default=None, description="Rating provided")
    comment: Optional[str] = Field(default=None, description="Comment provided")
    created_at: str = Field(..., description="Feedback creation timestamp")


class ConfigurationResponse(BaseModel):
    """Response schema for system configuration"""
    confidence_threshold: float = Field(..., description="Current confidence threshold")
    max_text_length: int = Field(..., description="Maximum text length allowed")
    batch_size: int = Field(..., description="Batch processing size")
    enable_monitoring: bool = Field(..., description="Whether monitoring is enabled")
    enable_alerts: bool = Field(..., description="Whether alerts are enabled")
    model_settings: Dict[str, Any] = Field(..., description="Model-specific settings")
    alert_settings: Dict[str, Any] = Field(..., description="Alert configuration")


class ModelInfoResponse(BaseModel):
    """Response schema for model information"""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    type: str = Field(..., description="Model type")
    status: str = Field(..., description="Model status")
    accuracy: float = Field(..., description="Model accuracy")
    last_trained: str = Field(..., description="Last training timestamp")
    features: List[str] = Field(..., description="Model features")
    hyperparameters: Dict[str, Any] = Field(..., description="Model hyperparameters")


class CrisisSummaryResponse(BaseModel):
    """Response schema for crisis summary"""
    total_crises: int = Field(..., description="Total number of crises detected")
    high_risk_crises: int = Field(..., description="Number of high-risk crises")
    critical_crises: int = Field(..., description="Number of critical crises")
    recent_crises: List[AnalysisResponse] = Field(..., description="Recent crisis analyses")
    trends: Dict[str, Any] = Field(..., description="Crisis trends and patterns")
    recommendations: List[str] = Field(..., description="Recommendations based on analysis")
