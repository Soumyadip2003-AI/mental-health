"""
Custom exceptions for the Mental Health Crisis Detection System
"""
from typing import Optional, Dict, Any


class CrisisDetectionException(Exception):
    """Base exception for crisis detection system"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class ModelLoadException(CrisisDetectionException):
    """Exception raised when model loading fails"""
    
    def __init__(self, model_name: str, original_error: Optional[Exception] = None):
        message = f"Failed to load model: {model_name}"
        if original_error:
            message += f" - {str(original_error)}"
        
        super().__init__(
            message=message,
            error_code="MODEL_LOAD_ERROR",
            details={"model_name": model_name, "original_error": str(original_error) if original_error else None}
        )


class ModelPredictionException(CrisisDetectionException):
    """Exception raised when model prediction fails"""
    
    def __init__(self, model_name: str, input_data: Any, original_error: Optional[Exception] = None):
        message = f"Model prediction failed for model: {model_name}"
        if original_error:
            message += f" - {str(original_error)}"
        
        super().__init__(
            message=message,
            error_code="MODEL_PREDICTION_ERROR",
            details={
                "model_name": model_name,
                "input_data_type": type(input_data).__name__,
                "original_error": str(original_error) if original_error else None
            }
        )


class PreprocessingException(CrisisDetectionException):
    """Exception raised when text preprocessing fails"""
    
    def __init__(self, text: str, preprocessing_step: str, original_error: Optional[Exception] = None):
        message = f"Text preprocessing failed at step: {preprocessing_step}"
        if original_error:
            message += f" - {str(original_error)}"
        
        super().__init__(
            message=message,
            error_code="PREPROCESSING_ERROR",
            details={
                "text_length": len(text),
                "preprocessing_step": preprocessing_step,
                "original_error": str(original_error) if original_error else None
            }
        )


class FeatureExtractionException(CrisisDetectionException):
    """Exception raised when feature extraction fails"""
    
    def __init__(self, text: str, feature_type: str, original_error: Optional[Exception] = None):
        message = f"Feature extraction failed for feature type: {feature_type}"
        if original_error:
            message += f" - {str(original_error)}"
        
        super().__init__(
            message=message,
            error_code="FEATURE_EXTRACTION_ERROR",
            details={
                "text_length": len(text),
                "feature_type": feature_type,
                "original_error": str(original_error) if original_error else None
            }
        )


class DatabaseException(CrisisDetectionException):
    """Exception raised when database operations fail"""
    
    def __init__(self, operation: str, table: str, original_error: Optional[Exception] = None):
        message = f"Database operation failed: {operation} on table {table}"
        if original_error:
            message += f" - {str(original_error)}"
        
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            details={
                "operation": operation,
                "table": table,
                "original_error": str(original_error) if original_error else None
            }
        )


class AuthenticationException(CrisisDetectionException):
    """Exception raised when authentication fails"""
    
    def __init__(self, reason: str, user_id: Optional[str] = None):
        message = f"Authentication failed: {reason}"
        
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            details={"reason": reason, "user_id": user_id}
        )


class AuthorizationException(CrisisDetectionException):
    """Exception raised when authorization fails"""
    
    def __init__(self, resource: str, action: str, user_id: Optional[str] = None):
        message = f"Authorization failed: {action} on {resource}"
        
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            details={"resource": resource, "action": action, "user_id": user_id}
        )


class RateLimitExceededException(CrisisDetectionException):
    """Exception raised when rate limit is exceeded"""
    
    def __init__(self, client_id: str, limit: int, window: int, retry_after: int):
        message = f"Rate limit exceeded for client {client_id}: {limit} requests per {window} seconds"
        
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            details={
                "client_id": client_id,
                "limit": limit,
                "window": window,
                "retry_after": retry_after
            }
        )
        self.retry_after = retry_after


class ValidationException(CrisisDetectionException):
    """Exception raised when input validation fails"""
    
    def __init__(self, field: str, value: Any, constraint: str):
        message = f"Validation failed for field '{field}': {constraint}"
        
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details={
                "field": field,
                "value": str(value),
                "constraint": constraint
            }
        )


class ConfigurationException(CrisisDetectionException):
    """Exception raised when configuration is invalid"""
    
    def __init__(self, config_key: str, expected_type: str, actual_type: str):
        message = f"Configuration error for '{config_key}': expected {expected_type}, got {actual_type}"
        
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details={
                "config_key": config_key,
                "expected_type": expected_type,
                "actual_type": actual_type
            }
        )


class ExternalServiceException(CrisisDetectionException):
    """Exception raised when external service calls fail"""
    
    def __init__(self, service_name: str, operation: str, status_code: Optional[int] = None, original_error: Optional[Exception] = None):
        message = f"External service error: {service_name} - {operation}"
        if status_code:
            message += f" (HTTP {status_code})"
        if original_error:
            message += f" - {str(original_error)}"
        
        super().__init__(
            message=message,
            error_code="EXTERNAL_SERVICE_ERROR",
            details={
                "service_name": service_name,
                "operation": operation,
                "status_code": status_code,
                "original_error": str(original_error) if original_error else None
            }
        )


class MonitoringException(CrisisDetectionException):
    """Exception raised when monitoring operations fail"""
    
    def __init__(self, metric_name: str, operation: str, original_error: Optional[Exception] = None):
        message = f"Monitoring error for metric '{metric_name}': {operation}"
        if original_error:
            message += f" - {str(original_error)}"
        
        super().__init__(
            message=message,
            error_code="MONITORING_ERROR",
            details={
                "metric_name": metric_name,
                "operation": operation,
                "original_error": str(original_error) if original_error else None
            }
        )


class AlertException(CrisisDetectionException):
    """Exception raised when alert operations fail"""
    
    def __init__(self, alert_type: str, recipient: str, original_error: Optional[Exception] = None):
        message = f"Alert failed for type '{alert_type}' to '{recipient}'"
        if original_error:
            message += f" - {str(original_error)}"
        
        super().__init__(
            message=message,
            error_code="ALERT_ERROR",
            details={
                "alert_type": alert_type,
                "recipient": recipient,
                "original_error": str(original_error) if original_error else None
            }
        )


class DataProcessingException(CrisisDetectionException):
    """Exception raised when data processing fails"""
    
    def __init__(self, data_type: str, operation: str, original_error: Optional[Exception] = None):
        message = f"Data processing failed for {data_type}: {operation}"
        if original_error:
            message += f" - {str(original_error)}"
        
        super().__init__(
            message=message,
            error_code="DATA_PROCESSING_ERROR",
            details={
                "data_type": data_type,
                "operation": operation,
                "original_error": str(original_error) if original_error else None
            }
        )


class ModelTrainingException(CrisisDetectionException):
    """Exception raised when model training fails"""
    
    def __init__(self, model_type: str, training_data_size: int, original_error: Optional[Exception] = None):
        message = f"Model training failed for {model_type}"
        if original_error:
            message += f" - {str(original_error)}"
        
        super().__init__(
            message=message,
            error_code="MODEL_TRAINING_ERROR",
            details={
                "model_type": model_type,
                "training_data_size": training_data_size,
                "original_error": str(original_error) if original_error else None
            }
        )


class ModelEvaluationException(CrisisDetectionException):
    """Exception raised when model evaluation fails"""
    
    def __init__(self, model_name: str, evaluation_metric: str, original_error: Optional[Exception] = None):
        message = f"Model evaluation failed for {model_name} on metric {evaluation_metric}"
        if original_error:
            message += f" - {str(original_error)}"
        
        super().__init__(
            message=message,
            error_code="MODEL_EVALUATION_ERROR",
            details={
                "model_name": model_name,
                "evaluation_metric": evaluation_metric,
                "original_error": str(original_error) if original_error else None
            }
        )


class CacheException(CrisisDetectionException):
    """Exception raised when cache operations fail"""
    
    def __init__(self, cache_key: str, operation: str, original_error: Optional[Exception] = None):
        message = f"Cache operation failed for key '{cache_key}': {operation}"
        if original_error:
            message += f" - {str(original_error)}"
        
        super().__init__(
            message=message,
            error_code="CACHE_ERROR",
            details={
                "cache_key": cache_key,
                "operation": operation,
                "original_error": str(original_error) if original_error else None
            }
        )


class SecurityException(CrisisDetectionException):
    """Exception raised when security operations fail"""
    
    def __init__(self, security_check: str, details: Optional[Dict[str, Any]] = None):
        message = f"Security check failed: {security_check}"
        
        super().__init__(
            message=message,
            error_code="SECURITY_ERROR",
            details=details or {"security_check": security_check}
        )


class CrisisDetectionTimeoutException(CrisisDetectionException):
    """Exception raised when crisis detection times out"""
    
    def __init__(self, timeout_seconds: int, operation: str):
        message = f"Crisis detection timed out after {timeout_seconds} seconds during {operation}"
        
        super().__init__(
            message=message,
            error_code="TIMEOUT_ERROR",
            details={
                "timeout_seconds": timeout_seconds,
                "operation": operation
            }
        )


class CrisisDetectionResourceException(CrisisDetectionException):
    """Exception raised when system resources are insufficient"""
    
    def __init__(self, resource_type: str, required: int, available: int):
        message = f"Insufficient {resource_type}: required {required}, available {available}"
        
        super().__init__(
            message=message,
            error_code="RESOURCE_ERROR",
            details={
                "resource_type": resource_type,
                "required": required,
                "available": available
            }
        )
