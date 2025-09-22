"""
FastAPI application for the Mental Health Crisis Detection System
"""
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import asyncio
import time
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import structlog
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Import our modules
from config import settings
from models.database import (
    DatabaseManager, create_analysis_record, create_alert_record,
    get_user_analyses, get_crisis_analyses, get_system_stats
)
from services.crisis_detector import CrisisDetectionService
from services.auth_service import AuthService
from services.rate_limiter import RateLimiter
from services.monitoring import MonitoringService
from schemas.requests import AnalysisRequest, BatchAnalysisRequest
from schemas.responses import AnalysisResponse, BatchAnalysisResponse, SystemStatsResponse
from utils.exceptions import CrisisDetectionException, RateLimitExceededException

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
CRISIS_DETECTIONS = Counter('crisis_detections_total', 'Total crisis detections', ['risk_level'])
ANALYSIS_DURATION = Histogram('analysis_duration_seconds', 'Analysis duration in seconds')

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Advanced Mental Health Crisis Detection System with AI/ML capabilities",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Security
security = HTTPBearer()

# Initialize services
db_manager = DatabaseManager(settings.database_url)
crisis_service = CrisisDetectionService()
auth_service = AuthService()
rate_limiter = RateLimiter(settings.rate_limit_per_minute)
monitoring_service = MonitoringService()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if settings.debug else ["yourdomain.com"]
)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# Dependency injection
async def get_db_session():
    """Get database session"""
    session = db_manager.get_session()
    try:
        yield session
    finally:
        db_manager.close_session(session)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    try:
        user = await auth_service.verify_token(credentials.credentials)
        return user
    except Exception as e:
        logger.error("Authentication failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Middleware for request tracking
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track requests for monitoring"""
    start_time = time.time()
    
    # Check rate limiting
    client_ip = request.client.host
    if not await rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    response = await call_next(request)
    
    # Record metrics
    duration = time.time() - start_time
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    return response


# Health check endpoints
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.app_version
    }


@app.get("/health/detailed", tags=["Health"])
async def detailed_health_check():
    """Detailed health check with system metrics"""
    try:
        # Check database connection
        session = db_manager.get_session()
        session.execute("SELECT 1")
        db_manager.close_session(session)
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    # Check model availability
    model_status = "healthy" if crisis_service.is_ready() else "unhealthy"
    
    return {
        "status": "healthy" if db_status == "healthy" and model_status == "healthy" else "unhealthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.app_version,
        "database": db_status,
        "models": model_status,
        "uptime": monitoring_service.get_uptime()
    }


# Main analysis endpoints
@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_text(
    request: AnalysisRequest,
    current_user: Optional[Dict] = Depends(get_current_user),
    session = Depends(get_db_session)
):
    """Analyze text for mental health crisis indicators"""
    start_time = time.time()
    
    try:
        # Perform analysis
        result = await crisis_service.analyze_text(
            text=request.text,
            confidence_threshold=request.confidence_threshold or settings.confidence_threshold,
            include_explanation=request.include_explanation
        )
        
        # Record analysis in database
        analysis_record = create_analysis_record(
            session=session,
            text_content=request.text,
            crisis_probability=result.crisis_probability,
            risk_level=result.risk_level,
            confidence_score=result.confidence_score,
            model_name=result.model_name,
            user_id=current_user.get("id") if current_user else None,
            key_features=result.key_features,
            explanation=result.explanation,
            processing_time=time.time() - start_time,
            ip_address=request.client_ip if hasattr(request, 'client_ip') else None,
            user_agent=request.user_agent if hasattr(request, 'user_agent') else None
        )
        
        # Create alert if crisis detected
        if result.risk_level in ["high", "critical"]:
            alert = create_alert_record(
                session=session,
                analysis_id=analysis_record.id,
                alert_type="crisis",
                severity=result.risk_level,
                message=f"Crisis detected with {result.confidence_score:.2f} confidence",
                user_id=current_user.get("id") if current_user else None
            )
            
            # Record crisis detection metric
            CRISIS_DETECTIONS.labels(risk_level=result.risk_level).inc()
        
        # Record analysis duration
        ANALYSIS_DURATION.observe(time.time() - start_time)
        
        return AnalysisResponse(
            analysis_id=analysis_record.id,
            crisis_probability=result.crisis_probability,
            risk_level=result.risk_level,
            confidence_score=result.confidence_score,
            key_features=result.key_features,
            explanation=result.explanation,
            processing_time=time.time() - start_time,
            timestamp=analysis_record.created_at.isoformat()
        )
        
    except CrisisDetectionException as e:
        logger.error("Crisis detection failed", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error("Unexpected error during analysis", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@app.post("/analyze/batch", response_model=BatchAnalysisResponse, tags=["Analysis"])
async def analyze_batch(
    request: BatchAnalysisRequest,
    current_user: Optional[Dict] = Depends(get_current_user),
    session = Depends(get_db_session)
):
    """Analyze multiple texts in batch"""
    start_time = time.time()
    
    try:
        # Process batch analysis
        results = await crisis_service.analyze_batch(
            texts=request.texts,
            confidence_threshold=request.confidence_threshold or settings.confidence_threshold,
            include_explanation=request.include_explanation
        )
        
        # Record analyses in database
        analysis_records = []
        for i, (text, result) in enumerate(zip(request.texts, results)):
            analysis_record = create_analysis_record(
                session=session,
                text_content=text,
                crisis_probability=result.crisis_probability,
                risk_level=result.risk_level,
                confidence_score=result.confidence_score,
                model_name=result.model_name,
                user_id=current_user.get("id") if current_user else None,
                key_features=result.key_features,
                explanation=result.explanation,
                processing_time=result.processing_time,
                ip_address=request.client_ip if hasattr(request, 'client_ip') else None,
                user_agent=request.user_agent if hasattr(request, 'user_agent') else None
            )
            analysis_records.append(analysis_record)
        
        return BatchAnalysisResponse(
            analyses=[AnalysisResponse(
                analysis_id=record.id,
                crisis_probability=result.crisis_probability,
                risk_level=result.risk_level,
                confidence_score=result.confidence_score,
                key_features=result.key_features,
                explanation=result.explanation,
                processing_time=result.processing_time,
                timestamp=record.created_at.isoformat()
            ) for record, result in zip(analysis_records, results)],
            total_processed=len(results),
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error("Batch analysis failed", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


# User endpoints
@app.get("/user/analyses", response_model=List[AnalysisResponse], tags=["User"])
async def get_user_analyses_endpoint(
    limit: int = 100,
    current_user: Dict = Depends(get_current_user),
    session = Depends(get_db_session)
):
    """Get user's analysis history"""
    analyses = get_user_analyses(session, current_user["id"], limit)
    
    return [AnalysisResponse(
        analysis_id=analysis.id,
        crisis_probability=analysis.crisis_probability,
        risk_level=analysis.risk_level,
        confidence_score=analysis.confidence_score,
        key_features=analysis.key_features,
        explanation=analysis.explanation,
        processing_time=analysis.processing_time,
        timestamp=analysis.created_at.isoformat()
    ) for analysis in analyses]


# Admin endpoints
@app.get("/admin/stats", response_model=SystemStatsResponse, tags=["Admin"])
async def get_system_stats_endpoint(
    current_user: Dict = Depends(get_current_user),
    session = Depends(get_db_session)
):
    """Get system statistics (admin only)"""
    if not current_user.get("is_admin", False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    
    stats = get_system_stats(session)
    return SystemStatsResponse(**stats)


@app.get("/admin/crises", response_model=List[AnalysisResponse], tags=["Admin"])
async def get_crisis_analyses_endpoint(
    limit: int = 100,
    current_user: Dict = Depends(get_current_user),
    session = Depends(get_db_session)
):
    """Get recent crisis analyses (admin only)"""
    if not current_user.get("is_admin", False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    
    analyses = get_crisis_analyses(session, limit)
    
    return [AnalysisResponse(
        analysis_id=analysis.id,
        crisis_probability=analysis.crisis_probability,
        risk_level=analysis.risk_level,
        confidence_score=analysis.confidence_score,
        key_features=analysis.key_features,
        explanation=analysis.explanation,
        processing_time=analysis.processing_time,
        timestamp=analysis.created_at.isoformat()
    ) for analysis in analyses]


# Monitoring endpoints
@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Error handlers
@app.exception_handler(CrisisDetectionException)
async def crisis_detection_exception_handler(request: Request, exc: CrisisDetectionException):
    """Handle crisis detection exceptions"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc), "type": "crisis_detection_error"}
    )


@app.exception_handler(RateLimitExceededException)
async def rate_limit_exception_handler(request: Request, exc: RateLimitExceededException):
    """Handle rate limit exceptions"""
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "Rate limit exceeded", "retry_after": exc.retry_after}
    )


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Mental Health Crisis Detection System", version=settings.app_version)
    
    # Create database tables
    db_manager.create_tables()
    
    # Initialize crisis detection service
    await crisis_service.initialize()
    
    # Initialize monitoring
    monitoring_service.start()
    
    logger.info("System startup completed")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Mental Health Crisis Detection System")
    
    # Stop monitoring
    monitoring_service.stop()
    
    logger.info("System shutdown completed")


# Main application entry point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
