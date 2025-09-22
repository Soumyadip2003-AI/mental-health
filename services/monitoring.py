"""
Monitoring and observability service
"""
import time
import psutil
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
import structlog
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
import json

logger = structlog.get_logger()


class SystemMetrics:
    """System metrics collector"""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics = defaultdict(list)
        self.alerts = []
        self.thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'response_time': 5.0,
            'error_rate': 0.05
        }
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available
            memory_total = memory.total
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_free = disk.free
            disk_total = disk.total
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info().rss
            process_cpu = process.cpu_percent()
            
            metrics = {
                'timestamp': datetime.utcnow().isoformat(),
                'system': {
                    'cpu_percent': cpu_percent,
                    'cpu_count': cpu_count,
                    'memory_percent': memory_percent,
                    'memory_available_gb': memory_available / (1024**3),
                    'memory_total_gb': memory_total / (1024**3),
                    'disk_percent': disk_percent,
                    'disk_free_gb': disk_free / (1024**3),
                    'disk_total_gb': disk_total / (1024**3),
                    'uptime_seconds': time.time() - self.start_time
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                },
                'process': {
                    'memory_rss_mb': process_memory / (1024**2),
                    'cpu_percent': process_cpu,
                    'num_threads': process.num_threads(),
                    'num_fds': process.num_fds() if hasattr(process, 'num_fds') else 0
                }
            }
            
            # Check thresholds and create alerts
            self._check_thresholds(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error("Failed to collect system metrics", error=str(e))
            return {}
    
    def _check_thresholds(self, metrics: Dict[str, Any]):
        """Check metrics against thresholds and create alerts"""
        system = metrics.get('system', {})
        
        # CPU threshold
        if system.get('cpu_percent', 0) > self.thresholds['cpu_usage']:
            self._create_alert('high_cpu', f"CPU usage is {system['cpu_percent']:.1f}%")
        
        # Memory threshold
        if system.get('memory_percent', 0) > self.thresholds['memory_usage']:
            self._create_alert('high_memory', f"Memory usage is {system['memory_percent']:.1f}%")
        
        # Disk threshold
        if system.get('disk_percent', 0) > self.thresholds['disk_usage']:
            self._create_alert('high_disk', f"Disk usage is {system['disk_percent']:.1f}%")
    
    def _create_alert(self, alert_type: str, message: str):
        """Create a new alert"""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.utcnow().isoformat(),
            'severity': 'warning'
        }
        
        self.alerts.append(alert)
        logger.warning("System alert created", alert=alert)
    
    def get_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        return self.alerts[-limit:]
    
    def clear_alerts(self):
        """Clear all alerts"""
        self.alerts.clear()


class ApplicationMetrics:
    """Application-specific metrics"""
    
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.crisis_detections = 0
        self.response_times = deque(maxlen=1000)
        self.error_types = defaultdict(int)
        self.endpoint_usage = defaultdict(int)
        self.user_activity = defaultdict(int)
        
        # Prometheus metrics
        self.registry = CollectorRegistry()
        self.request_counter = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )
        self.crisis_counter = Counter(
            'crisis_detections_total',
            'Total crisis detections',
            ['risk_level'],
            registry=self.registry
        )
        self.error_counter = Counter(
            'http_errors_total',
            'Total HTTP errors',
            ['error_type', 'endpoint'],
            registry=self.registry
        )
        self.active_users = Gauge(
            'active_users_total',
            'Number of active users',
            registry=self.registry
        )
        self.system_health = Gauge(
            'system_health_score',
            'System health score (0-100)',
            registry=self.registry
        )
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record a request"""
        self.request_count += 1
        self.response_times.append(duration)
        self.endpoint_usage[endpoint] += 1
        
        # Update Prometheus metrics
        self.request_counter.labels(method=method, endpoint=endpoint, status=status_code).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
        
        if status_code >= 400:
            self.error_count += 1
            self.error_counter.labels(error_type=str(status_code), endpoint=endpoint).inc()
    
    def record_crisis_detection(self, risk_level: str):
        """Record a crisis detection"""
        self.crisis_detections += 1
        self.crisis_counter.labels(risk_level=risk_level).inc()
    
    def record_user_activity(self, user_id: str):
        """Record user activity"""
        self.user_activity[user_id] = time.time()
        self.active_users.set(len(self.user_activity))
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get application metrics"""
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        error_rate = self.error_count / self.request_count if self.request_count > 0 else 0
        
        return {
            'requests': {
                'total': self.request_count,
                'errors': self.error_count,
                'error_rate': error_rate,
                'avg_response_time': avg_response_time
            },
            'crisis_detections': self.crisis_detections,
            'active_users': len(self.user_activity),
            'endpoint_usage': dict(self.endpoint_usage),
            'error_types': dict(self.error_types)
        }
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        return generate_latest(self.registry).decode('utf-8')


class MonitoringService:
    """Main monitoring service"""
    
    def __init__(self):
        self.system_metrics = SystemMetrics()
        self.app_metrics = ApplicationMetrics()
        self.is_running = False
        self.monitoring_task = None
        self.health_checks = {}
        self.custom_metrics = {}
    
    def start(self):
        """Start monitoring service"""
        if not self.is_running:
            self.is_running = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Monitoring service started")
    
    def stop(self):
        """Stop monitoring service"""
        if self.is_running:
            self.is_running = False
            if self.monitoring_task:
                self.monitoring_task.cancel()
            logger.info("Monitoring service stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Collect system metrics
                system_metrics = self.system_metrics.collect_metrics()
                
                # Update health score
                health_score = self._calculate_health_score(system_metrics)
                self.app_metrics.system_health.set(health_score)
                
                # Run custom health checks
                await self._run_health_checks()
                
                # Sleep for 30 seconds
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error("Monitoring loop error", error=str(e))
                await asyncio.sleep(60)  # Wait longer on error
    
    def _calculate_health_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall system health score (0-100)"""
        try:
            system = metrics.get('system', {})
            
            # CPU health (0-25 points)
            cpu_score = max(0, 25 - (system.get('cpu_percent', 0) - 50) * 0.5)
            
            # Memory health (0-25 points)
            memory_score = max(0, 25 - (system.get('memory_percent', 0) - 50) * 0.5)
            
            # Disk health (0-25 points)
            disk_score = max(0, 25 - (system.get('disk_percent', 0) - 50) * 0.5)
            
            # Application health (0-25 points)
            app_metrics = self.app_metrics.get_metrics()
            error_rate = app_metrics['requests']['error_rate']
            app_score = max(0, 25 - error_rate * 500)  # Penalize high error rates
            
            total_score = cpu_score + memory_score + disk_score + app_score
            return min(100, max(0, total_score))
            
        except Exception as e:
            logger.error("Failed to calculate health score", error=str(e))
            return 50.0  # Default to medium health
    
    async def _run_health_checks(self):
        """Run custom health checks"""
        for check_name, check_func in self.health_checks.items():
            try:
                result = await check_func()
                if not result.get('healthy', False):
                    logger.warning(f"Health check failed: {check_name}", result=result)
            except Exception as e:
                logger.error(f"Health check error: {check_name}", error=str(e))
    
    def add_health_check(self, name: str, check_func):
        """Add a custom health check"""
        self.health_checks[name] = check_func
        logger.info(f"Added health check: {name}")
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record a request for monitoring"""
        self.app_metrics.record_request(method, endpoint, status_code, duration)
    
    def record_crisis_detection(self, risk_level: str):
        """Record a crisis detection"""
        self.app_metrics.record_crisis_detection(risk_level)
    
    def record_user_activity(self, user_id: str):
        """Record user activity"""
        self.app_metrics.record_user_activity(user_id)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        system_metrics = self.system_metrics.collect_metrics()
        app_metrics = self.app_metrics.get_metrics()
        health_score = self._calculate_health_score(system_metrics)
        
        return {
            'health_score': health_score,
            'status': 'healthy' if health_score > 80 else 'degraded' if health_score > 50 else 'unhealthy',
            'system': system_metrics,
            'application': app_metrics,
            'alerts': self.system_metrics.get_alerts(10),
            'uptime': time.time() - self.system_metrics.start_time
        }
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics"""
        return self.app_metrics.get_prometheus_metrics()
    
    def get_uptime(self) -> float:
        """Get system uptime in seconds"""
        return time.time() - self.system_metrics.start_time
    
    def get_custom_metric(self, name: str) -> Any:
        """Get a custom metric value"""
        return self.custom_metrics.get(name)
    
    def set_custom_metric(self, name: str, value: Any):
        """Set a custom metric value"""
        self.custom_metrics[name] = value
        logger.debug(f"Custom metric set: {name} = {value}")
    
    def increment_custom_metric(self, name: str, value: float = 1.0):
        """Increment a custom metric value"""
        current = self.custom_metrics.get(name, 0)
        self.custom_metrics[name] = current + value
        logger.debug(f"Custom metric incremented: {name} += {value}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        system_health = self.get_system_health()
        prometheus_metrics = self.get_prometheus_metrics()
        
        return {
            'system_health': system_health,
            'prometheus_metrics': prometheus_metrics,
            'timestamp': datetime.utcnow().isoformat()
        }
