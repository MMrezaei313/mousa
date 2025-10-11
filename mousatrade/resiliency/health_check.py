import time
import threading
from typing import Dict, List, Callable, Any
from dataclasses import dataclass
from enum import Enum

class HealthStatus(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"

@dataclass
class HealthCheckResult:
    service: str
    status: HealthStatus
    response_time: float
    message: str = ""
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class HealthChecker:
    """
    Health monitoring for trading system components
    """
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.health_checks = {}
        self.results = {}
        self._running = False
        self._thread = None
        
    def register_check(self, name: str, check_func: Callable[[], bool], 
                      timeout: int = 10, critical: bool = True):
        """Register a health check"""
        self.health_checks[name] = {
            'function': check_func,
            'timeout': timeout,
            'critical': critical
        }
    
    def check_exchange_connectivity(self, exchange_name: str, api_client) -> bool:
        """Health check for exchange connectivity"""
        try:
            start_time = time.time()
            # Simple API call to check connectivity
            api_client.ping()
            response_time = time.time() - start_time
            
            status = HealthStatus.HEALTHY if response_time < 2.0 else HealthStatus.DEGRADED
            self.results[f"exchange_{exchange_name}"] = HealthCheckResult(
                service=f"exchange_{exchange_name}",
                status=status,
                response_time=response_time,
                message=f"Response time: {response_time:.3f}s"
            )
            return status == HealthStatus.HEALTHY
            
        except Exception as e:
            self.results[f"exchange_{exchange_name}"] = HealthCheckResult(
                service=f"exchange_{exchange_name}",
                status=HealthStatus.UNHEALTHY,
                response_time=0,
                message=f"Connection failed: {str(e)}"
            )
            return False
    
    def check_database_connectivity(self) -> bool:
        """Health check for database"""
        try:
            start_time = time.time()
            # Your database connectivity check
            response_time = time.time() - start_time
            
            status = HealthStatus.HEALTHY if response_time < 1.0 else HealthStatus.DEGRADED
            self.results["database"] = HealthCheckResult(
                service="database",
                status=status,
                response_time=response_time
            )
            return status == HealthStatus.HEALTHY
            
        except Exception as e:
            self.results["database"] = HealthCheckResult(
                service="database",
                status=HealthStatus.UNHEALTHY,
                response_time=0,
                message=f"Database error: {str(e)}"
            )
            return False
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status"""
        if not self.results:
            return HealthStatus.HEALTHY
        
        critical_services = [name for name, check in self.health_checks.items() 
                           if check.get('critical', True)]
        
        for service in critical_services:
            if service in self.results:
                if self.results[service].status == HealthStatus.UNHEALTHY:
                    return HealthStatus.UNHEALTHY
        
        # Check if any service is degraded
        for result in self.results.values():
            if result.status == HealthStatus.DEGRADED:
                return HealthStatus.DEGRADED
        
        return HealthStatus.HEALTHY
    
    def start_monitoring(self):
        """Start background health monitoring"""
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self._running = False
        if self._thread:
            self._thread.join()
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._running:
            self.run_health_checks()
            time.sleep(self.check_interval)
    
    def run_health_checks(self):
        """Run all registered health checks"""
        for name, check_info in self.health_checks.items():
            try:
                check_info['function']()
            except Exception as e:
                print(f"Health check failed for {name}: {str(e)}")
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        return {
            "overall_status": self.get_overall_status().value,
            "services": {
                name: {
                    "status": result.status.value,
                    "response_time": result.response_time,
                    "message": result.message,
                    "timestamp": result.timestamp
                }
                for name, result in self.results.items()
            },
            "timestamp": time.time()
        }
