"""
Rate limiting service for API endpoints
"""
import time
import asyncio
from typing import Dict, Optional
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter using sliding window algorithm"""
    
    def __init__(self, max_requests: int = 60, window_size: int = 60):
        """
        Initialize rate limiter
        
        Args:
            max_requests: Maximum number of requests allowed
            window_size: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_size = window_size
        self.requests = defaultdict(deque)
        self.lock = asyncio.Lock()
    
    async def is_allowed(self, client_id: str) -> bool:
        """
        Check if a request is allowed for the given client
        
        Args:
            client_id: Unique identifier for the client (IP, user ID, etc.)
            
        Returns:
            True if request is allowed, False otherwise
        """
        async with self.lock:
            now = time.time()
            client_requests = self.requests[client_id]
            
            # Remove old requests outside the window
            while client_requests and client_requests[0] <= now - self.window_size:
                client_requests.popleft()
            
            # Check if we're under the limit
            if len(client_requests) < self.max_requests:
                client_requests.append(now)
                return True
            else:
                return False
    
    async def get_remaining_requests(self, client_id: str) -> int:
        """
        Get the number of remaining requests for a client
        
        Args:
            client_id: Unique identifier for the client
            
        Returns:
            Number of remaining requests
        """
        async with self.lock:
            now = time.time()
            client_requests = self.requests[client_id]
            
            # Remove old requests outside the window
            while client_requests and client_requests[0] <= now - self.window_size:
                client_requests.popleft()
            
            return max(0, self.max_requests - len(client_requests))
    
    async def get_reset_time(self, client_id: str) -> float:
        """
        Get the time when the rate limit will reset for a client
        
        Args:
            client_id: Unique identifier for the client
            
        Returns:
            Unix timestamp when the rate limit resets
        """
        async with self.lock:
            client_requests = self.requests[client_id]
            
            if not client_requests:
                return time.time()
            
            # The reset time is when the oldest request in the window will expire
            oldest_request = client_requests[0]
            return oldest_request + self.window_size
    
    async def reset_client(self, client_id: str):
        """
        Reset the rate limit for a specific client
        
        Args:
            client_id: Unique identifier for the client
        """
        async with self.lock:
            if client_id in self.requests:
                del self.requests[client_id]
    
    async def get_stats(self) -> Dict[str, int]:
        """
        Get rate limiter statistics
        
        Returns:
            Dictionary with statistics
        """
        async with self.lock:
            now = time.time()
            active_clients = 0
            total_requests = 0
            
            for client_requests in self.requests.values():
                # Remove old requests
                while client_requests and client_requests[0] <= now - self.window_size:
                    client_requests.popleft()
                
                if client_requests:
                    active_clients += 1
                    total_requests += len(client_requests)
            
            return {
                "active_clients": active_clients,
                "total_requests": total_requests,
                "max_requests_per_client": self.max_requests,
                "window_size_seconds": self.window_size
            }


class AdvancedRateLimiter:
    """Advanced rate limiter with multiple tiers and burst handling"""
    
    def __init__(self):
        """Initialize advanced rate limiter with multiple tiers"""
        self.limiters = {
            "free": RateLimiter(max_requests=10, window_size=60),      # 10 requests per minute
            "basic": RateLimiter(max_requests=100, window_size=60),    # 100 requests per minute
            "premium": RateLimiter(max_requests=1000, window_size=60),  # 1000 requests per minute
            "admin": RateLimiter(max_requests=10000, window_size=60),  # 10000 requests per minute
        }
        
        # Burst handling
        self.burst_limits = {
            "free": 5,      # 5 requests in 10 seconds
            "basic": 20,    # 20 requests in 10 seconds
            "premium": 100, # 100 requests in 10 seconds
            "admin": 500,   # 500 requests in 10 seconds
        }
        
        self.burst_limiters = {
            tier: RateLimiter(max_requests=limit, window_size=10)
            for tier, limit in self.burst_limits.items()
        }
    
    async def is_allowed(self, client_id: str, tier: str = "free") -> bool:
        """
        Check if a request is allowed for the given client and tier
        
        Args:
            client_id: Unique identifier for the client
            tier: Client tier (free, basic, premium, admin)
            
        Returns:
            True if request is allowed, False otherwise
        """
        if tier not in self.limiters:
            tier = "free"
        
        # Check burst limit first
        burst_allowed = await self.burst_limiters[tier].is_allowed(client_id)
        if not burst_allowed:
            logger.warning(f"Burst limit exceeded for client {client_id} (tier: {tier})")
            return False
        
        # Check regular limit
        regular_allowed = await self.limiters[tier].is_allowed(client_id)
        if not regular_allowed:
            logger.warning(f"Rate limit exceeded for client {client_id} (tier: {tier})")
            return False
        
        return True
    
    async def get_remaining_requests(self, client_id: str, tier: str = "free") -> Dict[str, int]:
        """
        Get remaining requests for both burst and regular limits
        
        Args:
            client_id: Unique identifier for the client
            tier: Client tier
            
        Returns:
            Dictionary with remaining requests for both limits
        """
        if tier not in self.limiters:
            tier = "free"
        
        regular_remaining = await self.limiters[tier].get_remaining_requests(client_id)
        burst_remaining = await self.burst_limiters[tier].get_remaining_requests(client_id)
        
        return {
            "regular_remaining": regular_remaining,
            "burst_remaining": burst_remaining,
            "tier": tier
        }
    
    async def get_reset_times(self, client_id: str, tier: str = "free") -> Dict[str, float]:
        """
        Get reset times for both burst and regular limits
        
        Args:
            client_id: Unique identifier for the client
            tier: Client tier
            
        Returns:
            Dictionary with reset times for both limits
        """
        if tier not in self.limiters:
            tier = "free"
        
        regular_reset = await self.limiters[tier].get_reset_time(client_id)
        burst_reset = await self.burst_limiters[tier].get_reset_time(client_id)
        
        return {
            "regular_reset": regular_reset,
            "burst_reset": burst_reset,
            "tier": tier
        }
    
    async def reset_client(self, client_id: str, tier: str = "free"):
        """
        Reset rate limits for a specific client and tier
        
        Args:
            client_id: Unique identifier for the client
            tier: Client tier
        """
        if tier not in self.limiters:
            return
        
        await self.limiters[tier].reset_client(client_id)
        await self.burst_limiters[tier].reset_client(client_id)
    
    async def get_all_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get statistics for all tiers
        
        Returns:
            Dictionary with statistics for all tiers
        """
        stats = {}
        
        for tier in self.limiters:
            regular_stats = await self.limiters[tier].get_stats()
            burst_stats = await self.burst_limiters[tier].get_stats()
            
            stats[tier] = {
                "regular": regular_stats,
                "burst": burst_stats
            }
        
        return stats


class IPBasedRateLimiter:
    """IP-based rate limiter with geographic and behavioral analysis"""
    
    def __init__(self):
        """Initialize IP-based rate limiter"""
        self.rate_limiter = AdvancedRateLimiter()
        self.suspicious_ips = set()
        self.blocked_ips = set()
        self.ip_behavior = defaultdict(list)
    
    async def is_allowed(self, client_ip: str, user_tier: str = "free") -> bool:
        """
        Check if a request is allowed for the given IP
        
        Args:
            client_ip: Client IP address
            user_tier: User tier
            
        Returns:
            True if request is allowed, False otherwise
        """
        # Check if IP is blocked
        if client_ip in self.blocked_ips:
            logger.warning(f"Blocked IP attempted access: {client_ip}")
            return False
        
        # Check if IP is suspicious
        if client_ip in self.suspicious_ips:
            # Apply stricter limits for suspicious IPs
            user_tier = "free"
        
        # Record IP behavior
        self.ip_behavior[client_ip].append(time.time())
        
        # Clean old behavior data (keep only last hour)
        cutoff = time.time() - 3600
        self.ip_behavior[client_ip] = [
            timestamp for timestamp in self.ip_behavior[client_ip] 
            if timestamp > cutoff
        ]
        
        # Check for suspicious behavior
        if len(self.ip_behavior[client_ip]) > 1000:  # More than 1000 requests per hour
            self.suspicious_ips.add(client_ip)
            logger.warning(f"IP marked as suspicious: {client_ip}")
        
        # Apply rate limiting
        return await self.rate_limiter.is_allowed(client_ip, user_tier)
    
    async def block_ip(self, client_ip: str, reason: str = "Suspicious activity"):
        """
        Block an IP address
        
        Args:
            client_ip: IP address to block
            reason: Reason for blocking
        """
        self.blocked_ips.add(client_ip)
        logger.warning(f"IP blocked: {client_ip}, reason: {reason}")
    
    async def unblock_ip(self, client_ip: str):
        """
        Unblock an IP address
        
        Args:
            client_ip: IP address to unblock
        """
        self.blocked_ips.discard(client_ip)
        logger.info(f"IP unblocked: {client_ip}")
    
    async def get_ip_status(self, client_ip: str) -> Dict[str, Any]:
        """
        Get status information for an IP address
        
        Args:
            client_ip: IP address to check
            
        Returns:
            Dictionary with IP status information
        """
        is_blocked = client_ip in self.blocked_ips
        is_suspicious = client_ip in self.suspicious_ips
        
        behavior_count = len(self.ip_behavior.get(client_ip, []))
        
        return {
            "ip": client_ip,
            "is_blocked": is_blocked,
            "is_suspicious": is_suspicious,
            "request_count_last_hour": behavior_count,
            "status": "blocked" if is_blocked else "suspicious" if is_suspicious else "normal"
        }
