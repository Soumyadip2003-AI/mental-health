"""
Authentication and authorization service
"""
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from passlib.context import CryptContext
from jose import JWTError, jwt as jose_jwt
import logging

from config import settings

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    """Authentication and authorization service"""
    
    def __init__(self):
        self.secret_key = settings.secret_key
        self.algorithm = "HS256"
        self.access_token_expire_minutes = settings.access_token_expire_minutes
    
    def hash_password(self, password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        
        try:
            encoded_jwt = jose_jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            return encoded_jwt
        except Exception as e:
            logger.error(f"Failed to create access token: {e}")
            raise
    
    async def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode a JWT token"""
        try:
            payload = jose_jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")
            
            if username is None:
                raise ValueError("Invalid token payload")
            
            return {"username": username, "id": payload.get("id"), "is_admin": payload.get("is_admin", False)}
            
        except JWTError as e:
            logger.error(f"JWT verification failed: {e}")
            raise ValueError("Invalid token")
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            raise ValueError("Token verification failed")
    
    def create_user_token(self, user_id: int, username: str, is_admin: bool = False) -> str:
        """Create a token for a user"""
        data = {
            "sub": username,
            "id": user_id,
            "is_admin": is_admin,
            "iat": datetime.utcnow()
        }
        
        return self.create_access_token(data)
    
    def refresh_token(self, token: str) -> str:
        """Refresh an access token"""
        try:
            payload = jose_jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username = payload.get("sub")
            user_id = payload.get("id")
            is_admin = payload.get("is_admin", False)
            
            if username is None or user_id is None:
                raise ValueError("Invalid token payload")
            
            return self.create_user_token(user_id, username, is_admin)
            
        except JWTError as e:
            logger.error(f"Token refresh failed: {e}")
            raise ValueError("Invalid token")
    
    def decode_token(self, token: str) -> Dict[str, Any]:
        """Decode a token without verification (for debugging)"""
        try:
            payload = jose_jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError as e:
            logger.error(f"Token decode failed: {e}")
            raise ValueError("Invalid token")
    
    def is_token_expired(self, token: str) -> bool:
        """Check if a token is expired"""
        try:
            payload = jose_jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            exp = payload.get("exp")
            
            if exp is None:
                return True
            
            return datetime.utcnow() > datetime.fromtimestamp(exp)
            
        except JWTError:
            return True
    
    def get_token_expiry(self, token: str) -> Optional[datetime]:
        """Get the expiry time of a token"""
        try:
            payload = jose_jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            exp = payload.get("exp")
            
            if exp is None:
                return None
            
            return datetime.fromtimestamp(exp)
            
        except JWTError:
            return None
    
    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password strength"""
        result = {
            "is_valid": True,
            "score": 0,
            "issues": []
        }
        
        if len(password) < 8:
            result["issues"].append("Password must be at least 8 characters long")
            result["is_valid"] = False
        
        if not any(c.isupper() for c in password):
            result["issues"].append("Password must contain at least one uppercase letter")
            result["score"] += 1
        
        if not any(c.islower() for c in password):
            result["issues"].append("Password must contain at least one lowercase letter")
            result["score"] += 1
        
        if not any(c.isdigit() for c in password):
            result["issues"].append("Password must contain at least one digit")
            result["score"] += 1
        
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            result["issues"].append("Password must contain at least one special character")
            result["score"] += 1
        
        # Check for common patterns
        common_patterns = ["123", "abc", "password", "qwerty", "admin"]
        if any(pattern in password.lower() for pattern in common_patterns):
            result["issues"].append("Password contains common patterns")
            result["score"] -= 1
        
        return result
    
    def generate_secure_password(self, length: int = 12) -> str:
        """Generate a secure random password"""
        import secrets
        import string
        
        characters = string.ascii_letters + string.digits + "!@#$%^&*()_+-=[]{}|;:,.<>?"
        password = ''.join(secrets.choice(characters) for _ in range(length))
        
        return password
    
    def create_password_reset_token(self, user_id: int, email: str) -> str:
        """Create a password reset token"""
        data = {
            "sub": email,
            "user_id": user_id,
            "type": "password_reset",
            "iat": datetime.utcnow()
        }
        
        # Shorter expiry for password reset
        expires_delta = timedelta(hours=1)
        return self.create_access_token(data, expires_delta)
    
    def verify_password_reset_token(self, token: str) -> Dict[str, Any]:
        """Verify a password reset token"""
        try:
            payload = jose_jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            if payload.get("type") != "password_reset":
                raise ValueError("Invalid token type")
            
            return {
                "user_id": payload.get("user_id"),
                "email": payload.get("sub")
            }
            
        except JWTError as e:
            logger.error(f"Password reset token verification failed: {e}")
            raise ValueError("Invalid password reset token")
    
    def create_email_verification_token(self, user_id: int, email: str) -> str:
        """Create an email verification token"""
        data = {
            "sub": email,
            "user_id": user_id,
            "type": "email_verification",
            "iat": datetime.utcnow()
        }
        
        # Longer expiry for email verification
        expires_delta = timedelta(days=7)
        return self.create_access_token(data, expires_delta)
    
    def verify_email_verification_token(self, token: str) -> Dict[str, Any]:
        """Verify an email verification token"""
        try:
            payload = jose_jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            if payload.get("type") != "email_verification":
                raise ValueError("Invalid token type")
            
            return {
                "user_id": payload.get("user_id"),
                "email": payload.get("sub")
            }
            
        except JWTError as e:
            logger.error(f"Email verification token verification failed: {e}")
            raise ValueError("Invalid email verification token")
