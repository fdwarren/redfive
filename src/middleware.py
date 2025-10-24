"""
FastAPI middleware for comprehensive request/response logging.
Captures all API requests and responses with user context and performance metrics.
"""

import json
import uuid
import time
import traceback
from datetime import datetime
from typing import Callable, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from logging_config import logger, log_api_request, log_api_response
from auth import verify_token

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all API requests and responses."""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Generate unique request ID for tracing
        request_id = str(uuid.uuid4())
        
        # Start timing
        start_time = time.time()
        
        # Extract user information from JWT token
        user_info = self._extract_user_info(request)
        
        # Get client information
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        
        # Extract request data
        request_body = await self._extract_request_body(request)
        query_params = dict(request.query_params)
        request_headers = dict(request.headers)
        
        # Determine query type based on endpoint
        query_type = self._determine_query_type(request.url.path)
        sql_query = None
        
        # Extract SQL query if this is a SQL-related request
        if query_type in ["sql_generation", "sql_execution"] and request_body:
            try:
                body_data = json.loads(request_body) if isinstance(request_body, str) else request_body
                sql_query = body_data.get("query") or body_data.get("sql")
            except (json.JSONDecodeError, AttributeError):
                pass
        
        # Log request to database
        try:
            log_api_request(
                request_id=request_id,
                user_id=user_info.get("user_id"),
                user_email=user_info.get("user_email"),
                user_name=user_info.get("user_name"),
                http_method=request.method,
                endpoint_path=request.url.path,
                query_params=query_params,
                request_headers=request_headers,
                request_body=request_body,
                client_ip=client_ip,
                user_agent=user_agent,
                sql_query=sql_query,
                query_type=query_type
            )
        except Exception as e:
            logger.error(f"Failed to log request: {str(e)}")
        
        # Log to file
        logger.info(
            f"REQUEST - ID: {request_id} | User: {user_info.get('user_email', 'anonymous')} | "
            f"Method: {request.method} | Path: {request.url.path} | "
            f"Query: {query_params} | Body Size: {len(request_body) if request_body else 0} bytes"
        )
        
        # Process request
        response = None
        error_info = None
        
        try:
            response = await call_next(request)
        except Exception as e:
            # Capture exception details
            error_info = {
                'error_message': str(e),
                'error_type': type(e).__name__,
                'stack_trace': traceback.format_exc()
            }
            logger.error(f"Request processing error - ID: {request_id} | Error: {str(e)}", exc_info=True)
            raise
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Extract response data
        response_body = None
        response_headers = None
        status_code = None
        
        if response:
            status_code = response.status_code
            
            # Try to get response body (this might not work for all response types)
            try:
                if hasattr(response, 'body'):
                    response_body = response.body.decode('utf-8') if response.body else None
            except Exception:
                pass
            
            # Get response headers
            response_headers = dict(response.headers)
        
        # Log response to database
        try:
            log_api_response(
                request_id=request_id,
                response_status_code=status_code,
                response_body=response_body,
                response_headers=response_headers,
                processing_time_ms=processing_time_ms,
                error_message=error_info.get('error_message') if error_info else None,
                error_type=error_info.get('error_type') if error_info else None,
                stack_trace=error_info.get('stack_trace') if error_info else None
            )
        except Exception as e:
            logger.error(f"Failed to log response: {str(e)}")
        
        # Log to file
        logger.info(
            f"RESPONSE - ID: {request_id} | Status: {status_code} | "
            f"Processing Time: {processing_time_ms}ms"
        )
        
        return response
    
    def _extract_user_info(self, request: Request) -> dict:
        """Extract user information from JWT token and request state."""
        try:
            # First, try to get user info from request state (set by auth dependency)
            user_email = getattr(request.state, 'user_email', None)
            user_name = getattr(request.state, 'user_name', None)
            user_id = getattr(request.state, 'user_id', None)
            
            # If we have user info from request state, use it
            if user_email and user_name:
                return {
                    "user_id": user_id,
                    "user_email": user_email,
                    "user_name": user_name
                }
            
            # Fallback: try to extract from JWT token directly
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                payload = verify_token(token)
                if payload:
                    user_id = payload.get("sub")
                    
                    # Try to get user info from the in-memory user database
                    from auth import fake_users_db
                    user_data = fake_users_db.get(user_id)
                    if user_data:
                        return {
                            "user_id": user_id,
                            "user_email": user_data.email,
                            "user_name": user_data.name
                        }
                    
                    # If no user data found, return what we have
                    return {
                        "user_id": user_id,
                        "user_email": None,
                        "user_name": None
                    }
        except Exception as e:
            logger.debug(f"Failed to extract user info: {str(e)}")
        
        return {"user_id": None, "user_email": None, "user_name": None}
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers first (for reverse proxy setups)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct client IP
        return request.client.host if request.client else "unknown"
    
    async def _extract_request_body(self, request: Request) -> Optional[str]:
        """Extract request body as string."""
        try:
            if request.method in ["POST", "PUT", "PATCH"]:
                body = await request.body()
                if body:
                    # Try to decode as JSON for better formatting
                    try:
                        json_data = json.loads(body.decode('utf-8'))
                        return json.dumps(json_data, indent=2)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        return body.decode('utf-8', errors='replace')
        except Exception as e:
            logger.debug(f"Failed to extract request body: {str(e)}")
        
        return None
    
    def _determine_query_type(self, path: str) -> Optional[str]:
        """Determine the type of query based on endpoint path."""
        path_to_type = {
            "/generate-sql": "sql_generation",
            "/execute-sql": "sql_execution", 
            "/examine-sql": "sql_examination",
            "/validate-models": "model_validation",
            "/refresh-embeddings": "embedding_refresh",
            "/get-models": "model_retrieval",
            "/clear-cache": "cache_management",
            "/auth/google": "authentication",
            "/auth/me": "user_info",
            "/auth/refresh": "token_refresh",
            "/auth/logout": "logout"
        }
        
        return path_to_type.get(path)
