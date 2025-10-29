"""
Logging configuration and utilities for RedFive FastAPI application.
Handles structured logging to both files and database.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import create_engine, text
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Create logger instance
logger = logging.getLogger(__name__)

class DatabaseLogger:
    """Handles logging to database for API requests and responses."""
    
    def __init__(self, connection_string: Optional[str] = None):
        """Initialize database logger with logging connection string."""
        self.connection_string = connection_string or os.getenv("WRITE_CONNECTION_STRING")
        if not self.connection_string:
            raise ValueError("Logging connection string not configured. Please set WRITE_CONNECTION_STRING environment variable.")
        
        self.engine = create_engine(self.connection_string)
    
    def log_request(self, 
                   request_id: str,
                   user_id: Optional[str] = None,
                   user_email: Optional[str] = None,
                   user_name: Optional[str] = None,
                   http_method: str = None,
                   endpoint_path: str = None,
                   query_params: Optional[Dict] = None,
                   request_headers: Optional[Dict] = None,
                   request_body: Optional[str] = None,
                   client_ip: Optional[str] = None,
                   user_agent: Optional[str] = None,
                   sql_query: Optional[str] = None,
                   query_type: Optional[str] = None):
        """Log API request to database."""
        try:
            with self.engine.connect() as conn:
                # Set schema search path
                conn.execute(text("SET search_path TO redfive, upstream, public"))
                
                # Prepare request body size
                request_body_size = len(request_body.encode('utf-8')) if request_body else 0
                
                # Convert headers to JSON, excluding sensitive information
                safe_headers = self._sanitize_headers(request_headers) if request_headers else None
                
                insert_sql = text("""
                    INSERT INTO redfive.api_request_logs (
                        request_id, user_id, user_email, user_name,
                        http_method, endpoint_path, query_params, request_headers,
                        request_body, request_body_size, client_ip, user_agent,
                        sql_query, query_type, request_timestamp
                    ) VALUES (
                        :request_id, :user_id, :user_email, :user_name,
                        :http_method, :endpoint_path, :query_params, :request_headers,
                        :request_body, :request_body_size, :client_ip, :user_agent,
                        :sql_query, :query_type, :request_timestamp
                    )
                """)
                
                conn.execute(insert_sql, {
                    'request_id': request_id,
                    'user_id': user_id,
                    'user_email': user_email,
                    'user_name': user_name,
                    'http_method': http_method,
                    'endpoint_path': endpoint_path,
                    'query_params': json.dumps(query_params) if query_params else None,
                    'request_headers': json.dumps(safe_headers) if safe_headers else None,
                    'request_body': request_body,
                    'request_body_size': request_body_size,
                    'client_ip': client_ip,
                    'user_agent': user_agent,
                    'sql_query': sql_query,
                    'query_type': query_type,
                    'request_timestamp': datetime.now()
                })
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to log request to database: {str(e)}")
    
    def log_response(self,
                    request_id: str,
                    response_status_code: int,
                    response_body: Optional[str] = None,
                    response_headers: Optional[Dict] = None,
                    processing_time_ms: Optional[int] = None,
                    error_message: Optional[str] = None,
                    error_type: Optional[str] = None,
                    stack_trace: Optional[str] = None):
        """Log API response to database."""
        try:
            with self.engine.connect() as conn:
                # Set schema search path
                conn.execute(text("SET search_path TO redfive, upstream, public"))
                
                # Prepare response body size
                response_body_size = len(response_body.encode('utf-8')) if response_body else 0
                
                # Truncate response body if too large (keep first 10KB)
                if response_body and len(response_body) > 10000:
                    response_body = response_body[:10000] + "... [TRUNCATED]"
                
                update_sql = text("""
                    UPDATE redfive.api_request_logs 
                    SET response_status_code = :response_status_code,
                        response_body = :response_body,
                        response_body_size = :response_body_size,
                        response_headers = :response_headers,
                        processing_time_ms = :processing_time_ms,
                        error_message = :error_message,
                        error_type = :error_type,
                        stack_trace = :stack_trace,
                        response_timestamp = :response_timestamp,
                        updated_at = :updated_at
                    WHERE request_id = :request_id
                """)
                
                conn.execute(update_sql, {
                    'request_id': request_id,
                    'response_status_code': response_status_code,
                    'response_body': response_body,
                    'response_body_size': response_body_size,
                    'response_headers': json.dumps(response_headers) if response_headers else None,
                    'processing_time_ms': processing_time_ms,
                    'error_message': error_message,
                    'error_type': error_type,
                    'stack_trace': stack_trace,
                    'response_timestamp': datetime.now(),
                    'updated_at': datetime.now()
                })
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to log response to database: {str(e)}")
    
    def _sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Remove sensitive information from headers."""
        sensitive_headers = {'authorization', 'cookie', 'x-api-key', 'x-auth-token'}
        return {k: v for k, v in headers.items() if k.lower() not in sensitive_headers}

# Global database logger instance
db_logger = None

def get_db_logger() -> DatabaseLogger:
    """Get the global database logger instance."""
    global db_logger
    if db_logger is None:
        db_logger = DatabaseLogger()
    return db_logger

def log_api_request(request_id: str, **kwargs):
    """Convenience function to log API request."""
    get_db_logger().log_request(request_id, **kwargs)

def log_api_response(request_id: str, **kwargs):
    """Convenience function to log API response."""
    get_db_logger().log_response(request_id, **kwargs)
