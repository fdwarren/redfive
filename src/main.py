"""
FastAPI server for RedFive SQL Generator.
This module provides HTTP endpoints for the SQL generation functionality.
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from core import RedFiveCore
from auth import (
    get_current_active_user, 
    get_current_user_with_context,
    create_access_token, 
    create_refresh_token,
    create_or_update_user,
    verify_google_token,
    verify_token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    Token,
    User,
    GoogleTokenRequest
)
from middleware import RequestLoggingMiddleware
from logging_config import logger
from queries import SavedQueryManager
from typing import List, Optional

load_dotenv()

app = FastAPI(title="RedFive SQL Generator", description="Generate SQL from natural language queries")

# Get CORS origins from environment variable
cors_origins = os.getenv("CORS_ALLOW_ORIGINS").split(",")
cors_origins = [origin.strip() for origin in cors_origins if origin.strip()]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request logging middleware
app.add_middleware(RequestLoggingMiddleware)

# Initialize the core business logic
core = RedFiveCore()

# Pydantic models for request/response
class SqlRequest(BaseModel):
    query: str

class SqlResponse(BaseModel):
    sql: str

class DataRequest(BaseModel):
    sql: str

class DataResponse(BaseModel):
    response_type: str
    data: list
    columns: list
    row_count: int

class SqlExaminationRequest(BaseModel):
    sql: str

class SqlExaminationResponse(BaseModel):
    response_type: str
    sql: str
    explanation: dict

# Saved Query models
class SavedQueryRequest(BaseModel):
    guid: str
    name: str
    description: Optional[str] = None
    sqlText: str
    chartConfig: Optional[dict] = None
    isPublic: bool = False

class SavedQueryResponse(BaseModel):
    guid: str
    name: str
    description: Optional[str] = None
    sqlText: str
    chartConfig: Optional[dict] = None
    isPublic: bool
    createdAt: str
    updatedAt: str

class SavedQueryListResponse(BaseModel):
    queries: List[SavedQueryResponse]
    total: int

# Handle OPTIONS requests for CORS preflight
@app.options("/generate-sql")
async def options_generate_sql():
    """Handle CORS preflight requests for the generate-sql endpoint."""
    return {"message": "OK"}

@app.options("/execute-sql")
async def options_execute_sql():
    """Handle CORS preflight requests for the execute-sql endpoint."""
    return {"message": "OK"}

@app.options("/clear-cache")
async def options_clear_cache():
    """Handle CORS preflight requests for the clear-cache endpoint."""
    return {"message": "OK"}

@app.options("/validate-models")
async def options_validate_models():
    """Handle CORS preflight requests for the validate-models endpoint."""
    return {"message": "OK"}

@app.options("/refresh-embeddings")
async def options_refresh_embeddings():
    """Handle CORS preflight requests for the refresh-embeddings endpoint."""
    return {"message": "OK"}

@app.options("/list-models")
async def options_list_models():
    """Handle CORS preflight requests for the list-models endpoint."""
    return {"message": "OK"}

@app.options("/examine-sql")
async def options_examine_sql():
    """Handle CORS preflight requests for the examine-sql endpoint."""
    return {"message": "OK"}

@app.options("/save-query")
async def options_save_query():
    """Handle CORS preflight requests for the save-query endpoint."""
    return {"message": "OK"}

@app.options("/list-queries")
async def options_list_queries():
    """Handle CORS preflight requests for the list-queries endpoint."""
    return {"message": "OK"}

@app.options("/delete-query/{guid}")
async def options_delete_query(guid: str):
    """Handle CORS preflight requests for the delete-query endpoint."""
    return {"message": "OK"}

# Authentication endpoints
@app.post("/auth/google", response_model=Token)
async def google_auth(request: GoogleTokenRequest):
    """Verify Google ID token and return JWT tokens."""
    try:
        # Verify Google token
        user_info = verify_google_token(request.token)
        
        # Create or update user
        user = create_or_update_user(user_info)
        
        # Create JWT tokens
        access_token = create_access_token(data={"sub": user.id})
        refresh_token = create_refresh_token(data={"sub": user.id})
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Authentication failed: {str(e)}")

@app.get("/auth/me", response_model=User)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """Get current user information."""
    return current_user

@app.post("/auth/refresh", response_model=Token)
async def refresh_token_endpoint(request: Request):
    """Refresh access token using refresh token."""
    try:
        # Get refresh token from request body or query params
        body = await request.json()
        refresh_token = body.get("refresh_token")
        
        if not refresh_token:
            raise HTTPException(status_code=400, detail="Refresh token required")
        
        # Verify refresh token
        payload = verify_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        
        # Create new access token
        access_token = create_access_token(data={"sub": user_id})
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,  # Keep the same refresh token
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error refreshing token: {str(e)}")

@app.post("/auth/logout")
async def logout():
    """Logout endpoint (client-side token removal)."""
    return {"message": "Logged out successfully"}

# FastAPI endpoints
@app.post("/generate-sql", response_model=SqlResponse)
async def generate_sql_endpoint(request: SqlRequest, current_user: User = Depends(get_current_user_with_context)):
    """Generate SQL from a natural language query."""
    logger.info(f"SQL Generation Request - User: {current_user.email} | Query: {request.query[:100]}...")
    
    try:
        response = core.generate_sql(request.query)
        logger.info(f"SQL Generation Success - User: {current_user.email} | Response Type: {type(response)}")
        return SqlResponse(sql=response["sql"])
    except Exception as e:
        logger.error(f"SQL Generation Error - User: {current_user.email} | Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating SQL: {str(e)}")

@app.post("/execute-sql", response_model=DataResponse)
async def execute_sql_endpoint(request: DataRequest, current_user: User = Depends(get_current_user_with_context)):
    """Execute SQL statement against the database and return results."""
    logger.info(f"SQL Execution Request - User: {current_user.email} | SQL: {request.sql[:100]}...")
    
    try:
        result = core.execute_sql_query(request.sql)
        logger.info(f"SQL Execution Success - User: {current_user.email} | Rows: {result['row_count']}")
        return DataResponse(
            response_type="data",
            data=result["data"],
            columns=result["columns"],
            row_count=result["row_count"]
        )
    except ValueError as e:
        logger.error(f"SQL Execution Validation Error - User: {current_user.email} | Error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"SQL Execution Error - User: {current_user.email} | Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error executing SQL: {str(e)}")

@app.post("/clear-cache")
async def clear_cache_endpoint(current_user: User = Depends(get_current_user_with_context)):
    """Clear the models cache to force reloading of model files."""
    logger.info(f"Cache Clear Request - User: {current_user.email}")
    
    try:
        core.clear_models_cache()
        logger.info(f"Cache Clear Success - User: {current_user.email}")
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Cache Clear Error - User: {current_user.email} | Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")

@app.get("/validate-models")
async def validate_models_endpoint(current_user: User = Depends(get_current_user_with_context)):
    """Validate all models against the database by selecting all columns explicitly."""
    logger.info(f"Model Validation Request - User: {current_user.email}")
    
    try:
        validation_results = core.validate_models_against_database()
        
        # Calculate summary statistics
        total_models = len(validation_results)
        successful_validations = sum(1 for result in validation_results.values() if result.get("status") == "success")
        failed_validations = total_models - successful_validations
        
        # Count models with issues
        models_with_missing_columns = sum(1 for result in validation_results.values() 
                                         if result.get("status") == "success" and result.get("missing_columns"))
        models_with_extra_columns = sum(1 for result in validation_results.values() 
                                      if result.get("status") == "success" and result.get("extra_columns"))
        
        logger.info(f"Model Validation Success - User: {current_user.email} | "
                   f"Total: {total_models} | Success: {successful_validations} | Failed: {failed_validations}")
        
        return {
            "summary": {
                "total_models": total_models,
                "successful_validations": successful_validations,
                "failed_validations": failed_validations,
                "models_with_missing_columns": models_with_missing_columns,
                "models_with_extra_columns": models_with_extra_columns
            },
            "results": validation_results
        }
    except Exception as e:
        logger.error(f"Model Validation Error - User: {current_user.email} | Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error validating models: {str(e)}")

@app.post("/refresh-embeddings")
async def refresh_embeddings_endpoint(current_user: User = Depends(get_current_active_user)):
    """Refresh the semantic embeddings by deleting old ones and creating new ones."""
    try:
        result = core.refresh_embeddings()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error refreshing embeddings: {str(e)}")

@app.get("/list-models")
async def list_models_endpoint(current_user: User = Depends(get_current_active_user)):
    """Get all models in JSON format."""
    try:
        result = core.list_models()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")

@app.post("/examine-sql", response_model=SqlExaminationResponse)
async def examine_sql_endpoint(request: SqlExaminationRequest, current_user: User = Depends(get_current_active_user)):
    """Examine a SQL query and return statistics about tables, columns, and relationships."""
    try:
        result = core.examine_sql(request.sql)
        return SqlExaminationResponse(
            response_type=result.get("response_type", "sql_examination"),
            sql=result.get("sql", request.sql),
            explanation=result.get("explanation", {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error examining SQL: {str(e)}")

# Saved Query endpoints
@app.post("/save-query", response_model=SavedQueryResponse)
async def save_query_endpoint(request: SavedQueryRequest, current_user: User = Depends(get_current_user_with_context)):
    """Save or update a query for the authenticated user."""
    logger.info(f"Save Query Request - User: {current_user.email} | Query: {request.name}")
    
    try:
        # Convert request to payload dictionary
        payload = {
            "guid": request.guid,
            "name": request.name,
            "description": request.description,
            "sqlText": request.sqlText,
            "chartConfig": request.chartConfig,
            "isPublic": request.isPublic
        }
        
        logger.info(f"Payload created - Type: {type(payload)}")
        logger.info(f"ChartConfig type: {type(payload.get('chartConfig'))}")
        logger.info(f"ChartConfig value: {payload.get('chartConfig')}")
        
        query_manager = SavedQueryManager()
        result = query_manager.save_query(
            user_id=current_user.id,
            payload=payload
        )
        
        logger.info(f"Save Query Success - User: {current_user.email} | GUID: {request.guid}")
        logger.info(f"Result type: {type(result)}")
        logger.info(f"Result content: {result}")
        
        # Try to create the response object
        try:
            response = SavedQueryResponse(**result)
            logger.info(f"Response object created successfully")
            return response
        except Exception as response_error:
            logger.error(f"Error creating SavedQueryResponse: {str(response_error)}")
            logger.error(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            raise response_error
        
    except Exception as e:
        logger.error(f"Save Query Error - User: {current_user.email} | Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error saving query: {str(e)}")

@app.get("/list-queries", response_model=SavedQueryListResponse)
async def list_queries_endpoint(current_user: User = Depends(get_current_user_with_context)):
    """Get all saved queries for the authenticated user."""
    logger.info(f"List Queries Request - User: {current_user.email}")
    
    try:
        query_manager = SavedQueryManager()
        queries_data = query_manager.list_queries(current_user.id)
        
        # Convert to response format
        queries = [SavedQueryResponse(**query) for query in queries_data]
        
        logger.info(f"List Queries Success - User: {current_user.email} | Count: {len(queries)}")
        
        return SavedQueryListResponse(
            queries=queries,
            total=len(queries)
        )
        
    except Exception as e:
        logger.error(f"List Queries Error - User: {current_user.email} | Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing queries: {str(e)}")

@app.delete("/delete-query/{guid}")
async def delete_query_endpoint(guid: str, current_user: User = Depends(get_current_user_with_context)):
    """Delete a saved query by GUID for the authenticated user."""
    logger.info(f"Delete Query Request - User: {current_user.email} | GUID: {guid}")
    
    try:
        query_manager = SavedQueryManager()
        deleted = query_manager.delete_query(current_user.id, guid)
        
        if deleted:
            logger.info(f"Delete Query Success - User: {current_user.email} | GUID: {guid}")
            return {"message": "Query deleted successfully", "deleted": True}
        else:
            logger.info(f"Delete Query Not Found - User: {current_user.email} | GUID: {guid}")
            raise HTTPException(status_code=404, detail="Query not found")
        
    except HTTPException:
        # Re-raise HTTPExceptions (like 404) without wrapping
        raise
    except Exception as e:
        logger.error(f"Delete Query Error - User: {current_user.email} | GUID: {guid} | Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
