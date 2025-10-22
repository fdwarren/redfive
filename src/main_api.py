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
    create_access_token, 
    create_refresh_token,
    create_or_update_user,
    verify_google_token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    Token,
    User,
    GoogleTokenRequest
)

load_dotenv()

app = FastAPI(title="RedFive SQL Generator", description="Generate SQL from natural language queries")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default port
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.options("/get-models")
async def options_get_models():
    """Handle CORS preflight requests for the get-models endpoint."""
    return {"message": "OK"}

@app.options("/examine-sql")
async def options_examine_sql():
    """Handle CORS preflight requests for the examine-sql endpoint."""
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
async def generate_sql_endpoint(request: SqlRequest, current_user: User = Depends(get_current_active_user)):
    """Generate SQL from a natural language query."""
    try:
        response = core.generate_sql(request.query)
        print("Response: ", type(response))
        return SqlResponse(sql=response["sql"])
    except Exception as e:
        print("Error generating SQL: ", e)
        raise HTTPException(status_code=500, detail=f"Error generating SQL: {str(e)}")

@app.post("/execute-sql", response_model=DataResponse)
async def execute_sql_endpoint(request: DataRequest, current_user: User = Depends(get_current_active_user)):
    """Execute SQL statement against the database and return results."""
    try:
        result = core.execute_sql_query(request.sql)
        return DataResponse(
            response_type="data",
            data=result["data"],
            columns=result["columns"],
            row_count=result["row_count"]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing SQL: {str(e)}")

@app.post("/clear-cache")
async def clear_cache_endpoint(current_user: User = Depends(get_current_active_user)):
    """Clear the models cache to force reloading of model files."""
    try:
        core.clear_models_cache()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")

@app.get("/validate-models")
async def validate_models_endpoint(current_user: User = Depends(get_current_active_user)):
    """Validate all models against the database by selecting all columns explicitly."""
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
        raise HTTPException(status_code=500, detail=f"Error validating models: {str(e)}")

@app.post("/refresh-embeddings")
async def refresh_embeddings_endpoint(current_user: User = Depends(get_current_active_user)):
    """Refresh the semantic embeddings by deleting old ones and creating new ones."""
    try:
        result = core.refresh_embeddings()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error refreshing embeddings: {str(e)}")

@app.get("/get-models")
async def get_models_endpoint(current_user: User = Depends(get_current_active_user)):
    """Get all models in JSON format."""
    try:
        result = core.get_models()
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
