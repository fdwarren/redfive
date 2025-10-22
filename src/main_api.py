"""
FastAPI server for RedFive SQL Generator.
This module provides HTTP endpoints for the SQL generation functionality.
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from core import RedFiveCore

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

# FastAPI endpoints
@app.post("/generate-sql", response_model=SqlResponse)
async def generate_sql_endpoint(request: SqlRequest):
    """Generate SQL from a natural language query."""
    try:
        sql = core.generate_sql(request.query)
        return SqlResponse(sql=sql)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating SQL: {str(e)}")

@app.post("/execute-sql", response_model=DataResponse)
async def execute_sql_endpoint(request: DataRequest):
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
async def clear_cache_endpoint():
    """Clear the models cache to force reloading of model files."""
    try:
        core.clear_models_cache()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")

@app.get("/validate-models")
async def validate_models_endpoint():
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
async def refresh_embeddings_endpoint():
    """Refresh the semantic embeddings by deleting old ones and creating new ones."""
    try:
        result = core.refresh_embeddings()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error refreshing embeddings: {str(e)}")

@app.get("/get-models")
async def get_models_endpoint():
    """Get all models in JSON format."""
    try:
        result = core.get_models()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
