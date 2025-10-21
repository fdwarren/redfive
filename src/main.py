import os, yaml
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from rag import create_embeddings, retrieve_embeddings, build_graph, expand_with_graph, build_llm_context, generate_sql

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

# def create_embeddings():
#     connection_string = os.getenv("CONNECTION_STRING")
#     output_dir = Path(os.path.expanduser("./io/upstream/models"))
#     create_embeddings(connection_string, output_dir)

# Global cache for models
_models_cache = None
_cache_timestamp = None

def load_models():
    """
    Load models from YAML files with caching.
    Cache is invalidated when model files are modified.
    """
    global _models_cache, _cache_timestamp
    
    base_path = Path(os.path.expanduser("./io/models"))
    schema_names = [p.name for p in base_path.iterdir() if p.is_dir()]

    current_max_mtime = 0
    for schema_name in schema_names:
        models_dir = Path(os.path.expanduser(f"./io/models/{schema_name}"))

        for fname in os.listdir(models_dir):
            if fname.endswith('.yaml'):
                file_path = models_dir / fname
                current_max_mtime = max(current_max_mtime, file_path.stat().st_mtime)
        
    if _models_cache is not None and _cache_timestamp == current_max_mtime:
        print("Using cached models")
        return _models_cache
    
    # Load models from files
    print("Loading models from files (cache miss or files changed)")
    models = {}
    for schema_name in schema_names:
        models_dir = Path(os.path.expanduser(f"./io/models/{schema_name}"))
        for fname in os.listdir(models_dir):
            if fname.endswith('.yaml'):
                with open(models_dir / fname) as f:
                    model = yaml.safe_load(f)
                    model["name"] = fname.replace(".yaml", "")
                    model["schema_name"] = schema_name
                    model["path"] = f"{schema_name}.{model['name']}"
                    models[model["path"]] = model

                    for fk in model.get("keys", {}).get("foreign", []):
                        fk["ref_schema"] = fk.get('ref_schema', schema_name)
                        fk["ref_table_path"] = f"{fk['ref_schema']}.{fk['ref_table']}"
        
    _models_cache = models
    _cache_timestamp = current_max_mtime
        
    print(f"Loaded {len(models)} models for {len(schema_names)} schemas and updated cache")
    
    return models

def clear_models_cache():
    """
    Manually clear the models cache.
    Useful for forcing a reload of models when needed.
    """
    global _models_cache, _cache_timestamp
    _models_cache = None
    _cache_timestamp = None

def validate_models_against_database():
    """
    Validate all models against the database by selecting all columns explicitly.
    Returns a dictionary with validation results for each model.
    """
    models = load_models()
    db_connection_string = os.getenv("CONNECTION_STRING")
    
    if not db_connection_string:
        raise HTTPException(status_code=500, detail="Database connection string not configured")
    
    validation_results = {}
    
    try:
        engine = create_engine(db_connection_string)
        
        for model_name, model in models.items():
            schema_name = model.get("schema_name", "upstream")
            table_path = model["path"]
            
            # Get column names from the model
            model_columns = [col["name"] for col in model.get("columns", [])]
            
            # Create explicit column list for SELECT
            if model_columns:
                column_list = ", ".join([f'{col}' for col in model_columns])
                sql = f'SELECT {column_list} FROM {table_path} LIMIT 1'
            else:
                sql = f'SELECT * FROM {table_path} LIMIT 1'
            
            try:
                # Use a new connection for each model to avoid transaction issues
                with engine.connect() as conn:
                    # Set the schema search path
                    conn.execute(text("SET search_path TO redfive, upstream, public"))
                    
                    result = conn.execute(text(sql))
                    columns = list(result.keys())
                    
                    # Check if all model columns exist in the database
                    missing_columns = set(model_columns) - set(columns)
                    extra_columns = set(columns) - set(model_columns)
                    
                    validation_results[model_name] = {
                        "status": "success",
                        "schema": schema_name,
                        "table": table_path,
                        "model_columns": model_columns,
                        "db_columns": columns,
                        "missing_columns": list(missing_columns),
                        "extra_columns": list(extra_columns),
                        "column_count_match": len(model_columns) == len(columns),
                        "all_columns_exist": len(missing_columns) == 0
                    }
                    
            except Exception as e:
                validation_results[model_name] = {
                    "status": "error",
                    "schema": schema_name,
                    "table": table_path,
                    "error": str(e),
                    "sql": sql
                }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")
    
    return validation_results

def to_sql(user_query: str):
    models = load_models()

    db_connection_string = os.getenv("CONNECTION_STRING")

    results = retrieve_embeddings(db_connection_string, user_query, 5)
    graph = build_graph(list(models.values()))
    table_names = expand_with_graph(graph, results)
    context = build_llm_context(models, table_names)
    print("Context: ", context)
    return generate_sql(context, user_query)

def execute_sql_query(sql: str):
    """
    Execute SQL query against the database and return results.
    Only SELECT statements are allowed.
    """
    db_connection_string = os.getenv("CONNECTION_STRING")
    if not db_connection_string:
        raise HTTPException(status_code=500, detail="Database connection string not configured")
    
    # Check if SQL starts with SELECT (case insensitive)
    sql_stripped = sql.strip().upper()
    if not sql_stripped.startswith('SELECT'):
        raise HTTPException(
            status_code=400, 
            detail="Only SELECT statements are allowed. INSERT, UPDATE, DELETE operations are not permitted."
        )
    
    # Additional check for dangerous keywords
    dangerous_keywords = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE', 'EXEC', 'EXECUTE']
    for keyword in dangerous_keywords:
        if keyword in sql_stripped:
            raise HTTPException(
                status_code=400,
                detail=f"Operation '{keyword}' is not allowed. Only SELECT statements are permitted."
            )
    
    try:
        engine = create_engine(db_connection_string)
        with engine.connect() as conn:
            # Set the schema to 'upstream' after connection
            conn.execute(text("SET search_path TO redfive, upstream, public"))
            result = conn.execute(text(sql))
            columns = list(result.keys())
            rows = result.fetchall()
            
            data = []
            for row in rows:
                row_dict = {}
                for i, value in enumerate(row):
                    # Convert non-serializable types to strings
                    if hasattr(value, 'isoformat'):  # datetime objects
                        row_dict[columns[i]] = value.isoformat()
                    else:
                        row_dict[columns[i]] = value
                data.append(row_dict)
            
            return {
                "data": data,
                "columns": columns,
                "row_count": len(data)
            }
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"SQL execution error: {str(e)}")

# Handle OPTIONS requests for CORS preflight
@app.options("/generate-sql")
async def options_generate_sql():
    """
    Handle CORS preflight requests for the generate-sql endpoint.
    """
    return {"message": "OK"}

@app.options("/execute-sql")
async def options_execute_sql():
    """
    Handle CORS preflight requests for the execute-sql endpoint.
    """
    return {"message": "OK"}

@app.options("/clear-cache")
async def options_clear_cache():
    """
    Handle CORS preflight requests for the clear-cache endpoint.
    """
    return {"message": "OK"}

@app.options("/validate-models")
async def options_validate_models():
    """
    Handle CORS preflight requests for the validate-models endpoint.
    """
    return {"message": "OK"}

@app.options("/refresh-embeddings")
async def options_refresh_embeddings():
    """
    Handle CORS preflight requests for the refresh-embeddings endpoint.
    """
    return {"message": "OK"}

@app.options("/get-models")
async def options_get_models():
    """
    Handle CORS preflight requests for the get-models endpoint.
    """
    return {"message": "OK"}

# FastAPI endpoint
@app.post("/generate-sql", response_model=SqlResponse)
async def generate_sql_endpoint(request: SqlRequest):
    """
    Generate SQL from a natural language query.
    """
    try:
        sql = to_sql(request.query)
        return SqlResponse(sql=sql)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating SQL: {str(e)}")

@app.post("/execute-sql", response_model=DataResponse)
async def execute_sql_endpoint(request: DataRequest):
    """
    Execute SQL statement against the database and return results.
    """
    try:
        result = execute_sql_query(request.sql)
        return DataResponse(
            response_type="data",
            data=result["data"],
            columns=result["columns"],
            row_count=result["row_count"]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing SQL: {str(e)}")

@app.post("/clear-cache")
async def clear_cache_endpoint():
    """
    Clear the models cache to force reloading of model files.
    """
    try:
        clear_models_cache()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")

@app.get("/validate-models")
async def validate_models_endpoint():
    """
    Validate all models against the database by selecting all columns explicitly.
    Returns detailed validation results for each model.
    """
    try:
        validation_results = validate_models_against_database()
        
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
    """
    Refresh the semantic embeddings by deleting old ones and creating new ones.
    This will update the embeddings based on the current model definitions.
    """
    try:
        # Load current models
        models = load_models()
        
        # Get database connection string
        db_connection_string = os.getenv("CONNECTION_STRING")
        if not db_connection_string:
            raise HTTPException(status_code=500, detail="Database connection string not configured")
        
        # Create embeddings
        create_embeddings(db_connection_string, models)
        
        return {
            "message": "Embeddings refreshed successfully",
            "models_processed": len(models),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error refreshing embeddings: {str(e)}")

@app.get("/get-models")
async def get_models_endpoint():
    """
    Get all models in JSON format.
    Returns the current model definitions as a JSON array with their schemas, columns, and relationships.
    """
    try:
        models = load_models()
        
        # Convert dictionary to array of model values
        models_array = list(models.values())
        
        return {
            "models": models_array,
            "count": len(models_array),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)