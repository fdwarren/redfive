import os, yaml
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from rag import retrieve_embeddings, build_graph, expand_with_graph, build_llm_context, generate_sql

load_dotenv()

# Create FastAPI app
app = FastAPI(title="RedFive SQL Generator", description="Generate SQL from natural language queries")

# Add CORS middleware
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
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response_type: str
    text: str

class SqlRequest(BaseModel):
    sql: str

class SqlResponse(BaseModel):
    response_type: str
    data: list
    columns: list
    row_count: int

# def create_embeddings():
#     connection_string = os.getenv("CONNECTION_STRING")
#     output_dir = Path(os.path.expanduser("./io/upstream/models"))
#     create_embeddings(connection_string, output_dir)

def to_sql(user_query: str):
    models = {}
    models_dir = Path(os.path.expanduser("./io/upstream/models"))
    for fname in os.listdir(models_dir):
        with open(os.path.join(models_dir, fname)) as f:
            model = yaml.safe_load(f)
            model["name"] = fname.replace(".yaml", "")
            models[model["name"]] = model

    db_connection_string = os.getenv("CONNECTION_STRING")

    results = retrieve_embeddings(db_connection_string, user_query, 5)
    graph = build_graph(list(models.values()))
    table_names = expand_with_graph(graph, results)

    context = build_llm_context(models, table_names)

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
            
            # Get column names
            columns = list(result.keys())
            
            # Fetch all rows
            rows = result.fetchall()
            
            # Convert rows to list of dictionaries
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

# FastAPI endpoint
@app.post("/generate-sql", response_model=QueryResponse)
async def generate_sql_endpoint(request: QueryRequest):
    """
    Generate SQL from a natural language query.
    """
    try:
        sql = to_sql(request.query)
        return QueryResponse(response_type="sql", text=sql)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating SQL: {str(e)}")

@app.post("/execute-sql", response_model=SqlResponse)
async def execute_sql_endpoint(request: SqlRequest):
    """
    Execute SQL statement against the database and return results.
    """
    try:
        result = execute_sql_query(request.sql)
        return SqlResponse(
            response_type="data",
            data=result["data"],
            columns=result["columns"],
            row_count=result["row_count"]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing SQL: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)