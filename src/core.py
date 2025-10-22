"""
Core business logic for RedFive SQL Generator.
This module contains all the reusable functionality that can be used by both API and MCP server.
"""

import os
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from sqlalchemy import create_engine, text
from rag import create_embeddings, retrieve_embeddings, build_graph, expand_with_graph, build_llm_context, generate_sql


class RedFiveCore:
    """
    Core business logic class for RedFive SQL Generator.
    Contains all the reusable functionality for model management, SQL generation, and execution.
    """
    
    def __init__(self, connection_string: Optional[str] = None, models_path: Optional[str] = None):
        """
        Initialize the RedFive core with optional configuration.
        
        Args:
            connection_string: Database connection string. If None, will use CONNECTION_STRING env var.
            models_path: Path to models directory. If None, will use ./io/models.
        """
        self.connection_string = connection_string or os.getenv("CONNECTION_STRING")
        self.models_path = Path(models_path) if models_path else Path(os.path.expanduser("./io/models"))
        
        # Global cache for models
        self._models_cache = None
        self._cache_timestamp = None
    
    def load_models(self) -> Dict[str, Any]:
        """
        Load models from YAML files with caching.
        Cache is invalidated when model files are modified.
        
        Returns:
            Dictionary of models keyed by their path (schema.table)
        """
        schema_names = [p.name for p in self.models_path.iterdir() if p.is_dir()]

        current_max_mtime = 0
        for schema_name in schema_names:
            models_dir = self.models_path / schema_name
            for fname in models_dir.glob('*.yaml'):
                current_max_mtime = max(current_max_mtime, fname.stat().st_mtime)
        
        if self._models_cache is not None and self._cache_timestamp == current_max_mtime:
            print("Using cached models")
            return self._models_cache
        
        # Load models from files
        print("Loading models from files (cache miss or files changed)")
        models = {}
        for schema_name in schema_names:
            models_dir = self.models_path / schema_name
            for fname in models_dir.glob('*.yaml'):
                with open(fname) as f:
                    model = yaml.safe_load(f)
                    model["name"] = fname.stem
                    model["schema_name"] = schema_name
                    model["path"] = f"{schema_name}.{model['name']}"
                    models[model["path"]] = model

                    for fk in model.get("keys", {}).get("foreign", []):
                        fk["ref_schema"] = fk.get('ref_schema', schema_name)
                        fk["ref_table_path"] = f"{fk['ref_schema']}.{fk['ref_table']}"
        
        self._models_cache = models
        self._cache_timestamp = current_max_mtime
            
        print(f"Loaded {len(models)} models for {len(schema_names)} schemas and updated cache")
        
        return models

    def clear_models_cache(self):
        """
        Manually clear the models cache.
        Useful for forcing a reload of models when needed.
        """
        self._models_cache = None
        self._cache_timestamp = None

    def validate_models_against_database(self) -> Dict[str, Any]:
        """
        Validate all models against the database by selecting all columns explicitly.
        Returns a dictionary with validation results for each model.
        """
        models = self.load_models()
        
        if not self.connection_string:
            raise ValueError("Database connection string not configured")
        
        validation_results = {}
        
        try:
            engine = create_engine(self.connection_string)
            
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
            raise RuntimeError(f"Database connection error: {str(e)}")
        
        return validation_results

    def generate_sql(self, user_query: str) -> str:
        """
        Generate SQL from a natural language query.
        
        Args:
            user_query: Natural language query
            
        Returns:
            Generated SQL query
        """
        models = self.load_models()
        
        if not self.connection_string:
            raise ValueError("Database connection string not configured")

        results = retrieve_embeddings(self.connection_string, user_query, 5)
        graph = build_graph(list(models.values()))
        table_names = expand_with_graph(graph, results)
        context = build_llm_context(models, table_names)
        print("Context: ", context)
        return generate_sql(context, user_query)

    def execute_sql_query(self, sql: str) -> Dict[str, Any]:
        """
        Execute SQL query against the database and return results.
        Only SELECT statements are allowed.
        
        Args:
            sql: SQL query to execute
            
        Returns:
            Dictionary containing data, columns, and row count
        """
        if not self.connection_string:
            raise ValueError("Database connection string not configured")
        
        # Check if SQL starts with SELECT (case insensitive)
        sql_stripped = sql.strip().upper()
        if not sql_stripped.startswith('SELECT'):
            raise ValueError("Only SELECT statements are allowed. INSERT, UPDATE, DELETE operations are not permitted.")
        
        # Additional check for dangerous keywords
        dangerous_keywords = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE', 'EXEC', 'EXECUTE']
        for keyword in dangerous_keywords:
            if keyword in sql_stripped:
                raise ValueError(f"Operation '{keyword}' is not allowed. Only SELECT statements are permitted.")
        
        try:
            engine = create_engine(self.connection_string)
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
            raise RuntimeError(f"SQL execution error: {str(e)}")

    def refresh_embeddings(self) -> Dict[str, Any]:
        """
        Refresh the semantic embeddings by deleting old ones and creating new ones.
        This will update the embeddings based on the current model definitions.
        
        Returns:
            Dictionary with refresh results
        """
        # Load current models
        models = self.load_models()
        
        if not self.connection_string:
            raise ValueError("Database connection string not configured")
        
        # Create embeddings
        create_embeddings(self.connection_string, models)
        
        return {
            "message": "Embeddings refreshed successfully",
            "models_processed": len(models),
            "timestamp": datetime.now().isoformat()
        }

    def get_models(self) -> Dict[str, Any]:
        """
        Get all models in JSON format.
        Returns the current model definitions as a JSON array with their schemas, columns, and relationships.
        
        Returns:
            Dictionary containing models array and metadata
        """
        models = self.load_models()
        
        # Convert dictionary to array of model values
        models_array = list(models.values())
        
        return {
            "models": models_array,
            "count": len(models_array),
            "timestamp": datetime.now().isoformat()
        }
