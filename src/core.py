"""
Core business logic for RedFive SQL Generator.
This module contains all the reusable functionality that can be used by both API and MCP server.
"""

import json
import os
import traceback
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

    def generate_sql(self, user_query: str) -> Dict[str, Any]:
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

        with open("./resources/schemas/sql-generation.schema.json", "r") as f:
            sql_generation_schema = f.read()

        response = generate_sql(context, sql_generation_schema, user_query)

        print("Response: ", response)
        return json.loads(response)


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

    def examine_sql(self, sql: str) -> Dict[str, Any]:
        """
        Examine a SQL query and return statistics about tables, columns, and relationships.
        
        Args:
            sql: SQL query to examine
            
        Returns:
            Dictionary containing SQL examination results
        """
        models = self.load_models()
        
        # Use LLM to analyze the SQL query
        context = self._build_sql_analysis_context(models, sql)
        print("Building context for SQL analysis...")
        analysis = self._analyze_sql_with_llm(context, sql)
        print("SQL analysis completed.")
        return analysis

    def _build_sql_analysis_context(self, models: Dict[str, Any], sql: str) -> str:
        """Build context for SQL analysis by finding relevant models."""
        # Extract table names from SQL (simple regex approach)
        import re
        table_pattern = r'FROM\s+([a-zA-Z_][a-zA-Z0-9_.]*)|\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_.]*)'
        matches = re.findall(table_pattern, sql, re.IGNORECASE)
        
        # Flatten matches and clean up
        table_names = []
        for match in matches:
            for name in match:
                if name:
                    # Remove schema prefix if present
                    clean_name = name.split('.')[-1]
                    table_names.append(clean_name)
        
        # Find matching models
        relevant_models = []
        for model_name, model in models.items():
            table_name = model.get('name', '')
            if table_name in table_names or any(table_name in tn for tn in table_names):
                relevant_models.append(model)
        
        # Build context string
        context_lines = []
        for model in relevant_models:
            table_path = model["path"]
            cols = ", ".join(c["name"] for c in model.get("columns", []))
            context_lines.append(f"Table {table_path} ({model.get('description','')}): {cols}")
            
            # Add foreign key relationships
            fks = model.get("keys", {}).get("foreign", [])
            for fk in fks:
                ref = fk["ref_table_path"]
                src = fk["columns"]
                tgt = fk["ref_columns"]
                context_lines.append(f"  FK: {table_path}.{src} -> {ref}.{tgt}")
        
        return "\n".join(context_lines)

    def _analyze_sql_with_llm(self, context: str, sql: str) -> Dict[str, Any]:
        """Use LLM to analyze SQL and extract statistics."""
        from openai import OpenAI
        import json
        import re
        
        try:
            # Simplified prompt without schema file dependency
            prompt = f"""
            Analyze the following SQL query and provide detailed statistics about its structure.

            Available schema context:
            {context}

            SQL Query to analyze:
            {sql}

            Please provide a JSON response with this exact structure:
            {{
                "response_type": "sql_examination",
                "sql": "{sql}",
                "explanation": {{
                    "tables": ["list", "of", "table", "names"],
                    "columns": [
                        {{
                            "name": "column_name",
                            "label": "display_label",
                            "table_name": "source_table",
                            "type": "string|integer|float|date|timestamp|guid",
                            "nullable": true
                        }}
                    ],
                    "relationships": [
                        {{
                            "from_table": "source_table",
                            "to_table": "target_table",
                            "join_type": "inner|left|right|full|cross",
                            "condition": "join condition"
                        }}
                    ]
                }}
            }}

            Rules:
            - Extract all table names referenced in the query
            - Identify all columns that will be returned
            - Determine join relationships between tables
            - Use only the table and column names from the provided schema context
            - Return ONLY valid JSON, no other text, no markdown formatting
            """

            client = OpenAI()
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            
            raw_response = response.choices[0].message.content.strip()
            print(f"Raw LLM response: {raw_response[:200]}...")
            
            # Clean up the response - remove markdown code blocks if present
            if raw_response.startswith("```json"):
                raw_response = raw_response[7:]  # Remove ```json
            elif raw_response.startswith("```"):
                raw_response = raw_response[3:]   # Remove ```
            if raw_response.endswith("```"):
                raw_response = raw_response[:-3]  # Remove trailing ```
            
            raw_response = raw_response.strip()
            print(f"Cleaned response: {raw_response[:200]}...")
            
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                print(f"Extracted JSON: {json_str[:200]}...")
                analysis = json.loads(json_str)
            else:
                # Try parsing the whole response
                analysis = json.loads(raw_response)
            
            print("Successfully parsed LLM response")
            return analysis
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Raw response was: {raw_response}")
            
            # Try to extract basic information manually
            return self._extract_basic_sql_info(sql, context)
            
        except Exception as e:
            print("Error in _analyze_sql_with_llm: ", e)
            traceback.print_exc()
            
            # Try to extract basic information manually
            return self._extract_basic_sql_info(sql, context)

    def _extract_basic_sql_info(self, sql: str, context: str) -> Dict[str, Any]:
        """Extract basic SQL information without LLM."""
        import re
        
        # Extract table names from SQL
        table_pattern = r'FROM\s+([a-zA-Z_][a-zA-Z0-9_.]*)|\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_.]*)'
        matches = re.findall(table_pattern, sql, re.IGNORECASE)
        
        tables = []
        for match in matches:
            for name in match:
                if name:
                    clean_name = name.split('.')[-1]
                    if clean_name not in tables:
                        tables.append(clean_name)
        
        # Extract column names from SELECT
        select_pattern = r'SELECT\s+(.*?)\s+FROM'
        select_match = re.search(select_pattern, sql, re.IGNORECASE | re.DOTALL)
        columns = []
        
        if select_match:
            select_clause = select_match.group(1)
            if '*' not in select_clause:
                # Extract individual columns
                col_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)'
                col_matches = re.findall(col_pattern, select_clause)
                for col in col_matches:
                    if col.upper() not in ['SELECT', 'DISTINCT', 'TOP']:
                        columns.append({
                            "name": col,
                            "label": col,
                            "table_name": tables[0] if tables else "unknown",
                            "type": "string",
                            "nullable": True
                        })
        
        return {
            "response_type": "sql_examination",
            "sql": sql,
            "explanation": {
                "tables": tables,
                "columns": columns,
                "relationships": []
            }
        }