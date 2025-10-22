"""
MCP (Model Context Protocol) server for RedFive SQL Generator.
This module provides MCP server functionality for the SQL generation capabilities.
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    ListToolsRequest,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)
from core import RedFiveCore

load_dotenv()

class RedFiveMCPServer:
    """
    MCP server implementation for RedFive SQL Generator.
    Provides tools for SQL generation through MCP protocol.
    """
    
    def __init__(self):
        """Initialize the MCP server with core functionality."""
        self.core = RedFiveCore()
        self.server = Server("redfive-sql-generator")
        
        # Register tools
        self.server.list_tools = self.list_tools
        self.server.call_tool = self.call_tool
    
    async def list_tools(self) -> List[Tool]:
        """Return list of available tools."""
        return [
            Tool(
                name="generate_sql",
                description="Generate SQL from a natural language query",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query to convert to SQL"
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="execute_sql",
                description="Execute SQL query against the database and return results",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "SQL query to execute (SELECT statements only)"
                        }
                    },
                    "required": ["sql"]
                }
            ),
            Tool(
                name="validate_models",
                description="Validate all models against the database",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            Tool(
                name="get_models",
                description="Get all available models",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            Tool(
                name="refresh_embeddings",
                description="Refresh semantic embeddings",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            Tool(
                name="clear_cache",
                description="Clear the models cache",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            )
        ]
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle tool call requests."""
        try:
            if name == "generate_sql":
                query = arguments.get("query", "")
                if not query:
                    raise ValueError("Query parameter is required")
                
                sql = self.core.generate_sql(query)
                return [TextContent(
                    type="text",
                    text=f"Generated SQL:\n```sql\n{sql}\n```"
                )]
            
            elif name == "execute_sql":
                sql = arguments.get("sql", "")
                if not sql:
                    raise ValueError("SQL parameter is required")
                
                result = self.core.execute_sql_query(sql)
                return [TextContent(
                    type="text",
                    text=f"Query executed successfully.\nRows: {result['row_count']}\nColumns: {', '.join(result['columns'])}\n\nData:\n```json\n{json.dumps(result['data'][:10], indent=2)}\n```"
                )]
            
            elif name == "validate_models":
                validation_results = self.core.validate_models_against_database()
                
                # Calculate summary
                total_models = len(validation_results)
                successful = sum(1 for r in validation_results.values() if r.get("status") == "success")
                failed = total_models - successful
                
                summary = f"Validation Results:\n- Total models: {total_models}\n- Successful: {successful}\n- Failed: {failed}\n\n"
                
                details = []
                for model_name, result in validation_results.items():
                    if result.get("status") == "success":
                        details.append(f"✅ {model_name}: {result['table']}")
                    else:
                        details.append(f"❌ {model_name}: {result.get('error', 'Unknown error')}")
                
                return [TextContent(
                    type="text",
                    text=summary + "\n".join(details)
                )]
            
            elif name == "get_models":
                models_result = self.core.get_models()
                models = models_result["models"]
                
                model_list = []
                for model in models:
                    model_list.append(f"- {model['path']}: {model.get('description', 'No description')}")
                
                return [TextContent(
                    type="text",
                    text=f"Available Models ({len(models)}):\n" + "\n".join(model_list)
                )]
            
            elif name == "refresh_embeddings":
                result = self.core.refresh_embeddings()
                return [TextContent(
                    type="text",
                    text=f"Embeddings refreshed successfully.\nModels processed: {result['models_processed']}\nTimestamp: {result['timestamp']}"
                )]
            
            elif name == "clear_cache":
                self.core.clear_models_cache()
                return [TextContent(
                    type="text",
                    text="Models cache cleared successfully."
                )]
            
            else:
                raise ValueError(f"Unknown tool: {name}")
        
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error executing {name}: {str(e)}"
            )]
    
    async def run_server(self):
        """Run the MCP server."""
        print("RedFive MCP Server starting...")
        print("Available tools:")
        tools = await self.list_tools()
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
        
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="redfive-sql-generator",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=None,
                        experimental_capabilities=None,
                    ),
                ),
            )

async def main():
    """Main entry point for the MCP server."""
    server = RedFiveMCPServer()
    await server.run_server()

if __name__ == "__main__":
    asyncio.run(main())
