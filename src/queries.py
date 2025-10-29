"""
Database operations for saved queries.
Handles saving, retrieving, and managing user queries.
"""

import json
import os
from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import JSONB
from logging_config import logger


class SavedQueryManager:
    """Manages saved queries in the database."""
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize the SavedQueryManager.
        
        Args:
            connection_string: Database connection string. If None, will use DATABASE_CONNECTION_STRING env var.
        """
        # Try to use a write-enabled connection first, fall back to readonly
        self.connection_string = os.getenv("WRITE_CONNECTION_STRING")
        if not self.connection_string:
            raise ValueError("Database connection string not configured. Please set WRITE_CONNECTION_STRING environment variable.")
        
        self.engine = create_engine(self.connection_string)
    
    def save_query(self, user_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save or update a query for a user.
        
        Args:
            user_id: The user's Google ID
            payload: Complete query payload as dictionary
            
        Returns:
            Dictionary containing the saved query data with timestamps
            
        Raises:
            Exception: If database operation fails
        """
        try:
            guid = payload["guid"]
            
            # Debug: Check what we're working with
            logger.info(f"Payload type: {type(payload)}")
            logger.info(f"Payload content: {payload}")
            
            # Serialize the payload to JSON string for psycopg2
            try:
                payload_json = json.dumps(payload)
                logger.info(f"JSON serialized successfully, type: {type(payload_json)}")
            except Exception as json_error:
                logger.error(f"JSON serialization failed: {str(json_error)}")
                logger.error(f"Payload causing error: {payload}")
                raise json_error
            
            # Upsert the query
            with self.engine.connect() as conn:
                # Start a transaction
                trans = conn.begin()
                try:
                    logger.info(f"Executing database insert/update for GUID: {guid}")
                    result = conn.execute(text("""
                        INSERT INTO saved_queries (guid, user_id, payload)
                        VALUES (:guid, :user_id, :payload)
                        ON CONFLICT (guid) 
                        DO UPDATE SET 
                            user_id = EXCLUDED.user_id,
                            payload = EXCLUDED.payload,
                            updated_at = CURRENT_TIMESTAMP
                        RETURNING guid, payload, created_at, updated_at
                    """), {
                        "guid": guid,
                        "user_id": user_id,
                        "payload": payload_json
                    })
                    
                    row = result.fetchone()
                    if not row:
                        raise Exception("Failed to save query - no row returned")
                    
                    logger.info(f"Database operation successful - Row returned: {row is not None}")
                    logger.info(f"Returned GUID: {row.guid if row else 'None'}")
                    
                    # The payload is already a dictionary from PostgreSQL JSONB
                    saved_payload = row.payload.copy()  # Make a copy to avoid modifying the original
                    saved_payload["createdAt"] = row.created_at.isoformat()
                    saved_payload["updatedAt"] = row.updated_at.isoformat()
                    
                    logger.info(f"Query saved successfully - User: {user_id} | GUID: {guid} | Name: {saved_payload.get('name', 'Unknown')}")
                    logger.info(f"Saved payload type: {type(saved_payload)}")
                    logger.info(f"ChartConfig type: {type(saved_payload.get('chartConfig'))}")
                    
                    # Verify the data was actually saved by querying it back
                    verify_result = conn.execute(text("""
                        SELECT guid, payload FROM saved_queries 
                        WHERE guid = :guid AND user_id = :user_id
                    """), {"guid": guid, "user_id": user_id})
                    
                    verify_row = verify_result.fetchone()
                    if verify_row:
                        logger.info(f"Verification successful - Query found in database with GUID: {verify_row.guid}")
                    else:
                        logger.error(f"Verification failed - Query not found in database for GUID: {guid}")
                    
                    # Commit the transaction
                    trans.commit()
                    logger.info("Transaction committed successfully")
                    
                    return saved_payload
                    
                except Exception as e:
                    # Rollback the transaction on error
                    trans.rollback()
                    logger.error(f"Transaction rolled back due to error: {str(e)}")
                    raise e
                
        except Exception as e:
            logger.error(f"Error saving query - User: {user_id} | GUID: {payload.get('guid', 'Unknown')} | Error: {str(e)}", exc_info=True)
            raise Exception(f"Failed to save query: {str(e)}")
    
    def list_queries(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all saved queries for a user.
        
        Args:
            user_id: The user's Google ID
            
        Returns:
            List of dictionaries containing query data with timestamps
            
        Raises:
            Exception: If database operation fails
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT guid, payload, created_at, updated_at
                    FROM saved_queries 
                    WHERE user_id = :user_id
                    ORDER BY updated_at DESC
                """), {"user_id": user_id})
                
                rows = result.fetchall()
                queries = []
                
                for row in rows:
                    payload = row.payload.copy()  # Already a dictionary from JSONB
                    # Add timestamps to the payload
                    payload["createdAt"] = row.created_at.isoformat()
                    payload["updatedAt"] = row.updated_at.isoformat()
                    queries.append(payload)
                
                logger.info(f"Queries retrieved successfully - User: {user_id} | Count: {len(queries)}")
                
                return queries
                
        except Exception as e:
            logger.error(f"Error listing queries - User: {user_id} | Error: {str(e)}", exc_info=True)
            raise Exception(f"Failed to list queries: {str(e)}")
    
    def get_query(self, user_id: str, guid: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific query by GUID for a user.
        
        Args:
            user_id: The user's Google ID
            guid: The query GUID
            
        Returns:
            Dictionary containing query data with timestamps or None if not found
            
        Raises:
            Exception: If database operation fails
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT guid, payload, created_at, updated_at
                    FROM saved_queries 
                    WHERE user_id = :user_id AND guid = :guid
                """), {"user_id": user_id, "guid": guid})
                
                row = result.fetchone()
                if not row:
                    return None
                
                payload = row.payload.copy()  # Already a dictionary from JSONB
                # Add timestamps to the payload
                payload["createdAt"] = row.created_at.isoformat()
                payload["updatedAt"] = row.updated_at.isoformat()
                
                logger.info(f"Query retrieved successfully - User: {user_id} | GUID: {guid}")
                
                return payload
                
        except Exception as e:
            logger.error(f"Error getting query - User: {user_id} | GUID: {guid} | Error: {str(e)}", exc_info=True)
            raise Exception(f"Failed to get query: {str(e)}")
    
    def delete_query(self, user_id: str, guid: str) -> bool:
        """
        Delete a query by GUID for a user.
        
        Args:
            user_id: The user's Google ID
            guid: The query GUID
            
        Returns:
            True if deleted, False if not found
            
        Raises:
            Exception: If database operation fails
        """
        try:
            with self.engine.connect() as conn:
                # Start a transaction
                trans = conn.begin()
                try:
                    result = conn.execute(text("""
                        DELETE FROM saved_queries 
                        WHERE user_id = :user_id AND guid = :guid
                    """), {"user_id": user_id, "guid": guid})
                    
                    deleted = result.rowcount > 0
                    
                    if deleted:
                        logger.info(f"Query deleted successfully - User: {user_id} | GUID: {guid}")
                    else:
                        logger.info(f"Query not found for deletion - User: {user_id} | GUID: {guid}")
                    
                    # Commit the transaction
                    trans.commit()
                    logger.info(f"Delete transaction committed successfully - User: {user_id} | GUID: {guid}")
                    
                    return deleted
                    
                except Exception as e:
                    # Rollback the transaction on error
                    trans.rollback()
                    logger.error(f"Delete transaction rolled back due to error: {str(e)}")
                    raise e
                
        except Exception as e:
            logger.error(f"Error deleting query - User: {user_id} | GUID: {guid} | Error: {str(e)}", exc_info=True)
            raise Exception(f"Failed to delete query: {str(e)}")
