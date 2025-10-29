from typing import Optional
from pydantic import BaseModel


class HistoryItem(BaseModel):
    user_prompt: str
    system_response: str

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
