import logging
from typing import List
from pydantic import BaseModel
import sqlparse, json
import networkx as nx
from openai import OpenAI
from sqlalchemy import create_engine, Table, MetaData, text
from sqlalchemy.orm import sessionmaker
from openai import OpenAI


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Create logger instance
logger = logging.getLogger(__name__)


class HistoryItem(BaseModel):
    user_prompt: str
    system_response: str

def create_embeddings(db_connection_string: str, models: dict):
    client = OpenAI()
    engine = create_engine(db_connection_string)
    metadata = MetaData()
    semantic_embeddings = Table("semantic_embeddings", metadata, schema="redfive", autoload_with=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Delete existing embeddings before inserting new ones
    print("Deleting existing embeddings...")
    session.execute(semantic_embeddings.delete())
    print("Existing embeddings marked for deletion")

    def flatten_schema(model, table_name):
        yield f"Table {table_name}: {model.get('description','')}"
        for col in model.get("columns", []):
            yield f"Column {col['name']} ({col['type']}) in {table_name}: {col.get('description','')}"

    print("Creating new embeddings...")
    embedding_count = 0
    
    for model in models.values():
        table_path = model["path"]
        for text in flatten_schema(model, table_path):
            emb = client.embeddings.create(
                model="text-embedding-3-small", input=text
            ).data[0].embedding
            session.execute(
                semantic_embeddings.insert().values(
                    entity_type="column" if text.startswith("Column") else "table",
                    entity_name=table_path if text.startswith("Table") else text.split()[1],
                    parent_table=None if text.startswith("Table") else table_path,
                    content=text,
                    embedding=emb
                )
            )
            embedding_count += 1

    session.commit()
    session.close()
    print(f"Successfully created {embedding_count} new embeddings")


def retrieve_embeddings(db_connection_string: str, query: str, limit: int = 5):
    client = OpenAI()
    q_emb = client.embeddings.create(model="text-embedding-3-small", input=query).data[0].embedding

    engine = create_engine(db_connection_string)
    with engine.connect() as conn:
        # Set the schema to 'upstream' after connection
        conn.execute(text("SET search_path TO redfive, upstream, public"))
        cursor = conn.connection.cursor()
        cursor.execute(f"""
            SELECT entity_name
            FROM redfive.semantic_embeddings
            where entity_type = 'table'
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (q_emb, limit))

        results = []
        for r in cursor.fetchall():
            results.append(r[0])

        return results

def build_graph(models):
    G = nx.DiGraph()
    for model in models:
        table = model.get("path")
        fks = (model.get("keys") or {}).get("foreign", [])
        for fk in fks:
            target = fk["ref_table_path"]
            G.add_edge(table, target)
    return G


def expand_with_graph(graph, requested_tables, max_depth=5):
    expanded = set(requested_tables)
    for table in requested_tables:
        for target in nx.single_source_shortest_path_length(graph, table, cutoff=max_depth):
            expanded.add(target)

    return list(expanded)

def build_llm_context(models, tables):
    lines = []
    for t in tables:
        print("Building context for table: ", t)
        m = models[t]
        cols = ", ".join(c["name"] for c in m.get("columns", []))
        lines.append(f"Table {t} ({m.get('description','')}): {cols}")
        fks = m.get("keys", {}).get("foreign", [])
        for fk in fks:
            ref = fk["ref_table_path"]
            if ref in tables:
                src = fk["columns"]
                tgt = fk["ref_columns"]
                lines.append(f"  FK: {t}.{src} -> {ref}.{tgt}")
                
    return "\n".join(lines)

def generate_sql(context: str, sql_generation_schema: str, user_query: str, history: List[HistoryItem]):
    prompt = f"""
        You are an expert SQL generation assistant. You will receive:
        1. A user request in natural language.
        2. Context retrieved from a semantic model (tables, columns, relationships).
        3. The history of the conversation.

        Your task:
        - Generate a **valid SQL query** for the request.
        - Provide a **structured summary** of which columns were used in the query.

        Use **only** the data structures and relationships in the provided schema:

        {context}

        Respond **only** in JSON with this schema:

        {sql_generation_schema}

        Here is the history of the conversation:
        {"\n".join([f"User: {item.user_prompt}\nSystem: {item.system_response}" for item in history])}

        Generate a single SQL query answering:
        '{user_query}'

        Rules:
        - Use only tables and columns from the schema.
        - Do not invent column names.
        - Do not invent table names.
        - The user request should be mapped to specific tables and columns in the schema.
        - Use case insensitive matching for user provided query parameters.
        - Use explicit JOIN conditions where foreign keys exist.
        - Use postgres compatible syntax.
        - Only use the columns and tables provided in the schema.
        - Return only the SQL query, no other text.
        """

    logger.info(f"Prompt: {prompt}")

    client = OpenAI()

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = resp.choices[0].message.content
    
    # Remove markdown code block formatting if present
    if raw.startswith("```json"):
        raw = raw[7:]  # Remove ```sql
    if raw.startswith("```"):
        raw = raw[3:]   # Remove ```
    if raw.endswith("```"):
        raw = raw[:-3]  # Remove trailing ```
    # Strip any leading/trailing whitespace
    raw = raw.strip()
    
    return sqlparse.format(raw, reindent=True, keyword_case="upper")
