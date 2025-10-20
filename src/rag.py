import sqlparse, json
import networkx as nx
from openai import OpenAI
from sqlalchemy import create_engine, Table, MetaData, text
from sqlalchemy.orm import sessionmaker
from openai import OpenAI

def create_embeddings(db_connection_string: str, models_dir: str):
    client = OpenAI()
    engine = create_engine(db_connection_string)
    metadata = MetaData()
    semantic_embeddings = Table("semantic_embeddings", metadata, schema="redfive", autoload_with=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    def flatten_schema(model, table_name):
        yield f"Table {table_name}: {model.get('description','')}"
        for col in model.get("columns", []):
            yield f"Column {col['name']} ({col['type']}) in {table_name}: {col.get('description','')}"

    # for fname in os.listdir(models_dir):
    #     if not fname.endswith(".yaml"):
    #         continue
    #     with open(os.path.join(models_dir, fname)) as f:
    #         model = yaml.safe_load(f)
    #     table_name = os.path.splitext(fname)[0]
    #     for text in flatten_schema(model, table_name):
    #         emb = client.embeddings.create(
    #             model="text-embedding-3-small", input=text
    #         ).data[0].embedding
    #         session.execute(
    #             semantic_embeddings.insert().values(
    #                 entity_type="column" if text.startswith("Column") else "table",
    #                 entity_name=table_name if text.startswith("Table") else text.split()[1],
    #                 parent_table=None if text.startswith("Table") else table_name,
    #                 content=text,
    #                 embedding=emb
    #             )
    #         )

    session.commit()
    session.close()


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
        table = model.get("name")
        fks = (model.get("keys") or {}).get("foreign", [])
        for fk in fks:
            target = fk["ref_table"]
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
        m = models[t]
        cols = ", ".join(c["name"] for c in m.get("columns", []))
        lines.append(f"Table {t} ({m.get('description','')}): {cols}")
        fks = m.get("keys", {}).get("foreign", [])
        for fk in fks:
            ref = fk["ref_table"]
            if ref in tables:
                src = fk["columns"]
                tgt = fk["ref_columns"]
                lines.append(f"  FK: {t}.{src} -> {ref}.{tgt}")
    return "\n".join(lines)

def generate_sql(context: str, user_query: str):
    prompt = f"""
        You are a SQL generator. Use only the data structures and relationships in the schema below.

        {context}

        Generate a single SQL query answering:
        '{user_query}'

        Rules:
        - Use only tables and columns from the schema.
        - Do not invent column names.
        - Do not invent table names.
        - Verify the tables exist in the schema.
        - Verify the column names exist in the schema.
        - The user request should be mapped to specific tables and columns in the schema.
        - Use explicit JOIN conditions where foreign keys exist.
        - Return only the SQL query, no other text.
        """

    client = OpenAI()

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = resp.choices[0].message.content
    
    # Remove markdown code block formatting if present
    if raw.startswith("```sql"):
        raw = raw[6:]  # Remove ```sql
    if raw.startswith("```"):
        raw = raw[3:]   # Remove ```
    if raw.endswith("```"):
        raw = raw[:-3]  # Remove trailing ```
    
    # Strip any leading/trailing whitespace
    raw = raw.strip()
    return sqlparse.format(raw, reindent=True, keyword_case="upper")
