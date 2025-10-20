import yaml
import os
from openai import OpenAI
from sqlalchemy import create_engine, Table, MetaData
from sqlalchemy.orm import sessionmaker
import networkx as nx

def create_embeddings(models_dir: str):
    db_connection_string = os.getenv("CONNECTION_STRING")

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

    for fname in os.listdir(models_dir):
        if not fname.endswith(".yaml"):
            continue
        with open(os.path.join(models_dir, fname)) as f:
            model = yaml.safe_load(f)
        table_name = os.path.splitext(fname)[0]
        for text in flatten_schema(model, table_name):
            emb = client.embeddings.create(
                model="text-embedding-3-small", input=text
            ).data[0].embedding
            session.execute(
                semantic_embeddings.insert().values(
                    entity_type="column" if text.startswith("Column") else "table",
                    entity_name=table_name if text.startswith("Table") else text.split()[1],
                    parent_table=None if text.startswith("Table") else table_name,
                    content=text,
                    embedding=emb
                )
            )

    session.commit()
    session.close()


def retrieve_embeddings(query: str, limit: int = 5):
    db_connection_string = os.getenv("CONNECTION_STRING")

    client = OpenAI()
    q_emb = client.embeddings.create(model="text-embedding-3-small", input=query).data[0].embedding

    engine = create_engine(db_connection_string)
    with engine.connect() as conn:
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
