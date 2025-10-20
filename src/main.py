import os, yaml
from pathlib import Path
from dotenv import load_dotenv
from embeddings import retrieve_embeddings, build_graph, expand_with_graph

load_dotenv()

# def create_embeddings():
#     connection_string = os.getenv("CONNECTION_STRING")
#     output_dir = Path(os.path.expanduser("~/Projects/redfive/io/upstream/models"))
#     create_embeddings(connection_string, output_dir)


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

if __name__ == "__main__":
    models = {}
    models_dir = Path(os.path.expanduser("~/Projects/redfive/io/upstream/models"))
    for fname in os.listdir(models_dir):
        with open(os.path.join(models_dir, fname)) as f:
            model = yaml.safe_load(f)
            model["name"] = fname.replace(".yaml", "")
            models[model["name"]] = model

    db_connection_string = os.getenv("CONNECTION_STRING")
    query = "which table contains production volume data"

    results = retrieve_embeddings(query, 5)
    graph = build_graph(list(models.values()))
    table_names = expand_with_graph(graph, results)

    context = build_llm_context(models, table_names)

    user_query = """
        Write a sql statement which shows year to date production by production area for every day in 2025
    """

    prompt = f"""
        You are a SQL generator. Use only the schema below.

        {context}

        Generate a single SQL query answering:
        '{user_query}'

        Rules:
        - Use only tables and columns from the schema.
        - Use explicit JOIN conditions where foreign keys exist.
        - Do not invent columns or tables.
        SQL:
        """

    from openai import OpenAI
    client = OpenAI()

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    sql = resp.choices[0].message.content

    print(sql)
