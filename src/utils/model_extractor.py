import os, yaml
from sqlalchemy import create_engine, inspect

def extract_models_to_folder(db_connection_string: str, schema_name: str, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)

    engine = create_engine(db_connection_string)
    insp = inspect(engine)

    def map_type(coltype):
        t = str(coltype).lower()
        if "char" in t or "text" in t:
            if "(" in t:
                length = t[t.find('(')+1:t.find(')')]
                return f"string {length}"
            return "string"
        if "int" in t:
            return "integer"
        if "float" in t or "double" in t or "real" in t:
            return "float"
        if "date" in t and "time" not in t:
            return "date"
        if "time" in t:
            return "timestamp"
        if "uuid" in t:
            return "guid"
        return "string"

    with open(f"{output_folder}/_all_in.yaml", "w") as f:
        f.write("---\n")
        
    for table in insp.get_table_names(schema=schema_name):
        cols = insp.get_columns(table, schema=schema_name)
        pk = insp.get_pk_constraint(table, schema=schema_name)
        uniques = insp.get_unique_constraints(table, schema=schema_name)
        fks = insp.get_foreign_keys(table, schema=schema_name)
        
        # Debug: Print foreign key info
        print(f"Table: {table}")
        print(f"Foreign keys found: {len(fks)}")
        for f in fks:
            print(f"  FK: {f['constrained_columns']} -> {f.get('referred_schema', 'None')}.{f['referred_table']}.{f['referred_columns']}")

        foreign_keys = []
        for f in fks:
            ref_schema = f.get("referred_schema") or schema_name
            foreign_keys.append({
                "name": f["name"],
                "columns": ",".join(f["constrained_columns"]),
                "ref_schema": ref_schema,
                "ref_table": f["referred_table"],
                "ref_columns": ",".join(f["referred_columns"]),
            })
        
        keys = {
            "primary": ",".join(pk.get("constrained_columns", [])),
        }
        
        unique_constraints = [",".join(u.get("column_names", [])) for u in uniques if u.get("column_names")]
        if unique_constraints:
            keys["unique"] = ",".join(unique_constraints)
        
        if foreign_keys:
            keys["foreign"] = foreign_keys

        yaml_data = {
            "name": table,
            "schema": schema_name,
            "description": "None",
            "keys": keys,
            "columns": [
                {
                    "name": c["name"],
                    "type": map_type(c["type"]),
                    "description": f"{c.get("comment")}",
                    "nullable": c.get("nullable", True),
                }
                for c in cols
            ],
        }

        yaml_text = yaml.dump(yaml_data, None, sort_keys=False)

        with open(f"{output_folder}/{table}.yaml", "w") as f:
            f.write(yaml_text)

        with open(f"{output_folder}/_all_in.yaml", "a") as f:
            f.write(yaml_text)
            f.write("\n---\n")

def extract_models():
    target_schema = "workbench"
    output_dir = Path(os.path.expanduser(os.getenv("MODEL_OUTPUT_DIR"))) / target_schema
    connection_string = os.getenv("CONNECTION_STRING")

    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    extract_models_to_folder(
        connection_string=connection_string,
        output_folder=output_dir,
        schema_name=target_schema,
    )
