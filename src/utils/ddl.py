import os, yaml, json, jsonschema, jinja2, string
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def validate_model(project_dir: str, model_file: str):
    try:
        data = yaml.safe_load(open(os.path.join(project_dir, "models", model_file)))
        schema = json.load(open(f"{ROOT}/resources/schemas/model.schema.json"))
        jsonschema.validate(instance=data, schema=schema)
    except Exception as e:
        raise Exception(f"Invalid model: {model_file} in project folder: {project_dir}; from error: {e}")

def do_validation(project_dir: str): 
    for model_name in os.listdir(os.path.join(project_dir, "models")):
        print(f"Validating model: {model_name}")
        
        validate_model(project_dir, model_name)

        print(f"Model {model_name} is valid")

def apply_template(project_dir: str, output_dir: str, project_name: str, default_schema: str, target: str):
    os.makedirs(output_dir, exist_ok=True)

    create_sql = ""
    fk_sql = ""

    for model_file in os.listdir(os.path.join(project_dir, "models")):
        print(f"Applying template to model: {model_file}")
        
        with open(os.path.join(project_dir, "models", model_file)) as f:
            content = string.Template(f.read()).substitute(os.environ)
            data = yaml.safe_load(content)
            template = jinja2.Template(open(f"{ROOT}/resources/templates/{target}.j2").read())

            model_data = data.copy()
            if 'schema' not in model_data:
                model_data['schema'] = default_schema
            
            model_create, model_fk = template.render(name=model_file.replace(".yaml", ""), **model_data).split("--FOREIGN KEYS", 1)
            
            create_sql += model_create
            fk_sql += model_fk

    with open(f"{output_dir}/{project_name}.{target}.sql", "w") as script:
        script.write(create_sql)
        script.write(fk_sql)

def generate_ddl(project_dir: str, output_dir: str):
    try:
        data = yaml.safe_load(open(os.path.join(project_dir, "project.yaml")))
        schema = json.load(open(f"{ROOT}/resources/schemas/project.schema.json"))
        jsonschema.validate(instance=data, schema=schema)
    except Exception as e:
        raise Exception(f"Invalid project folder: {project_dir}; from error: {e}")

    print("Project is valid")

    with open(os.path.join(project_dir, "project.yaml")) as f:
        content = string.Template(f.read()).substitute(os.environ)
        project = yaml.safe_load(content)

        project_name = project.get("name")

        for action in project.get("actions", []):
            action_type = action["type"]
            schema_name = action["schema"]
            if action_type == "yaml-to-ddl":    
                do_validation(project_dir)
                apply_template(project_dir, output_dir, project_name, schema_name, action["target"])
            elif action_type == "ddl-to-yaml":
                print(f"Reverse-engineering YAML from {action['source']}")

def do_ddl():
    project_dir = Path(os.path.expanduser("~/Projects/redfive/io/upstream"))
    output_dir = Path(os.path.expanduser("~/Projects/redfive/io/ddl-out"))

    generate_ddl(
        project_dir=project_dir,
        output_dir=output_dir,
    )
