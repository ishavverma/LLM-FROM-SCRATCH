import os
import shutil
import json
import logging

logger = logging.getLogger(__name__)

def get_project_dir():
    name = os.environ.get("ACTIVE_PROJECT")
    return os.path.join("projects", name) if name else ""

def export_project():
    """
    Exports the trained model, config, and minimal inference code into a self-contained directory.
    """
    project_dir = get_project_dir()
    config_path = os.path.join(project_dir, "config.json") if project_dir else "config.json"
    models_dir = os.path.join(project_dir, "models") if project_dir else "models"
    vector_dir = os.path.join(project_dir, "checkpoints") if project_dir else "checkpoints"

    if not os.path.exists(config_path):
        logger.error(f"Config file {config_path} not found. Cannot export.")
        return
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        
    project_name = config.get("project_name", "MyLLM")
    export_dir = f"{project_name}_export"
    
    print(f"Exporting project '{project_name}' to standalone directory: {export_dir}/")
    os.makedirs(export_dir, exist_ok=True)
    
    # 1. Export Models (Weights + Tokenizer)
    print("Copying model weights and tokenizer...")
    export_models_dir = os.path.join(export_dir, "models")
    if os.path.exists(models_dir):
        if os.path.exists(export_models_dir):
             shutil.rmtree(export_models_dir)
        shutil.copytree(models_dir, export_models_dir)
    else:
        print(f"Warning: {models_dir} not found. Exporting without weights.")

    # 2. Export RAG (FAISS Index)
    print("Copying RAG assets...")
    export_vector_dir = os.path.join(export_dir, "checkpoints")
    if os.path.exists(vector_dir):
         shutil.copytree(vector_dir, export_vector_dir, dirs_exist_ok=True)
        
    # 3. Export Config
    print("Copying configuration...")
    shutil.copy2(config_path, os.path.join(export_dir, "config.json"))
    
    # 4. Export Code Files
    print("Copying inference source files...")
    export_src_dir = os.path.join(export_dir, "src")
    os.makedirs(export_src_dir, exist_ok=True)
    
    # Create necessary subdirectories
    subdirs = ["model", "api", "api/templates", "safety", "retrieval"]
    for sd in subdirs:
        os.makedirs(os.path.join(export_src_dir, sd), exist_ok=True)
        # Create __init__.py
        open(os.path.join(export_src_dir, sd, "__init__.py"), 'a').close()
    
    open(os.path.join(export_src_dir, "__init__.py"), 'a').close()
    
    # Copy essential inference dependencies
    files_to_copy = [
        ("src/model/transformer.py", "src/model/transformer.py"),
        ("src/model/tokenizer.py", "src/model/tokenizer.py"),
        ("src/model/lora.py", "src/model/lora.py"),
        ("src/safety/guardrail.py", "src/safety/guardrail.py"),
        ("src/retrieval/rag.py", "src/retrieval/rag.py"),
        ("src/api/app.py", "src/api/app.py"),
    ]
    
    for src_file, dest_file in files_to_copy:
        if os.path.exists(src_file):
            shutil.copy2(src_file, os.path.join(export_dir, dest_file))
        else:
            print(f"Warning: {src_file} missing.")
            
    # Also copy templates and static directories if they exist
    if os.path.exists("src/api/templates"):
        shutil.copytree("src/api/templates", os.path.join(export_dir, "src", "api", "templates"), dirs_exist_ok=True)
    if os.path.exists("src/api/static"):
         shutil.copytree("src/api/static", os.path.join(export_dir, "src", "api", "static"), dirs_exist_ok=True)
            
    # Include an explicit requirements.txt
    with open(os.path.join(export_dir, "requirements.txt"), "w") as f:
        f.write("torch\nfaiss-cpu\nflask\nnumpy\nrank_bm25\n")
        
    # Include a standalone entry point that sets the project name blank so it uses local folders
    with open(os.path.join(export_dir, "run_inference.py"), "w") as f:
        f.write('import sys\nimport os\nos.environ["ACTIVE_PROJECT"] = ""\nfrom src.api.app import app\nif __name__ == "__main__":\n    print("Starting Standalone API on port 5000...")\n    app.run(host="0.0.0.0", port=5000)\n')

    print(f"\n[SUCCESS] Export Complete! Path: {export_dir}/")
    
if __name__ == "__main__":
    export_project()
    
if __name__ == "__main__":
    export_project()
