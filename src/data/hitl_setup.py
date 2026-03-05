import json
import os
import argparse

def setup_dataset():
    print("\n=== Level 1 HITL: Dataset Configuration Setup ===")
    
    config_path = "config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = {"files": []}
        
        if "project_name" not in config:
            config["project_name"] = "MyLLM"
            
    print(f"Current Project Name: {config.get('project_name')}")
    print(f"Current Configured Files: {len(config.get('files', []))}")
    for i, file_obj in enumerate(config.get('files', [])):
        print(f"  {i+1}. {file_obj.get('path')} (Type: {file_obj.get('type')})")
        
    print("\nDo you want to change the Project Name? (y/n)")
    choice = input("Choice: ").strip().lower()
    if choice.startswith('y'):
        new_name = input("Enter new project name: ").strip()
        if new_name:
            config["project_name"] = new_name
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
            print(f"Project name updated to '{new_name}'.")
            
    print("\nDo you want to add a new file to the ingestion pipeline? (y/n)")
    choice = input("Choice: ").strip().lower()
    if choice.startswith('y'):
        file_path = input("Enter file path (e.g. data.pdf or data.jsonl): ").strip()
        dataset_type = input("Enter dataset type [base / instruct / chat]: ").strip().lower()
        if dataset_type not in ['base', 'instruct', 'chat']:
            print("Invalid type. Defaulting to 'base'.")
            dataset_type = 'base'
            
        if "files" not in config:
            config["files"] = []
            
        config["files"].append({"path": file_path, "type": dataset_type})
        
        # update config.json
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        print(f"Added {file_path} as {dataset_type}. Updated config.json.")
    else:
        print("No changes made to config.json.")
        
    print("Stage 1 Setup complete.")
        
if __name__ == "__main__":
    setup_dataset()
