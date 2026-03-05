import argparse
import os
import sys
import subprocess
import json
import shutil
from src.data.extractors.router import IngestionRouter
from src.data.document_intelligence import DocumentIntelligence
from src.data.chunker import SemanticChunker
from src.model.train import train

def get_project_dir(project_name):
    return os.path.join("projects", project_name)

def load_config(project_name=None):
    # Try local project config first, then global fallback
    if project_name:
        path = os.path.join(get_project_dir(project_name), "config.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    
    if os.path.exists("config.json"):
        with open("config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def run_configure():
    print("\n--- Interactive Project Configuration ---")
    project_name = input("Enter project name: ").strip()
    if not project_name:
        print("Error: Project name required.")
        return
        
    project_dir = get_project_dir(project_name)
    os.makedirs(project_dir, exist_ok=True)
    os.makedirs(os.path.join(project_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(project_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(project_dir, "logs"), exist_ok=True)
    
    # Discovery files in data/
    available_files = []
    if os.path.exists("data"):
        available_files = [f for f in os.listdir("data") if os.path.isfile(os.path.join("data", f))]
    
    print("\nAvailable files in global 'data/' folder:")
    for i, f in enumerate(available_files):
        print(f"[{i}] {f}")
        
    file_indices = input("Enter indices of files to include (comma separated, e.g. 0,2): ").strip()
    selected_files = []
    try:
        for idx in file_indices.split(","):
            if idx.strip():
                fname = available_files[int(idx.strip())]
                selected_files.append({"path": os.path.join("data", fname), "type": "base"})
    except (ValueError, IndexError):
        print("Invalid selection. Defaulting to first file if available.")
        if available_files:
            selected_files = [{"path": os.path.join("data", available_files[0]), "type": "base"}]

    config = {
        "project_name": project_name,
        "files": selected_files,
        "target_language": "en",
        "remove_headers": True,
        "remove_footers": True,
        "remove_page_numbers": True,
        "remove_watermarks": True
    }
    
    config_path = os.path.join(project_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
        
    # Also update the global config.json to point to the active project name if desired
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump({"active_project": project_name}, f, indent=4)
        
    print(f"\n[SUCCESS] Project '{project_name}' configured at {project_dir}")

def run_extraction(project_name):
    print(f"Running document extraction for project: {project_name}...")
    project_dir = get_project_dir(project_name)
    config = load_config(project_name)
    
    doc_intel = DocumentIntelligence(os.path.join(project_dir, "config.json") if os.path.exists(os.path.join(project_dir, "config.json")) else "config.json")
    
    files = config.get("files", [])
    all_chunks = []
    
    for f_obj in files:
        file_path = f_obj.get("path")
        ds_type = f_obj.get("type", "base")
        if not os.path.exists(file_path): continue
            
        extractor = IngestionRouter.get_extractor(file_path)
        chunker = SemanticChunker(max_words=256)
        
        print(f"Extracting {file_path}...")
        seen_text_blocks = set()
        
        for _, blocks in extractor.extract_pages():
            cleaned_text_blocks = []
            for b_tuple in blocks:
                if len(b_tuple) < 5: continue
                txt = b_tuple[4].strip()
                
                # Cleanup common OCR / Training artifacts
                txt = txt.replace("TRAINING COPY", "").strip()
                
                # Deduplication at block level
                if not txt or txt in seen_text_blocks:
                    continue
                
                # Heuristic: skip very short fragments that are usually artifacts
                if len(txt) < 5:
                    continue
                
                seen_text_blocks.add(txt)
                cleaned_text_blocks.append(txt)
            
            # Combine unique blocks from this page back into a filterable list for doc_intel
            # doc_intel expects blocks in same format, but we'll simplified list for it
            # Actually, doc_intel.filter_blocks takes the raw blocks. 
            # Let's adjust doc_intel to handle this or filter after.
            
            # Revised approach: filter inside the loop
            page_text = "\n".join(cleaned_text_blocks)
            if page_text:
                chunks = chunker.chunk_text(page_text)
                for c in chunks:
                    all_chunks.append({"text": c, "type": ds_type})
            
    out_path = os.path.join(project_dir, "data", "raw_chunks.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=4)
    print(f"Extraction complete. {len(all_chunks)} chunks saved to {out_path}.")

def run_hitl(project_name):
    print(f"Running HITL for project: {project_name}...")
    project_dir = get_project_dir(project_name)
    raw_path = os.path.join(project_dir, "data", "raw_chunks.json")
    app_path = os.path.join(project_dir, "data", "approved_chunks.json")
    subprocess.run([sys.executable, "-m", "src.data.hitl", "--input", raw_path, "--output", app_path])

def run_training(project_name):
    print(f"Running training for project: {project_name}...")
    project_dir = get_project_dir(project_name)
    config = load_config(project_name) # Load config here to get training parameters
    model_dir = os.path.join(project_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    data_path = os.path.join(project_dir, "data", "approved_chunks.json")
    if not os.path.exists(data_path):
        raw_path = os.path.join(project_dir, "data", "raw_chunks.json")
        if os.path.exists(raw_path):
            print(f"Applying automatic approval: using {raw_path}")
            data_path = raw_path
        else:
            print("Error: No data found for training. Run 'extract' first.")
            return
    
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Pass project path for state saving if updated train support it, else we move files
        train(data_path=data_path, 
              epochs=config.get("epochs", 5), 
              batch_size=config.get("batch_size", 4), 
              block_size=256,
              device=device)
        
        # Move global saved weights to project weights if train.py saves to 'models/' globally
        if os.path.exists("models/transformer_weights.pth"):
            shutil.move("models/transformer_weights.pth", os.path.join(model_dir, "transformer_weights.pth"))
        if os.path.exists("models/tokenizer.json"):
            shutil.move("models/tokenizer.json", os.path.join(model_dir, "tokenizer.json"))
        if os.path.exists("runs"): # Tensorboard logs
             log_dest = os.path.join(project_dir, "logs")
             # Move current experiment logs
             shutil.move("runs", log_dest)

    except Exception as e:
        print(f"Training failed: {e}")

def run_rag_index(project_name):
    print(f"Building RAG Index for project: {project_name}...")
    project_dir = get_project_dir(project_name)
    data_path = os.path.join(project_dir, "data", "approved_chunks.json")
    if not os.path.exists(data_path):
        raw_path = os.path.join(project_dir, "data", "raw_chunks.json")
        if os.path.exists(raw_path):
            data_path = raw_path
            
    model_path = os.path.join(project_dir, "models", "transformer_weights.pth")
    tok_path = os.path.join(project_dir, "models", "tokenizer.json")
    
    try:
        import torch
        from src.model.transformer import GPTLanguageModel
        from src.model.tokenizer import BPETokenizer
        from src.retrieval.rag import RAGPipeline, VectorStore
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = BPETokenizer()
        if os.path.exists(tok_path): tokenizer.load(tok_path)
            
        vocab_size = tokenizer.total_vocab_size if tokenizer.vocab else 1000
        model = GPTLanguageModel(vocab_size=vocab_size, d_model=128, n_heads=4, n_layer=2, block_size=256)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device).eval()
        
        vector_store = VectorStore(embedding_dim=128)
        rag = RAGPipeline(model, tokenizer, vector_store, device=device)
        rag.build_index(data_path)
        
        # Move faiss index from global checkpoints to project folder
        if os.path.exists("checkpoints"):
             shutil.move("checkpoints", os.path.join(project_dir, "checkpoints"))
    except Exception as e:
        print(f"RAG Indexing failed: {e}")

def start_api(project_name):
    print(f"Starting API for project: {project_name}...")
    env = os.environ.copy()
    env["FLASK_APP"] = "src.api.app"
    env["ACTIVE_PROJECT"] = project_name
    subprocess.run([sys.executable, "-m", "flask", "run", "--host=127.0.0.1", "--port=5000"], env=env)

def run_export(project_name):
    print(f"Exporting project: {project_name}...")
    env = os.environ.copy()
    env["ACTIVE_PROJECT"] = project_name
    subprocess.run([sys.executable, "-m", "src.model.exporter"], env=env)

def main():
    parser = argparse.ArgumentParser(description="LLM-FROM-SCRATCH Project Manager")
    parser.add_argument('--step', type=str, required=True, 
                        choices=['configure', 'extract', 'hitl', 'train', 'rag_index', 'export', 'api', 'all'])
    parser.add_argument('--project', type=str, help="Project name (overrides config.json active_project)")
    
    args = parser.parse_args()
    
    # Resolve active project
    project_name = args.project
    if not project_name:
        global_cfg = load_config()
        project_name = global_cfg.get("active_project")
        
    if args.step == 'configure':
        run_configure()
        return

    if not project_name:
        print("Error: No active project found. Run '--step configure' first.")
        return

    if args.step == 'extract': run_extraction(project_name)
    elif args.step == 'hitl': run_hitl(project_name)
    elif args.step == 'train': run_training(project_name)
    elif args.step == 'rag_index': run_rag_index(project_name)
    elif args.step == 'export': run_export(project_name)
    elif args.step == 'api': start_api(project_name)
    elif args.step == 'all':
        run_extraction(project_name)
        # For 'all', we might want to skip hitl if non-interactive, or prompt
        run_hitl(project_name)
        run_training(project_name)
        run_rag_index(project_name)
        run_export(project_name)
        start_api(project_name)

if __name__ == '__main__':
    main()
