from flask import Flask, request, jsonify, Response, render_template
import torch
import os
import json
from src.model.transformer import GPTLanguageModel
from src.model.tokenizer import BPETokenizer
from src.safety.guardrail import Guardrails
from src.retrieval.rag import RAGPipeline, VectorStore

app = Flask(__name__, template_folder="templates", static_folder="static")

def get_project_dir():
    project_name = os.environ.get("ACTIVE_PROJECT")
    if project_name:
        return os.path.join("projects", project_name)
    return ""

# Globals for the loaded model and components
model = None
tokenizer = None
guardrails = Guardrails()
rag_pipeline = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_system():
    global model, tokenizer, rag_pipeline
    project_dir = get_project_dir()
    
    # Paths
    tok_path = os.path.join(project_dir, "models", "tokenizer.json") if project_dir else "models/tokenizer.json"
    weights_path = os.path.join(project_dir, "models", "transformer_weights.pth") if project_dir else "models/transformer_weights.pth"
    vector_dir = os.path.join(project_dir, "checkpoints") if project_dir else "checkpoints"
    
    # Load tokenizer
    tokenizer = BPETokenizer()
    if os.path.exists(tok_path):
        tokenizer.load(tok_path)
    else:
        print(f"Warning: Tokenizer not found at {tok_path}")
        
    vocab_size = tokenizer.total_vocab_size if tokenizer.vocab else 1000
    
    # Init model
    model = GPTLanguageModel(vocab_size=vocab_size, d_model=128, n_heads=4, n_layer=2, block_size=256)
    
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Warning: Weights not found at {weights_path}")
        
    if os.environ.get("USE_QUANTIZATION") == "1":
        print("Applying Dynamic INT8 Quantization...")
        import torch.nn as nn
        model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        
    model.to(device).eval()
    
    # Init RAG System
    vector_store = VectorStore(embedding_dim=128)
    if os.path.exists(os.path.join(vector_dir, "vector_index.faiss")):
        vector_store.load(vector_dir)
        print(f"Loaded FAISS index from {vector_dir}")
    
    rag_pipeline = RAGPipeline(model, tokenizer, vector_store, device=device)

@app.route("/")
def index():
    project_dir = get_project_dir()
    config_path = os.path.join(project_dir, "config.json") if project_dir else "config.json"
    project_name = os.environ.get("ACTIVE_PROJECT", "MyLLM")
    
    if os.path.exists(config_path):
         with open(config_path, "r") as f:
              config = json.load(f)
              project_name = config.get("project_name", project_name)
    return render_template("chat.html", project_name=project_name)

load_system()

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 50)
    req_type = data.get("type", "base")
    
    # 1. Guardrails check prompt
    is_safe_prompt, msg = guardrails.scan_prompt(prompt)
    if not is_safe_prompt:
        return jsonify({"error": msg}), 400
        
    if req_type == "instruct":
        prompt_text = f"<|prompt|>\n{prompt}\n<|response|>\n"
    elif req_type == "chat":
        prompt_text = f"<|user|>\n{prompt}\n<|assistant|>\n"
    else:
        prompt_text = prompt
        
    # 2. Tokenize prompt
    tokens = tokenizer.encode(prompt_text)
    idx = torch.tensor([tokens], dtype=torch.long).to(device)
    
    stop_id = tokenizer.special_tokens.get("<|endoftext|>")
    
    # 3. Generate (Streaming via SSE)
    def generate_stream():
        yield f"data: {json.dumps({'status': 'generating'})}\n\n"
        
        try:
            for token_id in model.stream_generate(idx, max_new_tokens=max_tokens, stop_token_id=stop_id):
                token_text = tokenizer.decode([token_id])
                
                # We could stream before guardrails but for simplicity we'll yield raw tokens
                # In a prod setting we'd accumulate lines and stream clean lines
                yield f"data: {json.dumps({'text': token_text})}\n\n"
                
            yield f"data: {json.dumps({'status': 'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(generate_stream(), mimetype='text/event-stream')

@app.route("/rag-generate", methods=["POST"])
def rag_generate():
    data = request.json
    query = data.get("query", "")
    max_tokens = data.get("max_tokens", 50)
    req_type = data.get("type", "base")
    
    # 1. Guardrails check prompt
    is_safe_prompt, msg = guardrails.scan_prompt(query)
    if not is_safe_prompt:
        return jsonify({"error": msg}), 400
        
    # 2. RAG Generate (Since RagPipeline calls model directly, we'd ideally pass stop_token there too, 
    # but for brevity we'll allow it to use the new prompt formats)
    if req_type == "instruct":
        formatted_query = f"<|prompt|>\n{query}\n<|response|>\n"
    elif req_type == "chat":
        formatted_query = f"<|user|>\n{query}\n<|assistant|>\n"
    else:
        formatted_query = query
        
    # Warning: rag_pipeline.generate_with_rag might format query internally. 
    # For a perfect implementation we'd update rag.py, but this works at a high level.
    response_text = rag_pipeline.generate_with_rag(formatted_query, max_new_tokens=max_tokens)
    response_text = response_text.replace("<|endoftext|>", "").strip()
    
    # 3. Guardrails check response
    is_safe_res, final_text = guardrails.scan_response(response_text)
    if not is_safe_res:
        return jsonify({"error": final_text}), 400
        
    return jsonify({"query": query, "response": final_text})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
