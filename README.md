# LLM-FROM-SCRATCH 🦁

A state-of-the-art, decoder-only Transformer language model built entirely from scratch in Python and PyTorch. This project serves as a comprehensive "Laboratory" for large language model exploration, supporting advanced features like **Rotary Positional Embeddings (RoPE)**, **Grouped-Query Attention (GQA)**, **KV Caching**, and **Hybrid Retrieval-Augmented Generation (RAG)**—all without external API dependencies.

---

## 🚀 Architectural Blueprint (GPT-S)
This project doesn't just replicate basic Transformers; it implements the modern scaling laws used in models like Llama:
- **Core Engine:** Decoder-only blocks with **Pre-Normalization** (LayerNorm before attention) for training stability.
- **Advanced Attention:** 
  - **GQA (Grouped-Query Attention):** Reduces KV cache size, allowing for much larger batch sizes and faster inference.
  - **RoPE (Rotary Position Embeddings):** Allows the model to generalize beyond its training window by encoding relative positions mathematically.
- **Dynamic Context:** Scalable `block_size` (default 256) allowing for comprehensive prompt/context handling.
- **KV Caching:** Optimized inference with $O(1)$ step complexity for near-instant token generation.
- **Deduplicated Memory:** RAG pipeline with **FAISS** (dense) and **BM25** (sparse) fusion for factual accuracy.

---

## 📂 Component Breakdown
- **`run_pipeline.py`**: The Central CLI Command Center. Handles project lifecycle, configuration, and pipeline orchestration.
- **`src/model/`**:
  - `transformer.py`: The heart of the model (GPT architecture + RoPE + GQA).
  - `tokenizer.py`: Custom Byte-Pair Encoding (BPE) implementation.
  - `train.py`: High-performance training loop with AMP and Cosine Annealing.
- **`src/retrieval/`**: 
  - `rag.py`: Manages the RAG pipeline and hybrid search logic.
  - `vector_store.py`: Interface for FAISS vector storage.
- **`src/data/`**: 
  - `document_intelligence.py`: Heuristics for OCR artifact cleaning and quality filtering.
  - `extractors/`: Modular parsers for PDF, JSON, CSV, and XLSX.
- **`projects/`**: The workspace isolation layer. Ensures that context, weights, and configurations for `SampleGPT` don't interfere with your other models.

---

## 🛠️ Installation & Setup

1. **Clone & Enter Registry:**
   ```bash
   git clone https://github.com/ishavverma/LLM-FROM-SCRATCH.git
   cd LLM-FROM-SCRATCH
   ```

2. **Isolated Environment Setup:**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3. **Install Core Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📖 The User Manual: Building Your Personal LLM Knowledge Base

### Phase 1: Data Preparation
Place your raw knowledge documents (PDF, JSON, TXT) in the `/data` directory. 
> [!IMPORTANT]
> The model uses **Semantic Deduplication**. If you have repeated pages or headers across documents, the pipeline will automatically "fingerprint" and skip them to prevent overfitting on noise.

### Phase 2: Project Orchestration
Generate a new isolated project context:
```bash
python run_pipeline.py --step configure
```
- **Action:** Enter a project name (e.g., `FinancialAssistant`).
- **Action:** Select the raw data indices to incorporate.
- **Result:** A `projects/FinancialAssistant/config.json` is generated with your specific dataset path.

### Phase 3: The Training Pipeline
You can trigger the end-to-end "Automatic Pilot":
```bash
python run_pipeline.py --step all --project FinancialAssistant
```
This single command executes the following sequence:
1. **Extraction:** Cleans OCR noise and breaks data into semantic chunks.
2. **Human-in-the-Loop (Optional):** Allows you to review and approve data chunks.
3. **Training:** Fine-tunes the GPT-S layers on your specific vocabulary.
4. **Indexing:** Populates the FAISS vector database for RAG.
5. **Launch:** Starts the API server.

### Phase 4: Scaling & Fine-Tuning
- **Individual Steps:** You can re-run individual modules to refine the model. For example, to just rebuild the search index:
  ```bash
  python run_pipeline.py --step rag_index --project FinancialAssistant
  ```

### Phase 5: Deployment & Interaction
Launch your custom Chat Interface:
```bash
python run_pipeline.py --step api --project FinancialAssistant
```
- **Interface:** Navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000).
- **Manual Control:** Use the `Chat UI` to ask questions. The model will retrieve snippets from your PDFs and provide grounded, cited answers.

---

## 🔬 Scientific Methodology & Training Tips
- **Loss Masking:** We implement "Answer-only" loss. The model is NOT penalized for missing "System" prompt tokens—it only learns to generate the perfect answer.
- **Cosine Annealing:** The learning rate starts high (1e-3) and smoothly decays to zero (1e-5) following a cosine curve, ensuring the model doesn't overfit in the final epochs.
- **Top-p Sampling:** Inference uses Nucleus Sampling ($p=0.9$) to ensure diversity while maintaining factual grounding.

---

## 🛠️ Detailed Command Reference

| Command Flag | Description | Technical Output |
| :--- | :--- | :--- |
| `--step configure` | Wizard for workspace setup | `projects/{name}/config.json` |
| `--step extract` | Modular data parsing | `data/raw_chunks.json` |
| `--step train` | GPT optimization loop | `models/transformer_weights.pth` |
| `--step rag_index` | Vector database build | `checkpoints/faiss_index.bin` |
| `--step api` | Flask SSE Server | Live Chat UI on Port 5000 |
| `--step all` | End-to-end Automation | Complete trained project assets |

## 📜 License
MIT License. Built for modders, researchers, and developers who want to own their AI stack.
