# DOCTOR_ASSISTANT

A small Flask-based assistant that helps with:
- answering drug / disease related questions using semantic retrieval over a local medical text corpus, and
- explaining drug–drug interactions using a separate interaction dataset.

The app builds and caches FAISS indexes for fast nearest-neighbor retrieval, uses Ollama for text embeddings and chat generation (for the drug/disease pipeline), and uses SentenceTransformer embeddings (for drug interaction retrieval). The server streams responses back to clients.

---

Contents
- Overview & Purpose
- Requirements & Supported Platforms
- Setup (install dependencies, prepare models and data)
- How it works (high level)
- Running the app
- Example API calls
- Configuration & troubleshooting
- Notes & security

---

Overview & Purpose
------------------
This project is intended to provide a small, local medical assistant prototype that:
- Uses semantic search (FAISS) over textual medical resources to provide context-aware answers to freeform queries about drugs and diseases.
- Separately searches a drug-interactions dataset and uses an LLM to summarize and explain interactions.
- Streams model-generated text back to a client via HTTP endpoints.

This is intended as a research/demo tool — NOT a production clinical system. It is not a substitute for professional medical advice.

Prerequisites
-------------
- Python 3.8+
- pip
- A running Ollama daemon (local Ollama) with the embedding and chat models available:
  - The code expects an embedding model identified by `OLLAMA_MODEL = "nomic-embed-text"`.
  - The chat model used in code is `"llama3.2"`.
  Ensure you have installed/pulled the corresponding models into your Ollama instance.
- For drug interaction retrieval, SentenceTransformers requires PyTorch. If you want GPU acceleration, install a CUDA-enabled torch build.
- FAISS: either `faiss-cpu` or `faiss-gpu` (depending on your platform).
- Optional GPU for the SentenceTransformer model (the code uses `device="cuda"` by default). If you have no GPU, change the device to `"cpu"` in the source.

Recommended Python packages
- flask
- ollama (the Python client to talk to local Ollama)
- sentence-transformers
- numpy
- faiss-cpu (or faiss-gpu)
- pickle (built-in)
- (optionally) torch (required by sentence-transformers)
- (optionally) tqdm (for long builds, not required)

A sample pip command (adjust depending on CPU/GPU and your platform):
```
# Example (CPU-only FAISS + CPU PyTorch)
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install flask ollama sentence-transformers numpy faiss-cpu
# If sentence-transformers requires torch, install that too:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

If you want GPU:
- Install CUDA-capable PyTorch as per https://pytorch.org
- Install `faiss-gpu` (availability depends on your OS and Python wheel availability).

Setup
-----
1. Clone the repository:
```
git clone https://github.com/rohit-mewada-1125/DOCTOR_ASSISTANT.git
cd DOCTOR_ASSISTANT
```

2. Create and activate a virtual environment, then install dependencies (see recommended packages above).

3. Prepare the data files the app expects:
- `data/medical_data.txt` — a plain UTF-8 text file with medical/drug/disease content. The code will split it into chunks of ~500 words when building the FAISS index.
- `data/drug_interactions.txt` — a plain UTF-8 text file where each sentence / line corresponds to an interaction record (or a separated textual record) used for building the interaction FAISS index.

Example structure:
```
data/
  medical_data.txt
  drug_interactions.txt
```

4. Ensure Ollama is running locally and the models are installed/pulled. (See Ollama docs for installing and pulling models).
   - Example (pseudo):
     - `ollama run` or `ollama daemon` to start the Ollama server.
     - `ollama pull nomic-embed-text` (if that is the correct model identifier in your Ollama setup)
     - `ollama pull llama3.2` (or the correct chat model name)

Note: The exact model names must match the ones you have installed in your Ollama instance; the code uses `"nomic-embed-text"` and `"llama3.2"` by default.

How it works (high level)
-------------------------
- On startup the app will attempt to load two FAISS indexes from `cache/`:
  - `cache/faiss_index.bin` + `cache/chunks.pkl` — for the medical/drug/disease text (Ollama embeddings).
  - `cache/interaction/faiss.index` + `cache/interaction/sentences.pkl` — for drug interactions (SentenceTransformer embeddings).
- If the index files are absent, it will build them:
  - For `medical_data.txt`: text is split into 500-word chunks, an embedding is requested via Ollama for each chunk, and stored in a FAISS IndexFlatL2.
  - For `drug_interactions.txt`: SentenceTransformer encodes sentences and the embeddings are stored in a FAISS index.
- When a user POSTs to the relevant endpoint, the app:
  - Retrieves top-k nearest context chunks from the appropriate FAISS index.
  - Builds a prompt with the retrieved context and forwards it to Ollama chat to generate an answer.
  - Streams the response chunks back to the client.

Running the app
---------------
1. Ensure required data files exist (see Setup).
2. Ensure Ollama is running and required models are available.
3. Run the Flask app:
```
python drug_disease.py
```
This starts the server on http://0.0.0.0:5000 by default (Flask debug True in code).

API Endpoints
-------------
- Serve frontend:
  - GET `/` -> serves `templates/index.html` (if present)
- Drug & Disease chat (streams plain text):
  - POST `/drug_disease/chat`
  - Request JSON body: `{"message": "What are the side effects of aspirin?"}`
  - Response: streamed text/plain content
- Drug Interaction chat (streams plain text):
  - POST `/drug_interaction/chat`
  - Request JSON body: `{"message": "Does drug A interact with drug B?"}`
  - Response: streamed text/plain content

Example curl:
```
curl -X POST http://localhost:5000/drug_disease/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What are the common side effects of ibuprofen?"}'
```

Notes on caching
----------------
- Built FAISS indexes and chunk/sentences pickles are stored under:
  - `cache/` and `cache/interaction/`
- Re-running the app will reuse cached indexes if present. Delete the cache to force re-build.

Configuration & quick edits
---------------------------
- Change the embedding/chat model names in `drug_disease.py`:
  - OLLAMA_MODEL = "nomic-embed-text"
  - Chat model used inside streaming functions: currently `"llama3.2"`
- If you do not have a CUDA GPU, change:
  - `SentenceTransformer(SENTENCE_MODEL, device="cuda")` -> `device="cpu"` in `drug_disease.py`
- Adjust chunk size if you want to create larger/smaller FAISS chunks (currently 500 words).

Troubleshooting
---------------
- Ollama connection errors:
  - Ensure the Ollama daemon is running and reachable from the environment where the Flask app runs.
  - Check the Ollama client and install correct Python package/version for your system.
- FAISS import / install issues:
  - On some platforms you must install `faiss-cpu` or `faiss-gpu` via pip or conda. See FAISS docs.
- SentenceTransformer / Torch:
  - If sentence-transformers fails to load a model because of missing torch, install torch (CPU or GPU flavor).
- GPU memory errors:
  - If the SentenceTransformer or models are on GPU and you run out of memory, switch to CPU.
- If results are poor:
  - Improve or expand `data/medical_data.txt` and `data/drug_interactions.txt`.
  - Tune chunking or k (number of retrieved nearest neighbors).

Security & privacy
------------------
- This project may send text to local Ollama; ensure no sensitive patient data is processed unless you have proper approvals and security.
- This demo is not HIPAA-compliant or intended for real patient care.

Extending the project
---------------------
- Add authentication to the Flask app.
- Replace or augment Ollama with another LLM provider.
- Add structured metadata for the interaction dataset and more precise indexing.
- Add a web UI (templates/index.html is a placeholder).

Acknowledgements & references
-----------------------------
- Ollama (local LLMs & embeddings)
- SentenceTransformers (Hugging Face)
- FAISS (Facebook AI Similarity Search)
- Flask

License
-------
Please add a LICENSE file to your repository if you intend to publish this code.

If you want, I can also:
- generate a requirements.txt for the project,
- produce a Dockerfile to containerize the app,
- or create a simple `templates/index.html` example UI to test the streaming endpoints.
