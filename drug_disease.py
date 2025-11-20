from flask import Flask, request, Response, render_template_string
import ollama
import faiss
import numpy as np
import pickle
import os
import sys
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# === EMBEDDING MODELS ===
OLLAMA_MODEL = "nomic-embed-text"
SENTENCE_MODEL = "BAAI/bge-small-en"
st_model = SentenceTransformer(SENTENCE_MODEL, device="cuda")

# === 1️⃣ Embedding function for Ollama ===
def get_embedding(text):
    response = ollama.embeddings(model=OLLAMA_MODEL, prompt=text)
    return np.array(response["embedding"], dtype="float32")

# === 2️⃣ Build/load FAISS for Drug & Disease (Ollama embeddings) ===
def build_or_load_faiss_index(cache_dir="cache", text_file="data/medical_data.txt"):
    os.makedirs(cache_dir, exist_ok=True)
    index_path = os.path.join(cache_dir, "faiss_index.bin")
    chunks_path = os.path.join(cache_dir, "chunks.pkl")

    if os.path.exists(index_path) and os.path.exists(chunks_path):
        print("🔁 Loading cached FAISS for Drug & Disease...")
        index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        return index, chunks

    print("⚙️ Building FAISS for Drug & Disease...")
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()
    words = text.split()
    chunks = [" ".join(words[i:i + 500]) for i in range(0, len(words), 500)]

    vectors = np.array([get_embedding(c) for c in chunks]).astype("float32")

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    print("✅ FAISS built for Drug & Disease.")
    return index, chunks


# === 3️⃣ Build/load FAISS for Drug Interaction (SentenceTransformer embeddings) ===
def build_or_load_faiss_interactions(cache_dir="cache/interaction", text_file="data/drug_interactions.txt"):
    os.makedirs(cache_dir, exist_ok=True)
    idx_file = os.path.join(cache_dir, "faiss.index")
    sent_file = os.path.join(cache_dir, "sentences.pkl")

    if os.path.exists(idx_file) and os.path.exists(sent_file):
        print("🔁 Loading cached FAISS for Interactions...")
        index = faiss.read_index(idx_file)
        with open(sent_file, "rb") as f:
            sentences = pickle.load(f)
        return index, sentences


# === Load both indexes ===
disease_index, disease_chunks = build_or_load_faiss_index()
interaction_index, interaction_sentences = build_or_load_faiss_interactions()


# === 4️⃣ Retrieval Functions ===
def retrieve_disease_context(query, k=3):
    query_vector = np.array([get_embedding(query)]).astype("float32")
    distances, indices = disease_index.search(query_vector, k)
    return "\n".join(disease_chunks[i] for i in indices[0])


def retrieve_interaction_context(query, k=5):
    q_vec = st_model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = interaction_index.search(q_vec, k)
    return "\n".join(interaction_sentences[i] for i in indices[0])


# === 5️⃣ Stream Responses ===
def stream_disease_answer(query):
    context = retrieve_disease_context(query)
    prompt = f"""You are a medical assistant.
Use the following medical context to answer clearly.

Context:
{context}

Question: {query}
Answer:"""
    for chunk in ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ):
        content = chunk.get("message", {}).get("content", "")
        if content:
            # sys.stdout.write(content)
            # sys.stdout.flush()
            yield content


def stream_interaction_answer(query):
    context = retrieve_interaction_context(query)
    prompt = f"""You are a drug interaction expert.
Use the following data to identify and explain drug interactions accurately.

Context:
{context}

Query: {query}
Answer:"""
    for chunk in ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ):
        content = chunk.get("message", {}).get("content", "")
        if content:
            # sys.stdout.write(content)
            # sys.stdout.flush()
            yield content


# === 6️⃣ Flask Routes ===
@app.route("/drug_disease/chat", methods=["POST"])
def disease_chat():
    data = request.get_json()
    query = data.get("message", "")
    if not query.strip():
        return Response("No query provided.", status=400)
    return Response(stream_disease_answer(query), mimetype="text/plain")


@app.route("/drug_interaction/chat", methods=["POST"])
def interaction_chat():
    data = request.get_json()
    query = data.get("message", "")
    if not query.strip():
        return Response("No query provided.", status=400)
    return Response(stream_interaction_answer(query), mimetype="text/plain")


# === Serve frontend ===
@app.route("/")
def home():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()
    return render_template_string(html)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
