"""
STEP 4 — The actual chatbot app. This is what you demo live at the expo.

Pipeline for every question a student types:
  1. Your TRAINED classifier predicts the Subject + Chapter it belongs to.
  2. The retrieval index finds the most relevant textbook chunks.
  3. A local LLM (via Ollama) turns those chunks into a natural-language
     answer. If Ollama isn't running, we fall back to just showing the
     retrieved textbook text directly (still works fine as a demo).

Run with:  streamlit run 4_app.py
Then open the local URL it prints (usually http://localhost:8501)
"""

import os
import joblib
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer

MODEL_DIR = "models"
TOP_K = 3  # how many textbook chunks to retrieve per question

# Try to import ollama — it's optional. If not installed/running, we just
# skip the "natural language answer" step and show retrieved text instead.
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

OLLAMA_MODEL = "cbse-study-bot"  # your Unsloth-finetuned model (see Modelfile + GUIDE.md Step 4)
FALLBACK_MODEL = "llama3.2:3b"   # used automatically if the fine-tuned model isn't imported yet


@st.cache_resource
def load_everything():
    classifier_bundle = joblib.load(os.path.join(MODEL_DIR, "classifier.joblib"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    faiss_index = faiss.read_index(os.path.join(MODEL_DIR, "faiss_index.bin"))
    chunk_lookup = joblib.load(os.path.join(MODEL_DIR, "chunk_lookup.joblib"))
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return classifier_bundle, label_encoder, faiss_index, chunk_lookup, embedder


def predict_topic(question, classifier_bundle, label_encoder):
    model = classifier_bundle["model"]
    vectorizer = classifier_bundle["vectorizer"]
    X = vectorizer.transform([question])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0].max()
    label = label_encoder.inverse_transform([pred])[0]
    return label, proba


def retrieve_chunks(question, faiss_index, chunk_lookup, embedder, k=TOP_K):
    q_vec = embedder.encode([question]).astype("float32")
    faiss.normalize_L2(q_vec)
    scores, indices = faiss_index.search(q_vec, k)
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:
            continue
        row = chunk_lookup.iloc[idx]
        results.append({"text": row["text"], "subject": row["subject"], "chapter": row["chapter"], "score": float(score)})
    return results


def generate_answer(question, retrieved_chunks):
    context = "\n\n".join(f"- {c['text']}" for c in retrieved_chunks)
    prompt = f"""You are a helpful CBSE Class 11 study assistant.
Answer the student's question using ONLY the textbook excerpts below.
If the excerpts don't fully answer it, say what they do cover.
Keep the answer clear and exam-friendly (3-5 sentences).

Textbook excerpts:
{context}

Student's question: {question}

Answer:"""

    if OLLAMA_AVAILABLE:
        try:
            response = ollama.generate(model=OLLAMA_MODEL, prompt=prompt)
            return response["response"].strip()
        except Exception:
            # Fine-tuned model not imported yet (see GUIDE.md Step 4) — fall
            # back to the base model so the app still works during setup.
            try:
                response = ollama.generate(model=FALLBACK_MODEL, prompt=prompt)
                return response["response"].strip() + "\n\n*(using base model — run `ollama create cbse-study-bot -f Modelfile` to use your fine-tuned model)*"
            except Exception:
                return f"(Local LLM not reachable — is 'ollama serve' running? Showing retrieved text instead.)\n\n{context}"
    else:
        # Fallback: no LLM installed, just show the retrieved textbook text.
        return context


def main():
    st.set_page_config(page_title="CBSE Study Bot", page_icon="📚")
    st.title("📚 CBSE Class 11 Study Bot")
    st.caption("Ask a question from your syllabus — the bot classifies the topic with a model it trained itself, then answers using your textbook content.")

    classifier_bundle, label_encoder, faiss_index, chunk_lookup, embedder = load_everything()

    question = st.text_input("Ask a question:", placeholder="e.g. What is the first law of thermodynamics?")

    if st.button("Ask") and question.strip():
        with st.spinner("Thinking..."):
            topic, confidence = predict_topic(question, classifier_bundle, label_encoder)
            chunks = retrieve_chunks(question, faiss_index, chunk_lookup, embedder)
            answer = generate_answer(question, chunks)

        st.subheader("🏷️ Detected topic (from the trained classifier)")
        st.info(f"**{topic}**  (confidence: {confidence:.0%})")

        st.subheader("💡 Answer")
        st.write(answer)

        with st.expander("See the textbook excerpts used"):
            for c in chunks:
                st.markdown(f"**{c['subject']} - {c['chapter']}** (similarity: {c['score']:.2f})")
                st.write(c["text"])
                st.divider()


if __name__ == "__main__":
    main()
