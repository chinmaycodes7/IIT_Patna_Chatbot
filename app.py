import streamlit as st
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ================= SETTINGS =================
SAVE_PATH = "./index/"
TOP_K = 3

st.set_page_config(page_title="IIT Patna Assistant", page_icon="🎓")
st.title("🎓 IIT Patna Academic Assistant")
st.caption("Ask questions about calendar, syllabus, registration, semester schedule etc.")

# ================= LOAD MODELS =================
@st.cache_resource(show_spinner="Loading models... first run takes time")
def load_models():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Embedding model
    embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)

    # Load FAISS DB
    index = faiss.read_index(SAVE_PATH + "index.faiss")
    chunks = np.load(SAVE_PATH + "chunks.npy", allow_pickle=True)
    metadata = np.load(SAVE_PATH + "metadata.npy", allow_pickle=True)

    # LLM
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto"
    )

    llm = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=80,
        temperature=0.0,
        do_sample=False,
        repetition_penalty=1.1,
        return_full_text=False
    )

    return embed_model, index, chunks, metadata, llm


embed_model, index, chunks, metadata, llm = load_models()

# ================= RETRIEVE =================
def retrieve_context(query):

    query_embedding = embed_model.encode([query])
    distances, indices = index.search(query_embedding, TOP_K)

    retrieved = []
    for i in indices[0]:
        if i < len(chunks):
            retrieved.append(chunks[i])

    return "\n".join(retrieved)

# ================= GENERATE =================
def generate_answer(query):

    context = retrieve_context(query)

    prompt = f"""
You are IIT Patna Academic Assistant.

Rules:
- Answer ONLY using the given context
- If answer not present say: Information not available in academic database.
- Maximum 3 lines
- Do not guess

Context:
{context}

Question: {query}
Answer:
"""

    output = llm(prompt)
    answer = output[0]["generated_text"].strip()

    if len(answer) < 5:
        answer = "Information not available in academic database."

    return answer

# ================= CHAT UI =================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask something about IIT Patna...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("Searching academic database..."):
            response = generate_answer(query)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})