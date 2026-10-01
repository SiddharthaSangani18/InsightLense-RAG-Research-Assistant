import os
import sys
import gc
import stat
import uuid
import shutil

# Suppress HuggingFace and transformers warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "false"
import logging
import base64
from google import genai
from google.genai import types
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import time
import threading

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

from llama_parse import LlamaParse
import fitz

from dotenv import load_dotenv

load_dotenv()
client = genai.Client(vertexai=False, api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ─────────────────────────────────────────────────────────────────────────────
# SESSION / FAISS LIFECYCLE
#
# • The FAISS store + retriever live IN MEMORY and are the single source of
#   truth for /ask. Nothing re-opens the index files per question.
# • Every upload is saved to its OWN folder: DB_FAISS_PATH/session_<id>/
#   A tiny pointer file (ACTIVE_SESSION) records which folder is current so a
#   server restart can restore it.
# • Reset is LOGICAL: it clears memory + pointer and always succeeds.
#   Old folders are removed afterwards, best-effort, in the background.
# ─────────────────────────────────────────────────────────────────────────────
DB_FAISS_PATH = "vectorstore/db_faiss_v2"
POINTER_NAME = "ACTIVE_SESSION"
POINTER_PATH = os.path.join(DB_FAISS_PATH, POINTER_NAME)

# Where the embedding model files are cached (downloaded once on first start)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBED_CACHE_DIR = os.getenv("EMBED_CACHE_DIR", os.path.join(BASE_DIR, "model_cache"))
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

pdf_memory = None            # active FAISS vectorstore (in memory)
active_retriever = None      # active FAISS + BM25 ensemble retriever
active_session_name = None   # folder name of the active session on disk
building_sessions = set()    # session folders currently being written
conversation_history = []

state_lock = threading.RLock()   # guards the globals above
purge_lock = threading.Lock()    # only one purge at a time

last_interaction_time = time.time()
TIMEOUT_SECONDS = 3600  # 1 Hour

_embeddings = None
_embeddings_lock = threading.Lock()


def get_embeddings():
    """Single shared FastEmbed embeddings instance (loaded once, reused everywhere)."""
    global _embeddings
    with _embeddings_lock:
        if _embeddings is None:
            _embeddings = FastEmbedEmbeddings(
                model_name=EMBED_MODEL_NAME,
                cache_dir=EMBED_CACHE_DIR,
                max_length=256,   # this model truncates at 256 tokens anyway
                batch_size=16,    # default 256 can spike RAM on large PDFs
                threads=1,        # avoid spawning one ONNX thread per host core
            )
            print("Using FastEmbed (ONNX) embeddings.")
        return _embeddings


def update_interaction():
    """Reset the timer whenever the user does something."""
    global last_interaction_time
    last_interaction_time = time.time()


# ───────────────────────── disk helpers (best-effort) ─────────────────────────
def _on_rm_error(func, path, exc):
    """Clear read-only flag and retry once; never raise."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


def _safe_rmtree(path):
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_on_rm_error)
    else:
        shutil.rmtree(path, onerror=_on_rm_error)


def _write_pointer(session_name):
    """Atomically record which session folder is active."""
    try:
        os.makedirs(DB_FAISS_PATH, exist_ok=True)
        tmp = POINTER_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(session_name)
        os.replace(tmp, POINTER_PATH)
    except Exception as e:
        print(f"Warning: could not write session pointer: {e}")


def _clear_pointer():
    try:
        if os.path.exists(POINTER_PATH):
            os.remove(POINTER_PATH)
    except Exception:
        try:
            with open(POINTER_PATH, "w", encoding="utf-8") as f:
                f.write("")
        except Exception as e:
            print(f"Warning: could not clear session pointer: {e}")


def _read_pointer():
    try:
        if os.path.exists(POINTER_PATH):
            with open(POINTER_PATH, "r", encoding="utf-8") as f:
                return f.read().strip()
    except Exception:
        pass
    return ""


def purge_stale_sessions():
    """
    Delete every session folder (and legacy index files) that is not the active
    session and not currently being built. Never raises. Returns the number of
    entries that could not be removed (they are retried later).
    """
    if not purge_lock.acquire(blocking=False):
        return 0
    leftovers = 0
    try:
        if not os.path.isdir(DB_FAISS_PATH):
            return 0

        with state_lock:
            keep = set(building_sessions)
            if active_session_name:
                keep.add(active_session_name)
        keep.add(POINTER_NAME)

        for entry in os.listdir(DB_FAISS_PATH):
            if entry in keep:
                continue
            full = os.path.join(DB_FAISS_PATH, entry)
            try:
                if os.path.isdir(full):
                    _safe_rmtree(full)
                else:
                    try:
                        os.chmod(full, stat.S_IWRITE)
                    except Exception:
                        pass
                    os.remove(full)
            except Exception:
                pass
            if os.path.exists(full):
                leftovers += 1
        return leftovers
    except Exception as e:
        print(f"Purge error (will retry later): {e}")
        return 0
    finally:
        purge_lock.release()


def schedule_purge(attempts=5, delay=3):
    """Run purge in the background, retrying a few times, without blocking requests."""
    def _run():
        for _ in range(attempts):
            if purge_stale_sessions() == 0:
                return
            time.sleep(delay)
    threading.Thread(target=_run, daemon=True).start()


# ───────────────────────── in-memory session state ─────────────────────────
def _build_retriever(vector_db):
    """Build the FAISS + BM25 ensemble retriever once per session."""
    # FIX 3: k = 8 on both FAISS and BM25
    faiss_retriever = vector_db.as_retriever(search_kwargs={"k": 8})
    docstore_docs = list(vector_db.docstore._dict.values())

    if docstore_docs:
        bm25_retriever = BM25Retriever.from_documents(docstore_docs)
        bm25_retriever.k = 8
        return EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )
    return faiss_retriever


def _activate_session(vector_db, session_name):
    """Swap the active session in memory (atomic under the lock)."""
    global pdf_memory, active_retriever, active_session_name, conversation_history
    retriever = _build_retriever(vector_db)
    with state_lock:
        pdf_memory = vector_db
        active_retriever = retriever
        active_session_name = session_name
        conversation_history = []


def _clear_session_state():
    """Logical reset: drop everything from memory and forget the pointer."""
    global pdf_memory, active_retriever, active_session_name, conversation_history
    with state_lock:
        pdf_memory = None
        active_retriever = None
        active_session_name = None
        conversation_history = []
    _clear_pointer()
    gc.collect()


def load_active_session_from_disk():
    """Startup: restore the active session (if any) into memory."""
    name = _read_pointer()
    if not name or not name.startswith("session_") or os.sep in name or "/" in name:
        return False
    path = os.path.join(DB_FAISS_PATH, name)
    if not os.path.isdir(path):
        return False
    try:
        vector_db = FAISS.load_local(path, get_embeddings(), allow_dangerous_deserialization=True)
        _activate_session(vector_db, name)
        return True
    except Exception as e:
        print(f"Failed to load existing vectorstore: {e}")
        return False


def cleanup_loop():
    """Background thread: reset the session after 1 hour of inactivity and purge stale folders."""
    while True:
        time.sleep(60)
        try:
            elapsed = time.time() - last_interaction_time

            if elapsed > TIMEOUT_SECONDS and active_retriever is not None:
                print(f"--- INACTIVITY DETECTED ({elapsed:.0f}s). CLEARING SESSION... ---")
                _clear_session_state()
                print("--- CLEANUP COMPLETE ---")

            purge_stale_sessions()
        except Exception as e:
            print(f"Error during cleanup: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# FIX 1: Improved image summarization prompt — now captures spatial layout,
#         centre elements, connections, and positional relationships so that
#         questions like "what is the centre of the POPIT model?" are answered
#         correctly from the [VISUAL ANALYSIS] block.
# ─────────────────────────────────────────────────────────────────────────────
def summarize_image(image_bytes):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg",
                ),
                """Analyze this image in detail and describe ALL of the following:

1. LAYOUT & STRUCTURE: Describe the spatial arrangement — what is at the center, what surrounds it, how elements are positioned relative to each other (top, bottom, left, right, inside, outside, overlapping).

2. If it is a DIAGRAM (e.g. lifecycle, model, framework, flowchart):
   - List EVERY labeled element with its exact position in the diagram.
   - Explicitly state what element is at the CENTER of the diagram (if any).
   - Describe what is at the top, bottom, left, right, and middle.
   - Describe the overall shape or structure (circle, triangle, arrow, grid, etc.).

3. If it is a CHART or GRAPH:
   - Extract specific numbers, percentages, and values shown.
   - Describe trends, axis labels, legend meanings, and data series.

4. If it is a TABLE:
   - Summarize key rows and columns with their values.
   - Highlight important comparisons between columns (e.g. different years).

5. CONNECTIONS & ARROWS:
   - Describe every arrow, line, or connector — what it links and its direction.
   - Note whether arrows are single-headed, double-headed, curved, or straight.

6. TEXT LABELS:
   - List all visible text labels exactly as written in the image.

Be extremely specific about positions and relationships. Do not omit any labeled element."""
            ]
        )
        return response.text
    except Exception as e:
        print("Gemini Vision Error:", e)
        return "[Image caption failed]"


# ───────────────────────────── startup ─────────────────────────────
if os.path.isdir(DB_FAISS_PATH):
    print("Found existing vectorstore folder. Restoring active session...")
    if load_active_session_from_disk():
        print("Vectorstore loaded successfully.")
    else:
        print("No active session to restore.")
    schedule_purge()  # remove legacy / stale folders in the background

cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
cleanup_thread.start()


def process_multimodal_pdf(pdf_path: str):
    print("--- STARTING INGESTION ---")

    try:
        parser = LlamaParse(
            result_type="markdown",
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            verbose=True
        )
        parsed_docs = parser.load_data(pdf_path)
    except Exception as e:
        print("LlamaParse Error:", e)
        return 0

    doc = fitz.open(pdf_path)
    full_combined_text = ""

    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_num = page_index + 1

            images = page.get_images(full=True)
            drawings = page.get_drawings()
            has_visuals = len(images) > 0 or len(drawings) > 0

            caption = ""

            if has_visuals:
                print(f"Visuals found on Page {page_num}. Rendering page...")
                try:
                    time.sleep(2.5)

                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_bytes = pix.tobytes("png")

                    caption = summarize_image(img_bytes)

                    if "429" in caption or "quota" in caption.lower():
                        print(f"Skipping Page {page_num} due to Rate Limit.")
                        caption = ""
                    else:
                        caption = f"\n\n[VISUAL ANALYSIS OF PAGE {page_num}]\n{caption}\n[END VISUAL ANALYSIS]\n"

                except Exception as e:
                    print(f"Visual Analysis Failed for Page {page_num}: {e}")
                    caption = ""

            text_content = ""
            if page_index < len(parsed_docs):
                text_content = parsed_docs[page_index].text

            full_combined_text += f"--- PAGE {page_num} ---\n{text_content}\n{caption}\n\n"
    finally:
        # Release the file handle on the temp PDF immediately
        doc.close()

    # ─────────────────────────────────────────────────────────────────────────
    # FIX 2: Larger chunk size so [VISUAL ANALYSIS] blocks are NOT split across
    #         chunks. At 500 chars the blocks were frequently cut in half,
    #         causing the retriever to pick up incomplete visual descriptions.
    # ─────────────────────────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,   # increased from 500
        chunk_overlap=200  # increased from 100
    )
    chunks = splitter.split_text(full_combined_text)

    documents = [Document(page_content=chunk, metadata={"source": pdf_path}) for chunk in chunks]
    embeddings = get_embeddings()

    # Build the new index in memory, save it to its OWN folder, then swap it in.
    new_store = FAISS.from_documents(documents, embeddings)

    session_name = f"session_{uuid.uuid4().hex[:12]}"
    session_path = os.path.join(DB_FAISS_PATH, session_name)

    with state_lock:
        building_sessions.add(session_name)
    try:
        os.makedirs(DB_FAISS_PATH, exist_ok=True)
        new_store.save_local(session_path)
        _write_pointer(session_name)
        _activate_session(new_store, session_name)
    finally:
        with state_lock:
            building_sessions.discard(session_name)

    # The previous document's folder is now stale — remove it in the background.
    schedule_purge()

    print(f"--- INGESTION COMPLETE: {len(chunks)} chunks stored. ---")
    return len(chunks)


def get_ai_response(query: str):
    global conversation_history

    # Use the in-memory retriever — no per-question disk loading.
    with state_lock:
        retriever_to_use = active_retriever
        history_snapshot = list(conversation_history[-5:])

    if retriever_to_use is None:
        return "System Error: Please re-upload the PDF to initialize the database."

    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.1)

    clean_history = []
    for msg in history_snapshot:
        role = msg['role']
        content = msg['content'].replace("{", "(").replace("}", ")")
        clean_history.append(f"{role}: {content}")

    history_str = "\n".join(clean_history)

    # ─────────────────────────────────────────────────────────────────────────
    # FIX 4: Improved prompt — now explicitly instructs the LLM to:
    #   • PRIORITIZE [VISUAL ANALYSIS] sections for diagram/figure questions
    #   • Answer centre/position questions from the spatial description only
    #   • Not override visual analysis with surrounding text
    # ─────────────────────────────────────────────────────────────────────────
    prompt_template_str = f"""
    You are an expert Business and Financial Analyst.

    # CHAT HISTORY:
    {history_str}

    # CONTEXT:
    {{context}}

    # USER QUESTION:
    {{input}}

    # INSTRUCTIONS:
    0. **Scope:** Only answer from the provided context. If the topic is not covered in the context, apologize and clearly state that the document does not contain that information.

    1. **Figures & Diagrams — HIGHEST PRIORITY RULE:** When the question refers to a figure, diagram, model, or any visual element (e.g. "Figure 1.1", "Figure 1.3", "POPIT model", "business change lifecycle", "centre point", "what is in the middle"), you MUST search for and give PRIORITY to the [VISUAL ANALYSIS OF PAGE X] sections in the context. These sections contain the definitive, ground-truth spatial description of the diagram as it actually appears.

    2. **Spatial / Position Questions:** If the user asks about the centre, middle, top, bottom, left, right, or any positional aspect of a diagram or model, answer STRICTLY from the [VISUAL ANALYSIS] spatial description. Do NOT infer positions from surrounding text — the visual analysis is the authoritative source.

    3. **Figure Identification:** If asked about a specific figure number (e.g. "Figure 1.3"), find the [VISUAL ANALYSIS] block on that page and describe ALL labeled elements, their positions, arrows, and connections as stated in the analysis.

    4. **Charts & Graphs:** Use [VISUAL ANALYSIS] sections to answer questions about chart values, trends, axis labels, and legend meanings.

    5. **Exact Numbers:** Look for EXACT numbers (e.g., "2%", "20%") — do not approximate or guess.

    6. **Tables:** Do not confuse column years (e.g. 2024 vs 2025). Always read column headers carefully.

    Answer:
    """

    prompt = PromptTemplate(
        input_variables=["context", "input"],
        template=prompt_template_str
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever_to_use, document_chain)

    response = retrieval_chain.invoke({"input": query})
    answer = response["answer"]

    with state_lock:
        # Only record history if the session wasn't reset/replaced mid-question
        if active_retriever is retriever_to_use:
            conversation_history.append({"role": "User", "content": query})
            conversation_history.append({"role": "AI", "content": answer})

    return answer


@app.post("/reset-session")
async def reset_session():
    """Logically resets the session. Never depends on a directory delete succeeding."""
    try:
        print("--- RESETTING SESSION ---")

        # 1. Drop all in-memory state (store, retriever, history) and the pointer.
        _clear_session_state()

        # 2. Delete old folders in the background, best-effort (retried if locked).
        schedule_purge()

        update_interaction()

        print("--- SESSION RESET COMPLETE ---")

        return {
            "success": True,
            "message": "Session reset successfully."
        }

    except Exception as e:
        print(f"--- RESET ERROR: {e} ---")

        return {
            "success": False,
            "error": str(e)
        }


@app.get("/check-session")
async def check_session():
    """Returns True if a document session is active, telling UI to skip upload."""
    update_interaction()
    with state_lock:
        ready = active_retriever is not None
    return {"ready": ready}


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    update_interaction()
    global conversation_history
    with state_lock:
        conversation_history = []

    if not file.filename.lower().endswith(".pdf"):
        return {"success": False, "error": "Invalid file type."}

    temp_filename = f"temp_{file.filename}"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        num_chunks = process_multimodal_pdf(temp_filename)
        update_interaction()
        return {"success": True, "chunks": num_chunks}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        try:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        except Exception as e:
            print(f"Could not remove temp file {temp_filename}: {e}")


@app.post("/ask")
async def ask(question: str = Form(...)):
    update_interaction()
    with state_lock:
        has_session = active_retriever is not None
    if not has_session:
        return {"success": False, "result": "Please upload a PDF first."}
    return {"success": True, "result": get_ai_response(question)}


@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")


@app.get("/home")
async def home():
    return {"message": "Multi-Modal RAG API is Live."}