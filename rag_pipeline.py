"""
MedQuery RAG Pipeline — extracted directly from the Jupyter notebook.
Same documents, same chunking, same embeddings, same FAISS, same retrieval, same LLM call.
Only addition: lightweight state tracking for metrics.

Extended with live document ingestion / deletion without server restart.
"""

import time
import uuid
import threading
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import faiss
from sentence_transformers import SentenceTransformer
import anthropic

from config import ANTHROPIC_API_KEY


# ── Medical document knowledge base (identical to notebook Section 3) ──────────

MEDICAL_DOCUMENTS = [
    {
        "id": "doc_001",
        "type": "Lab Report",
        "title": "Complete Blood Count — Patient Ravi Kumar",
        "content": """
COMPLETE BLOOD COUNT (CBC) REPORT
Patient: Ravi Kumar | DOB: 14-Mar-1985 | MRN: RK-20241
Collected: 10-Jan-2025 | Lab: Apollo Diagnostics, Hyderabad

RESULTS SUMMARY:
Haemoglobin: 11.2 g/dL [Reference: 13.0–17.0] — LOW
RBC Count: 3.9 million/mcL [Reference: 4.5–5.5] — LOW
WBC Count: 9,800 /mcL [Reference: 4,500–11,000] — NORMAL
Platelet Count: 210,000 /mcL [Reference: 150,000–400,000] — NORMAL
MCV: 68 fL [Reference: 80–100] — LOW (microcytic)
MCH: 22 pg [Reference: 27–33] — LOW
Haematocrit: 34% [Reference: 40–52%] — LOW
Neutrophils: 62% | Lymphocytes: 30% | Monocytes: 6% | Eosinophils: 2%

INTERPRETATION:
Findings are consistent with microcytic hypochromic anaemia, most likely due to iron deficiency.
Low MCV and MCH with reduced haemoglobin suggest inadequate iron stores.

RECOMMENDATIONS:
1. Serum ferritin and serum iron levels to confirm iron deficiency.
2. Dietary counselling — increase iron-rich foods (red meat, spinach, lentils).
3. Consider oral iron supplementation (Ferrous Sulphate 200 mg BD) if ferritin confirmed low.
4. Repeat CBC in 8 weeks after starting treatment.
5. Rule out chronic blood loss (stool occult blood test recommended).

Reported by: Dr. Priya Mehta, MD Pathology | Signature verified digitally
"""
    },
    {
        "id": "doc_002",
        "type": "Prescription",
        "title": "Prescription — Hypertension Management",
        "content": """
PRESCRIPTION
Dr. Arun Sharma, MD (Internal Medicine) | Reg No: MCI-78432
Care Hospital, Jubilee Hills, Hyderabad | Tel: 040-2222-3333
Date: 05-Jan-2025

Patient: Sunita Reddy | Age: 58 years | Weight: 72 kg
Diagnosis: Essential Hypertension (Stage 2), Dyslipidaemia

Rx:
1. Tab. Amlodipine 5 mg — Once daily (morning) — 30 days
   [Calcium channel blocker — for blood pressure control]

2. Tab. Telmisartan 40 mg — Once daily (morning) — 30 days
   [ARB — for blood pressure and kidney protection]

3. Tab. Atorvastatin 20 mg — Once daily (night) — 30 days
   [Statin — to reduce LDL cholesterol]

4. Tab. Aspirin 75 mg — Once daily (after breakfast) — 30 days
   [Antiplatelet — cardiovascular risk reduction]

WARNINGS & INSTRUCTIONS:
- Do NOT stop medications without consulting the doctor, even if BP feels normal.
- Monitor BP at home twice daily and maintain a log.
- Avoid NSAIDs (ibuprofen, diclofenac) — can raise BP and interact with Telmisartan.
- Atorvastatin: Report any unexplained muscle pain or weakness immediately.
- Salt restriction: limit sodium intake to less than 2g/day.
- Alcohol: limit to minimal or avoid completely.
- Regular aerobic exercise: 30 minutes, 5 days a week.

Follow-up: 4 weeks | Fasting lipid profile + kidney function test before next visit.
"""
    },
    {
        "id": "doc_003",
        "type": "Discharge Summary",
        "title": "Discharge Summary — Acute Appendicitis",
        "content": """
DISCHARGE SUMMARY
Yashoda Hospitals, Secunderabad
IP No: YH-2025-00891 | Ward: Surgical Ward B

Patient: Mohammed Farhan | Age: 27 | Gender: Male
Admission: 08-Jan-2025 | Discharge: 11-Jan-2025 | Stay: 3 days

PRESENTING COMPLAINT:
Sudden onset right iliac fossa pain for 18 hours, nausea, low-grade fever (38.1°C).

DIAGNOSIS: Acute Appendicitis (confirmed intraoperatively)

INVESTIGATIONS:
- WBC: 14,200/mcL (elevated — suggestive of infection)
- CRP: 42 mg/L (elevated)
- USG Abdomen: Dilated, non-compressible appendix (8mm diameter), periappendiceal fat stranding
- CT Abdomen (plain): Confirmed acute appendicitis, no perforation

PROCEDURE:
Laparoscopic Appendicectomy performed on 08-Jan-2025 at 21:30 hrs.
Surgeon: Dr. Raghavendra Rao, MS (General Surgery).
Anaesthesia: General anaesthesia, uneventful.
Operative findings: Inflamed appendix, no perforation, no peritoneal soiling.
Histopathology sent — awaited.

HOSPITAL COURSE:
Post-operative recovery was smooth. Oral feeds started on Day 1 post-op.
IV antibiotics (Ceftriaxone + Metronidazole) given for 48 hours, switched to oral.
Ambulated on Day 1. No post-op complications.

DISCHARGE MEDICATIONS:
1. Tab. Amoxicillin-Clavulanate 625 mg — twice daily × 5 days
2. Tab. Metronidazole 400 mg — three times daily × 5 days
3. Tab. Paracetamol 500 mg — as needed for pain (max 4 times/day)
4. Tab. Pantoprazole 40 mg — once daily (morning, empty stomach) × 7 days

DISCHARGE INSTRUCTIONS:
- Keep wound clean and dry. Dress every 2 days.
- No heavy lifting or strenuous activity for 4 weeks.
- Soft diet for 1 week; gradually resume normal diet.
- Watch for: fever >38°C, wound redness/discharge, increasing pain → visit ER immediately.
- Histopathology report collection: after 7 days from Apollo Diagnostics.

Follow-up: 14-Jan-2025 with Dr. Raghavendra Rao (OPD)
"""
    },
    {
        "id": "doc_004",
        "type": "Radiology Report",
        "title": "MRI Brain Report — Headache Workup",
        "content": """
MRI BRAIN REPORT
Imaging Centre: KIMS Radiology, Hyderabad
Study Date: 12-Jan-2025 | Report Date: 12-Jan-2025
Ref. Physician: Dr. Sneha Kulkarni, DM Neurology

Patient: Lakshmi Narayana | Age: 45 | Gender: Female | MRN: KH-45901
Clinical History: Recurrent severe headaches for 3 months, worse in mornings, associated with visual disturbances.

TECHNIQUE:
MRI brain performed on 3 Tesla scanner.
Sequences: T1, T2, FLAIR, DWI, T1 post-gadolinium contrast.

FINDINGS:

Cerebral Parenchyma:
- No acute infarct or restricted diffusion on DWI.
- No intracranial haemorrhage.
- White matter: Few scattered T2/FLAIR hyperintensities in bilateral periventricular regions — non-specific, may represent early small vessel disease.
- Grey-white matter differentiation: Preserved.

Ventricles & CSF Spaces:
- Lateral ventricles mildly prominent bilaterally.
- Third and fourth ventricles: Normal.
- No midline shift.

Posterior Fossa:
- Cerebellum and brainstem: No focal lesion.
- No Chiari malformation.

Post-contrast:
- No abnormal parenchymal or meningeal enhancement.
- No space-occupying lesion identified.

Vessels (MRA not performed — clinically not requested):
- Major intracranial flow voids appear grossly preserved.

Orbits & Sinuses:
- Bilateral maxillary sinuses show mucosal thickening — suggestive of chronic sinusitis.
- Ethmoid sinuses partially opacified.

IMPRESSION:
1. No acute intracranial pathology.
2. Non-specific periventricular white matter changes — clinical correlation advised; may represent early cerebrovascular disease.
3. Chronic bilateral maxillary and ethmoid sinusitis — could be contributing to headache symptoms.
4. Clinical and further ENT evaluation recommended for sinus-related headache management.

Reported by: Dr. Kiran Bhat, MD Radiology, DNB | KIMS Imaging Centre
"""
    }
]


# ── Text Chunking (identical to notebook Section 4) ───────────────────────────

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_type: str
    doc_title: str
    content: str
    chunk_index: int


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 120) -> List[str]:
    """Split text into overlapping chunks, preferring sentence boundaries."""
    text = ' '.join(text.split())
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            for punct in ['.', '!', '?', '\n']:
                boundary = text.rfind(punct, start + int(chunk_size * 0.5), end)
                if boundary != -1 and boundary + 1 > start:
                    end = boundary + 1
                    break

        chunk = text[start:end].strip()
        if len(chunk) > 40:
            chunks.append(chunk)

        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def build_chunks(documents: List[Dict]) -> List[Chunk]:
    all_chunks = []
    for doc in documents:
        text_chunks = chunk_text(doc['content'])
        for i, chunk_text_content in enumerate(text_chunks):
            all_chunks.append(Chunk(
                chunk_id=f"{doc['id']}_chunk_{i:03d}",
                doc_id=doc['id'],
                doc_type=doc['type'],
                doc_title=doc['title'],
                content=chunk_text_content,
                chunk_index=i
            ))
    return all_chunks


# ── RAG Pipeline state + functions ────────────────────────────────────────────

# Thread lock for safe concurrent ingestion / deletion
_lock = threading.Lock()

# Module-level state
documents: List[Dict] = []          # Live registry of all indexed documents
chunks: List[Chunk] = []
chunk_embeddings: List[np.ndarray] = []   # Per-chunk embeddings (kept in sync with chunks)
index: Optional[faiss.IndexFlatIP] = None
embedding_model: Optional[SentenceTransformer] = None
client: Optional[anthropic.Anthropic] = None

# Metrics tracking
total_queries = 0
cumulative_latency_ms = 0.0
cumulative_similarity = 0.0
total_similarity_count = 0


SYSTEM_PROMPT = """You are MedQuery AI, an expert medical document assistant.

Your rules:
- Answer ONLY based on the provided document context.
- If the context does not contain the answer, say: "The document doesn't contain this information."
- Use clear, simple language suitable for patients and caregivers.
- Mention medications, values, or recommendations precisely as stated.
- Always remind the user to consult a healthcare professional for personal medical decisions.
- Use bullet points and bold text for readability when appropriate."""


def initialize():
    """Load model, build chunks, create FAISS index. Called once at startup."""
    global chunks, chunk_embeddings, index, embedding_model, client, documents

    print("[INFO] Loading SentenceTransformer model...")
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    dim_size = getattr(embedding_model, 'get_embedding_dimension', getattr(embedding_model, 'get_sentence_embedding_dimension', None))()
    print(f"[OK] Model loaded -- embedding dimension: {dim_size}")

    # Initialize documents registry from the hard-coded notebook data
    documents = [
        {"id": doc["id"], "type": doc["type"], "title": doc["title"], "content": doc["content"]}
        for doc in MEDICAL_DOCUMENTS
    ]

    # Build chunks
    chunks = build_chunks(MEDICAL_DOCUMENTS)
    print(f"[OK] Chunking complete -- {len(chunks)} chunks")

    # Generate embeddings
    chunk_texts = [c.content for c in chunks]
    emb_matrix = embedding_model.encode(
        chunk_texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True
    )
    emb_matrix = np.array(emb_matrix, dtype='float32')

    # Store per-chunk embeddings for efficient delete/rebuild
    chunk_embeddings = [emb_matrix[i] for i in range(emb_matrix.shape[0])]
    print(f"[OK] Embeddings generated -- shape: {emb_matrix.shape}")

    # Build FAISS index
    dim = emb_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb_matrix)
    print(f"[OK] FAISS index built -- {index.ntotal} vectors, {dim} dims")

    # Anthropic client
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    print("[OK] Anthropic client ready")


# ── Live document ingestion ───────────────────────────────────────────────────

def ingest_document(doc_type: str, title: str, content: str) -> Dict:
    """
    Ingest a new document at runtime:
      1. Auto-generate doc_id
      2. Chunk the text
      3. Embed the chunks
      4. Append to FAISS index
      5. Register in documents list

    Returns a summary dict.
    """
    global chunks, chunk_embeddings, index, documents

    doc_id = f"doc_{uuid.uuid4().hex[:8]}"

    with _lock:
        # Chunk
        text_chunks = chunk_text(content)
        if not text_chunks:
            return {"doc_id": doc_id, "chunks_added": 0, "total_chunks": len(chunks), "total_documents": len(documents)}

        new_chunks = []
        for i, tc in enumerate(text_chunks):
            new_chunks.append(Chunk(
                chunk_id=f"{doc_id}_chunk_{i:03d}",
                doc_id=doc_id,
                doc_type=doc_type,
                doc_title=title,
                content=tc,
                chunk_index=i
            ))

        # Embed
        new_texts = [c.content for c in new_chunks]
        new_emb = embedding_model.encode(
            new_texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        new_emb = np.array(new_emb, dtype='float32')

        # Append to FAISS
        index.add(new_emb)

        # Update module state
        chunks.extend(new_chunks)
        for i in range(new_emb.shape[0]):
            chunk_embeddings.append(new_emb[i])

        documents.append({"id": doc_id, "type": doc_type, "title": title, "content": content})

        print(f"[INGEST] Added '{title}' ({doc_id}) — {len(new_chunks)} chunks, index now {index.ntotal} vectors")

        return {
            "doc_id": doc_id,
            "chunks_added": len(new_chunks),
            "total_chunks": len(chunks),
            "total_documents": len(documents)
        }


def delete_document(doc_id: str) -> Dict:
    """
    Remove a document and rebuild the FAISS index.

    Reuses existing embeddings for remaining chunks — no re-encoding.
    Only the FAISS index is rebuilt from scratch.
    """
    global chunks, chunk_embeddings, index, documents

    with _lock:
        # Check document exists
        if not any(d["id"] == doc_id for d in documents):
            return {"success": False, "message": f"Document '{doc_id}' not found"}

        # Identify which chunk indices belong to this document
        keep_indices = [i for i, c in enumerate(chunks) if c.doc_id != doc_id]

        if len(keep_indices) == len(chunks):
            return {"success": False, "message": f"Document '{doc_id}' not found in chunks"}

        removed_count = len(chunks) - len(keep_indices)

        # Rebuild chunks and embeddings lists (reusing existing embeddings)
        chunks = [chunks[i] for i in keep_indices]
        chunk_embeddings = [chunk_embeddings[i] for i in keep_indices]

        # Rebuild FAISS index from remaining embeddings
        dim = index.d
        index = faiss.IndexFlatIP(dim)
        if chunk_embeddings:
            emb_matrix = np.stack(chunk_embeddings).astype('float32')
            index.add(emb_matrix)

        # Remove from documents registry
        doc_title = next((d["title"] for d in documents if d["id"] == doc_id), doc_id)
        documents = [d for d in documents if d["id"] != doc_id]

        print(f"[DELETE] Removed '{doc_title}' ({doc_id}) — {removed_count} chunks removed, index now {index.ntotal} vectors")

        return {"success": True, "message": f"Deleted '{doc_title}' — removed {removed_count} chunks"}


# ── Retrieval + RAG answer (unchanged logic) ─────────────────────────────────

def retrieve(query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
    """Embed a query and retrieve the top-K most relevant chunks."""
    query_vec = embedding_model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec, dtype='float32')

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append((chunks[idx], float(score)))

    return results


def rag_answer(question: str, top_k: int = 5) -> Dict:
    """
    Full RAG pipeline: retrieve → build prompt → call Claude → return answer + sources.
    """
    global total_queries, cumulative_latency_ms, cumulative_similarity, total_similarity_count

    start_time = time.time()

    # Step 1: Retrieve
    retrieved = retrieve(question, top_k=top_k)

    if not retrieved:
        return {
            "answer": "No relevant documents found for this question.",
            "sources": [],
            "retrieved_chunks": [],
            "similarity_scores": [],
            "latency_ms": 0,
            "tokens_used": 0
        }

    # Step 2: Build context
    context_parts = []
    for i, (chunk, score) in enumerate(retrieved):
        context_parts.append(
            f"[Excerpt {i+1} | Source: {chunk.doc_type} — {chunk.doc_title} | Relevance: {score:.0%}]\n"
            f"{chunk.content}"
        )
    context = "\n\n---\n\n".join(context_parts)

    user_message = (
        f"Answer the following question using ONLY the provided medical document excerpts.\n\n"
        f"Question: {question}\n\n"
        f"Document Excerpts:\n{context}"
    )

    # Step 3: Call Claude
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}]
    )

    answer = response.content[0].text
    tokens_used = (response.usage.input_tokens or 0) + (response.usage.output_tokens or 0)
    latency_ms = (time.time() - start_time) * 1000

    # Update metrics
    similarity_scores = [s for _, s in retrieved]
    total_queries += 1
    cumulative_latency_ms += latency_ms
    cumulative_similarity += sum(similarity_scores)
    total_similarity_count += len(similarity_scores)

    return {
        "answer": answer,
        "sources": [
            {"doc_type": c.doc_type, "doc_title": c.doc_title, "similarity_score": round(s, 4)}
            for c, s in retrieved
        ],
        "retrieved_chunks": [c.content for c, _ in retrieved],
        "similarity_scores": [round(s, 4) for _, s in retrieved],
        "latency_ms": round(latency_ms, 1),
        "tokens_used": tokens_used
    }


def get_metrics() -> Dict:
    """Return live pipeline metrics — only what actually exists."""
    avg_sim = (cumulative_similarity / total_similarity_count) if total_similarity_count > 0 else 0.0
    avg_lat = (cumulative_latency_ms / total_queries) if total_queries > 0 else 0.0

    return {
        "total_queries": total_queries,
        "documents_indexed": len(documents),
        "chunks_indexed": len(chunks),
        "avg_similarity_score": round(avg_sim, 4),
        "avg_latency_ms": round(avg_lat, 1),
        "embedding_model": "all-MiniLM-L6-v2",
        "vector_db_type": "FAISS IndexFlatIP",
        "embedding_dim": index.d if index is not None else 0
    }


def get_documents() -> List[Dict]:
    """Return list of indexed documents with chunk counts."""
    doc_chunks = {}
    for c in chunks:
        if c.doc_id not in doc_chunks:
            doc_chunks[c.doc_id] = {"id": c.doc_id, "type": c.doc_type, "title": c.doc_title, "chunk_count": 0}
        doc_chunks[c.doc_id]["chunk_count"] += 1
    return list(doc_chunks.values())
