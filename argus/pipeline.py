"""The system under test: a small RAG pipeline whose knobs Argus optimizes.

Retrieval is a deliberately SIMULATED loss surface (lexical BM25 + a tf-idf "semantic" view +
rerank + MMR) — fidelity of the retriever is not the point; it only needs real knob->behaviour
dynamics and to emit a realistic retrieval *log*. Synthesis, by contrast, is a REAL LLM call:
the `synthesis_prompt` knob is its system prompt, so it genuinely shapes the answer.

What the optimizer later gets to see is symptom-level only (see optimizer.build_logs): the
question, the answer, the grader's score, and the retrieval log — never where a "correct" fact
lives or which knob to turn.

    chunk -> embed -> retrieve(bm25/vector/hybrid) -> rerank(+floor) -> mmr-select -> synth(LLM)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .cache import StageCache
from .dataset import DOCS, SYNONYMS, Question
from .llm import model_for
from .params import Params, Stage

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)?|\$[0-9,]+")
_STOPWORDS = {
    "the", "a", "an", "of", "is", "are", "to", "for", "and", "or", "in", "on", "at",
    "what", "how", "who", "does", "do", "did", "with", "by", "from", "this", "that",
    "during", "first", "additional", "required", "require",
}


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def expand(tokens: list[str]) -> list[str]:
    """Token list augmented with synonyms — the 'semantic' (vector) view of text."""
    out = list(tokens)
    for t in tokens:
        out.extend(SYNONYMS.get(t, ()))
    return out


def salient(query: str) -> list[str]:
    """Content terms of a query (stopwords removed) — used for rerank coverage."""
    return [t for t in tokenize(query) if t not in _STOPWORDS]


# --- Stage 1: chunk ---------------------------------------------------------------

@dataclass(frozen=True)
class Chunk:
    id: str
    doc: str
    text: str


def _chunk_doc(doc_id: str, text: str, max_chars: int, overlap: int) -> list[Chunk]:
    if len(text) <= max_chars:
        return [Chunk(f"{doc_id}#0", doc_id, text)]
    chunks: list[Chunk] = []
    start, i = 0, 0
    step = max(1, max_chars - overlap)
    while start < len(text):
        chunks.append(Chunk(f"{doc_id}#{i}", doc_id, text[start : start + max_chars]))
        start += step
        i += 1
    return chunks


def build_chunks(params: Params) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc_id, text in DOCS.items():
        chunks.extend(_chunk_doc(doc_id, text, params.chunk_chars, params.chunk_overlap))
    return chunks


# --- Stage 2: embed / index -------------------------------------------------------

@dataclass
class Index:
    chunks: list[Chunk]
    tf: list[dict[str, int]]
    df: dict[str, int]
    doc_len: list[int]
    avgdl: float
    vec: list[dict[str, float]]


def _tfidf_vector(tokens: list[str], df: dict[str, int], n_docs: int) -> dict[str, float]:
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    vec: dict[str, float] = {}
    for t, c in counts.items():
        idf = math.log(1 + n_docs / (1 + df.get(t, 0)))
        vec[t] = (c / len(tokens)) * idf
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {t: v / norm for t, v in vec.items()}


def build_index(chunks: list[Chunk]) -> Index:
    tf: list[dict[str, int]] = []
    df: dict[str, int] = {}
    doc_len: list[int] = []
    exp_tokens: list[list[str]] = []
    exp_df: dict[str, int] = {}
    for ch in chunks:
        toks = tokenize(ch.text)
        counts: dict[str, int] = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
        tf.append(counts)
        doc_len.append(len(toks))
        for t in counts:
            df[t] = df.get(t, 0) + 1
        et = expand(toks)
        exp_tokens.append(et)
        for t in set(et):
            exp_df[t] = exp_df.get(t, 0) + 1
    n = len(chunks)
    avgdl = (sum(doc_len) / n) if n else 0.0
    vec = [_tfidf_vector(et, exp_df, n) for et in exp_tokens]
    return Index(chunks, tf, df, doc_len, avgdl, vec)


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(t, 0.0) for t, w in a.items())


def _bm25_scores(index: Index, query_tokens: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    n = len(index.chunks)
    scores = [0.0] * n
    for t in set(query_tokens):
        df = index.df.get(t, 0)
        if not df:
            continue
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        for i in range(n):
            f = index.tf[i].get(t, 0)
            if not f:
                continue
            denom = f + k1 * (1 - b + b * index.doc_len[i] / (index.avgdl or 1.0))
            scores[i] += idf * (f * (k1 + 1)) / denom
    return scores


def _minmax(scores: list[float]) -> list[float]:
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


# --- Stage 3: retrieve ------------------------------------------------------------

def retrieve(index: Index, query: str, params: Params) -> list[tuple[int, float]]:
    q_tokens = tokenize(query)
    q_vec = _tfidf_vector(expand(q_tokens), {t: 1 for t in set(expand(q_tokens))}, 2)
    bm25 = _minmax(_bm25_scores(index, q_tokens))
    vector = _minmax([_cosine(q_vec, v) for v in index.vec])
    if params.retrieval_mode == "bm25":
        fused = bm25
    elif params.retrieval_mode == "vector":
        fused = vector
    else:
        a = params.hybrid_alpha
        fused = [a * vector[i] + (1 - a) * bm25[i] for i in range(len(index.chunks))]
    ranked = sorted(range(len(index.chunks)), key=lambda i: fused[i], reverse=True)
    return [(i, fused[i]) for i in ranked[: params.candidate_count]]


# --- Stage 4: rerank (cross-encoder stand-in + relevance floor) -------------------

def _rerank_score(index: Index, query: str, q_vec: dict[str, float], idx: int) -> float:
    sem = _cosine(q_vec, index.vec[idx])
    terms = salient(query)
    chunk_tokens = set(tokenize(index.chunks[idx].text))
    coverage = (sum(1 for t in terms if t in chunk_tokens) / len(terms)) if terms else 0.0
    return 0.4 * min(1.0, sem * 3.0) + 0.6 * coverage


@dataclass
class Reranked:
    idx: int
    score: float
    kept: bool


def rerank(index: Index, query: str, candidates: list[tuple[int, float]], params: Params) -> list[Reranked]:
    q_tokens = expand(tokenize(query))
    q_vec = _tfidf_vector(q_tokens, {t: 1 for t in set(q_tokens)}, 2)
    scored = [Reranked(i, _rerank_score(index, query, q_vec, i), True) for i, _ in candidates]
    for r in scored:
        r.kept = r.score >= params.rerank_min_score
    scored.sort(key=lambda r: r.score, reverse=True)
    return scored


# --- Stage 5: MMR diversity select ------------------------------------------------

@dataclass
class Selection:
    selected: list[int]
    collapsed: list[int]


def select(index: Index, reranked: list[Reranked], params: Params) -> Selection:
    pool = [r for r in reranked if r.kept]
    budget = params.rerank_result_count
    if not params.mmr_enabled:
        return Selection([r.idx for r in pool[:budget]], [])
    selected: list[int] = []
    collapsed: list[int] = []
    for r in pool:
        if len(selected) >= budget:
            break
        if any(_cosine(index.vec[r.idx], index.vec[s]) >= params.mmr_sim_threshold for s in selected):
            collapsed.append(r.idx)
            continue
        if selected:
            max_sim = max(_cosine(index.vec[r.idx], index.vec[s]) for s in selected)
            if params.mmr_lambda * r.score - (1 - params.mmr_lambda) * max_sim <= 0:
                collapsed.append(r.idx)
                continue
        selected.append(r.idx)
    return Selection(selected, collapsed)


# --- Stage 6: synthesize (REAL LLM) -----------------------------------------------

@dataclass
class Synthesis:
    answer_text: str
    snippets: list[tuple[str, str]]   # (doc_id, context text the model was shown)


def _build_context(index: Index, selection: Selection, params: Params) -> list[tuple[str, str]]:
    """Per-doc context slice (doc_char_limit), preserving selection order, one entry per doc."""
    order: list[str] = []
    by_doc: dict[str, str] = {}
    for i in selection.selected:
        ch = index.chunks[i]
        if ch.doc not in by_doc:
            by_doc[ch.doc] = ""
            order.append(ch.doc)
        if len(by_doc[ch.doc]) < params.doc_char_limit:
            by_doc[ch.doc] = (by_doc[ch.doc] + ch.text)[: params.doc_char_limit]
    return [(doc, by_doc[doc]) for doc in order]


def synthesize(index: Index, question: Question, selection: Selection, params: Params, backend) -> Synthesis:
    snippets = _build_context(index, selection, params)
    if snippets:
        context = "\n\n".join(f"[source: {doc}]\n{txt}" for doc, txt in snippets)
    else:
        context = "(no documents were retrieved)"
    # The synthesis_prompt knob IS the system prompt — so it genuinely shapes the answer.
    user = f"Question: {question.query}\n\nRetrieved context:\n{context}\n\nAnswer:"
    answer = backend.complete_text(params.synthesis_prompt, user,
                                   model=model_for("synthesis"), max_tokens=400)
    return Synthesis(answer.strip(), snippets)


# --- Orchestration ----------------------------------------------------------------

@dataclass
class QuestionTrace:
    question: Question
    retrieval_mode: str
    retrieved: list[tuple[str, float]]        # (doc, fused score) candidate pool
    reranked: list[tuple[str, float, bool]]   # (doc, rerank score, kept)
    selected_snippets: list[tuple[str, str]]  # (doc, context the synth model saw)
    collapsed_docs: list[str]
    answer_text: str


def run_question(index: Index, question: Question, params: Params, cache: StageCache, backend) -> QuestionTrace:
    sig_r = params.stage_signature(Stage.RETRIEVE)
    candidates = cache.get_or_compute("retrieve", sig_r, question.id,
                                      lambda: retrieve(index, question.query, params))
    sig_rr = params.stage_signature(Stage.RERANK)
    reranked = cache.get_or_compute("rerank", sig_rr, question.id,
                                    lambda: rerank(index, question.query, candidates, params))
    sig_m = params.stage_signature(Stage.MMR)
    selection = cache.get_or_compute("mmr", sig_m, question.id,
                                     lambda: select(index, reranked, params))
    sig_s = params.stage_signature(Stage.SYNTH)
    synthesis: Synthesis = cache.get_or_compute("synth", sig_s, question.id,
                                                lambda: synthesize(index, question, selection, params, backend))

    def _doc(i: int) -> str:
        return index.chunks[i].doc

    return QuestionTrace(
        question=question,
        retrieval_mode=params.retrieval_mode,
        retrieved=[(_doc(i), round(s, 3)) for i, s in candidates],
        reranked=[(_doc(r.idx), round(r.score, 3), r.kept) for r in reranked],
        selected_snippets=synthesis.snippets,
        collapsed_docs=[_doc(i) for i in selection.collapsed],
        answer_text=synthesis.answer_text,
    )


def run_pipeline(params: Params, questions: tuple[Question, ...], cache: StageCache, backend) -> dict[str, QuestionTrace]:
    sig_c = params.stage_signature(Stage.CHUNK)
    chunks = cache.get_or_compute("chunk", sig_c, "*", lambda: build_chunks(params))
    sig_e = params.stage_signature(Stage.EMBED)
    index = cache.get_or_compute("embed", sig_e, "*", lambda: build_index(chunks))
    return {q.id: run_question(index, q, params, cache, backend) for q in questions}
