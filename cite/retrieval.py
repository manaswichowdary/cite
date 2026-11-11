"""Finding the passages of a source page that a claim is about.

The checker shows the judge only a few passages from each cited page, so this step decides what
evidence the judge can see. Three ways to do it:

words       pick passages that share the most words with the claim (fast, no model, misses paraphrase)
embeddings  embed every passage and the claim, pick the closest by cosine similarity (finds paraphrase)
hybrid      the closest passages by embedding, plus the best word matches, without duplicates

Embeddings for each page are saved on disk, so a page is only embedded once. That small store of
vectors, searched by cosine similarity, is the vector index.
"""

from __future__ import annotations

import hashlib
import re
import threading
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
STOPWORDS = {"that", "this", "with", "from", "have", "been", "which", "their", "there", "these",
             "those", "they", "them", "then", "than", "into", "also", "such", "when", "what",
             "where", "will", "would", "could", "should", "about", "other", "more", "most",
             "some", "only", "very", "over", "your", "like", "used", "using", "can", "and",
             "the", "for", "are", "was", "were", "its", "not", "but", "you", "any", "all"}


def split_chunks(text: str, size: int = 500, overlap: int = 100) -> list[str]:
    """Cut page text into passages of about `size` characters, keeping line breaks where possible."""
    chunks, buf = [], ""
    for piece in re.split(r"\n+", text):
        piece = piece.strip()
        if not piece:
            continue
        if buf and len(buf) + len(piece) + 1 > size:
            chunks.append(buf)
            buf = ""
        buf = f"{buf} {piece}".strip()
        while len(buf) > size:
            chunks.append(buf[:size])
            buf = buf[size - overlap:]
    if buf:
        chunks.append(buf)
    return chunks


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 4 and w not in STOPWORDS}


def _clean(claim: str) -> str:
    return CITE_RE.sub("", claim).strip()


class Retriever(Protocol):
    name: str

    def passages(self, claim: str, page_text: str) -> list[str]: ...


class WordRetriever:
    """The original method: passages that share at least two content words with the claim."""

    name = "words"

    def __init__(self, top: int = 8):
        self.top = top

    def passages(self, claim: str, page_text: str) -> list[str]:
        claim_words = _words(_clean(claim))
        if not claim_words:
            return []
        need = min(2, len(claim_words))
        scored = [(len(claim_words & _words(c)), c) for c in split_chunks(page_text)]
        scored = [s for s in scored if s[0] >= need]
        scored.sort(key=lambda s: -s[0])
        return [c for _, c in scored[:self.top]]


class Embedder(Protocol):
    model_name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Runs a small sentence-transformers model on this machine. Loaded on first use."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()     # sentences are checked in parallel threads

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        with self._lock:
            return self._encode(texts)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as err:   # optional dependency, see requirements-rag.txt
                raise RuntimeError("Embedding retrieval needs: pip install -r requirements-rag.txt") from err
            self._model = SentenceTransformer(self.model_name)
        vectors = self._model.encode(list(texts), batch_size=32, convert_to_numpy=True,
                                     normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)


def _unit(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


class EmbeddingRetriever:
    """Picks the passages closest in meaning to the claim, with page vectors cached on disk."""

    name = "embeddings"

    def __init__(self, embedder: Embedder, cache_dir: str | Path = "data/cache/vectors",
                 top: int = 8, min_score: float = 0.2):
        self.embedder, self.top, self.min_score = embedder, top, min_score
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pages_embedded = 0     # pages that had to be embedded (not served from the cache)
        self._lock = threading.Lock()

    def _page_vectors(self, chunks: list[str]) -> np.ndarray:
        with self._lock:
            return self._page_vectors_locked(chunks)

    def _page_vectors_locked(self, chunks: list[str]) -> np.ndarray:
        key = hashlib.sha256((self.embedder.model_name + "\n\x00".join(chunks)).encode()).hexdigest()
        path = self.dir / f"{key}.npy"
        if path.exists():
            return np.load(path)
        vectors = _unit(np.asarray(self.embedder.encode(chunks), dtype=np.float32))
        np.save(path, vectors)
        self.pages_embedded += 1
        return vectors

    def ranked(self, claim: str, page_text: str) -> list[tuple[float, str]]:
        chunks = split_chunks(page_text)
        if not chunks or not _clean(claim):
            return []
        page = self._page_vectors(chunks)
        query = _unit(np.asarray(self.embedder.encode([_clean(claim)]), dtype=np.float32))[0]
        scores = page @ query
        order = np.argsort(-scores)
        return [(float(scores[i]), chunks[i]) for i in order if scores[i] >= self.min_score]

    def passages(self, claim: str, page_text: str) -> list[str]:
        return [c for _, c in self.ranked(claim, page_text)[:self.top]]


class HybridRetriever:
    """Closest passages by meaning first, then the best word matches, without repeats."""

    name = "hybrid"

    def __init__(self, embeddings: EmbeddingRetriever, words: WordRetriever, top: int = 8, word_slots: int = 2):
        self.embeddings, self.words, self.top, self.word_slots = embeddings, words, top, word_slots

    def passages(self, claim: str, page_text: str) -> list[str]:
        by_meaning = self.embeddings.passages(claim, page_text)[:self.top - self.word_slots]
        out = list(by_meaning)
        for c in self.words.passages(claim, page_text):
            if len(out) >= self.top:
                break
            if c not in out:
                out.append(c)
        return out


def make_retriever(name: str, embedder: Embedder | None = None,
                   cache_dir: str | Path = "data/cache/vectors") -> Retriever:
    if name == "words":
        return WordRetriever()
    embedder = embedder or SentenceTransformerEmbedder()
    emb = EmbeddingRetriever(embedder, cache_dir)
    if name == "embeddings":
        return emb
    if name == "hybrid":
        return HybridRetriever(emb, WordRetriever())
    raise ValueError(f"unknown retriever {name!r}: use words, embeddings or hybrid")
