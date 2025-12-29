from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None

from config.settings import BASE_DIR
from utils.security import _tokenize_text

MOVIE_DATA_PATH = BASE_DIR / "movie.json"
STYLE_HINT_TAGS = {
    "bright": ["冒險", "療癒", "溫暖", "夢想", "勇氣", "希望"],
    "steady": ["日常", "平穩", "家庭", "陪伴", "放鬆", "安定"],
    "healer": ["療癒", "修復", "心靈", "安定", "重建", "擁抱"],
}

KEYWORD_SYNONYMS = {
    "壓力": ["療癒", "放鬆", "溫柔"],
    "疲憊": ["療癒", "溫暖"],
    "失戀": ["愛情", "浪漫", "療癒"],
    "失戀痛苦": ["浪漫", "療癒", "溫柔"],
    "孤單": ["陪伴", "友情", "家庭"],
    "家庭": ["家庭", "親情"],
    "青春": ["成長", "夢想", "懷舊"],
    "青春流逝": ["懷舊", "夢想", "成長"],
    "重生": ["夢想", "勇氣"],
    "恐懼": ["勇氣", "冒險"],
    "焦慮": ["療癒", "放鬆"],
    "悲傷": ["療癒", "溫暖"],
    "委屈": ["療癒", "陪伴"],
    "期待": ["浪漫", "冒險"],
    "友情": ["友情", "家庭"],
    "療癒": ["療癒", "溫暖"],
    "冒險": ["冒險", "勇氣"],
    "成長": ["成長", "夢想"],
    "愛情": ["浪漫", "愛情"],
    "自我價值": ["勵志", "成長"],
    "身體壓力": ["療癒", "放鬆"],
    "情緒困惑": ["療癒", "心靈"],
}

DEFAULT_SCENARIOS = {
    "bright": ["冒險", "夢想", "勇氣", "成長"],
    "steady": ["家庭", "友情", "溫暖", "親情"],
    "healer": ["療癒", "溫柔", "陪伴", "浪漫"],
}


def _load_movie_recommendations() -> list[dict]:
    if not MOVIE_DATA_PATH.exists():
        logging.warning("Movie knowledge base not found at %s", MOVIE_DATA_PATH)
        return []
    try:
        with MOVIE_DATA_PATH.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        logging.warning("Failed to load movie knowledge base: %s", exc)
        return []
    entries = raw.get("movie_recommendations") if isinstance(raw, dict) else raw
    normalized = []
    for entry in entries or []:
        title = str(entry.get("title") or "").strip()
        if not title:
            continue
        normalized.append(
            {
                "title": title,
                "english_title": str(entry.get("english_title") or "").strip(),
                "tags": [
                    str(tag).strip()
                    for tag in entry.get("tags", [])
                    if isinstance(tag, str) and tag.strip()
                ],
                "reason": str(entry.get("reason") or "").strip(),
            }
        )
    logging.debug("Loaded %d movie recommendations", len(normalized))
    return normalized


MOVIE_KNOWLEDGE_BASE = _load_movie_recommendations()
_SENTENCE_EMBEDDER = None
MOVIE_EMBEDDING_INDEX: list[tuple[np.ndarray, dict]] = []


def _get_sentence_embedder():
    global _SENTENCE_EMBEDDER
    if SentenceTransformer is None:
        logging.warning(
            "SentenceTransformer not available; semantic movie recommendations disabled."
        )
        return None
    if _SENTENCE_EMBEDDER is None:
        try:
            _SENTENCE_EMBEDDER = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            logging.debug("SentenceTransformer model loaded for semantic movie search.")
        except Exception as exc:
            logging.error("Failed to load sentence-transformers model: %s", exc)
            _SENTENCE_EMBEDDER = False
            return None
    if _SENTENCE_EMBEDDER is False:
        return None
    return _SENTENCE_EMBEDDER


def _rebuild_movie_embedding_index():
    MOVIE_EMBEDDING_INDEX.clear()
    if not MOVIE_KNOWLEDGE_BASE:
        return
    embedder = _get_sentence_embedder()
    if not embedder:
        return
    texts = []
    metadata = []
    for movie in MOVIE_KNOWLEDGE_BASE:
        parts = [
            movie.get("title") or "",
            movie.get("english_title") or "",
            " ".join(movie.get("tags") or []),
            movie.get("reason") or "",
        ]
        texts.append(" ".join(part for part in parts if part))
        metadata.append(movie)
    if not texts:
        return
    try:
        embeddings = embedder.encode(
            texts, normalize_embeddings=True, batch_size=32
        )
    except Exception as exc:
        logging.error("Failed to encode movie embeddings: %s", exc)
        return
    for vec, movie in zip(embeddings, metadata):
        MOVIE_EMBEDDING_INDEX.append((np.array(vec, dtype=np.float32), movie))
    logging.debug(
        "Movie embedding index built with %d entries.", len(MOVIE_EMBEDDING_INDEX)
    )


_rebuild_movie_embedding_index()


def semantic_movie_recommendations(
    psychology: dict | None, style_key: str, top_n: int = 2
) -> list[dict[str, str]]:
    if not MOVIE_EMBEDDING_INDEX:
        _rebuild_movie_embedding_index()
    if not MOVIE_EMBEDDING_INDEX:
        return []

    embedder = _get_sentence_embedder()
    if not embedder:
        return []

    psychology = psychology or {}
    components = []
    summary = psychology.get("summary") or psychology.get("description") or ""
    if summary:
        components.append(summary)
    keywords = psychology.get("keywords") or []
    if keywords:
        components.append("、".join(keywords))
    scenario_tags = DEFAULT_SCENARIOS.get(style_key) or STYLE_HINT_TAGS.get(
        style_key
    ) or []
    if scenario_tags:
        components.append("希望獲得：" + "、".join(scenario_tags))

    if not components:
        return []

    query_text = " ".join(components)
    try:
        query_vec = embedder.encode(
            [query_text], normalize_embeddings=True
        )[0]
    except Exception as exc:
        logging.error("Failed to encode movie query embedding: %s", exc)
        return []

    scored = []
    for vec, movie in MOVIE_EMBEDDING_INDEX:
        score = float(np.dot(query_vec, vec))
        scored.append((score, movie))

    scored.sort(key=lambda item: item[0], reverse=True)

    if scored:
        logging.debug(
            "Semantic movie RAG top candidates: %s",
            [
                {"title": movie["title"], "score": round(score, 3)}
                for score, movie in scored[:5]
            ],
        )

    if not scored:
        return []

    top_candidates = scored[:top_n]
    scores = [s for s, _ in top_candidates]
    exp_scores = [math.exp(s) for s in scores]
    total = sum(exp_scores)
    weights = [e / total for e in exp_scores]
    logging.debug(f"Semantic top {top_n} weights: {weights}")

    selected_item = random.choices(top_candidates, weights=weights, k=1)[0]
    selected = [
        {
            "title": selected_item[1]["title"],
            "reason": selected_item[1].get("reason", ""),
        }
    ]
    logging.debug(
        "Semantic weighted selection: %s", selected_item[1]["title"]
    )

    return selected


def keyword_movie_recommendations(
    psychology: dict | None, style_key: str, top_n: int = 2
) -> list[dict[str, str]]:
    if not MOVIE_KNOWLEDGE_BASE:
        return []
    user_tokens: set[str] = set()
    psychology = psychology or {}
    for kw in psychology.get("keywords") or []:
        user_tokens.update(_tokenize_text(kw))
    summary = psychology.get("summary") or psychology.get("description") or ""
    user_tokens.update(_tokenize_text(summary))
    mood_score = psychology.get("combined_score") or psychology.get("mind_score")
    health_score = psychology.get("health_score")
    if mood_score is not None:
        mood_score = float(mood_score)
        if mood_score >= 80:
            user_tokens.update(["鼓舞", "冒險", "夢想"])
        elif mood_score <= 50:
            user_tokens.update(["療癒", "溫柔", "陪伴"])
        else:
            user_tokens.update(["日常", "放鬆"])
    if health_score is not None:
        health_score = float(health_score)
        if health_score < 60:
            user_tokens.update(["重生", "修復", "心靈"])
    enriched = set(user_tokens)
    for token in list(user_tokens):
        for key, synonyms in KEYWORD_SYNONYMS.items():
            if key in token:
                enriched.update(syn.lower() for syn in synonyms)
    user_tokens = {tok.lower() for tok in enriched if tok}
    style_tokens = {tag.lower() for tag in STYLE_HINT_TAGS.get(style_key, [])}

    def matches_tokens(movie, token_set):
        if not token_set:
            return []
        movie_tags = [tag.lower() for tag in movie.get("tags", [])]
        matched = [tag for tag in movie_tags if any(token in tag for token in token_set)]
        return matched

    def filter_pool(token_set):
        if not token_set:
            return []
        return [
            movie
            for movie in MOVIE_KNOWLEDGE_BASE
            if matches_tokens(movie, token_set)
        ]

    candidate_pool = filter_pool(user_tokens)
    tokens = user_tokens.copy()
    if not candidate_pool:
        candidate_pool = filter_pool(style_tokens)
        tokens = style_tokens.copy()
    if not candidate_pool:
        fallback_tokens = {
            tag.lower()
            for tag in DEFAULT_SCENARIOS.get(style_key, [])
        }
        tokens = fallback_tokens or set()
        candidate_pool = filter_pool(tokens) if tokens else MOVIE_KNOWLEDGE_BASE.copy()
    if not candidate_pool:
        candidate_pool = MOVIE_KNOWLEDGE_BASE.copy()
        tokens = set()

    results = []
    for movie in candidate_pool:
        tags = [tag.lower() for tag in movie.get("tags", [])]
        reason_text = movie.get("reason", "").lower()
        if not tokens:
            score = 1.0
        else:
            score = 0.0
            for tag in tags:
                if any(token in tag for token in tokens):
                    score += 2.0
            for token in tokens:
                if token and token in reason_text:
                    score += 1.0
        for tag in tags:
            for token in tokens:
                if token and token == tag:
                    score += 1.0
                    break
        if score <= 0:
            score = 0.1
        results.append(
            (
                score,
                {"title": movie["title"], "reason": movie.get("reason", "")},
            )
        )
    results.sort(key=lambda item: item[0], reverse=True)

    if results:
        top_k = 5
        candidates = results[:top_k]
        logging.debug(
            "Movie RAG top 5 candidates for style=%s: %s",
            style_key,
            [
                {"title": entry[1]["title"], "score": round(entry[0], 2)}
                for entry in candidates
            ],
        )

        scores = [item[0] for item in candidates]
        exp_scores = [math.exp(s) for s in scores]
        total = sum(exp_scores)
        weights = [e / total for e in exp_scores]
        logging.debug(f"Weights: {weights}")

        selected_item = random.choices(candidates, weights=weights, k=1)[0]
        selected = [selected_item[1]]

        logging.debug(
            "Movie RAG weighted selection for style=%s: %s",
            style_key,
            selected_item[1]["title"],
        )
    else:
        selected = []

    return selected


def rag_movie_recommendations(
    psychology: dict | None, style_key: str, top_n: int = 2
) -> list[dict[str, str]]:
    semantic = semantic_movie_recommendations(psychology, style_key, top_n=5)
    if semantic:
        return semantic
    return keyword_movie_recommendations(psychology, style_key, top_n)
