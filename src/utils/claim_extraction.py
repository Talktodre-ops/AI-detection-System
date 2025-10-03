"""
Shared utilities for claim extraction and fact-checking.
Extracted from duplicated code in fact_checker.py and mixed_content_app.py
"""
from typing import List

try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None

from textblob import TextBlob

FACT_ENTITY_TYPES = {
    "PERSON", "ORG", "GPE", "LOC", "NORP", "FAC", "PRODUCT", "EVENT",
    "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY",
    "QUANTITY", "ORDINAL", "CARDINAL"
}


def _score_sentence(sent) -> float:
    """
    Calculate heuristic 'claiminess' score for a sentence.

    Args:
        sent: spaCy Span object representing a sentence

    Returns:
        float: Claim score (higher means more likely to be a factual claim)
    """
    toks = [t for t in sent if not t.is_space]
    has_verb = any(t.pos_ in ("VERB", "AUX") for t in toks)
    has_cop = any(t.lemma_ in ("be", "have") for t in toks)
    ents = [e for e in sent.ents if e.label_ in FACT_ENTITY_TYPES]
    has_digit = any(ch.isdigit() for ch in sent.text)
    is_question = "?" in sent.text
    length = len(toks)

    score = 0
    if has_verb:
        score += 1
    if has_cop:
        score += 0.5
    score += min(2, len(ents)) * 0.8
    if has_digit:
        score += 0.5
    if 6 <= length <= 40:
        score += 0.5
    if is_question:
        score -= 2
    return score


def extract_claims(text: str, max_claims: int = 5) -> List[str]:
    """
    Extract factual claims from text using NLP techniques.

    Args:
        text (str): Input text to analyze
        max_claims (int): Maximum number of claims to extract

    Returns:
        List[str]: List of extracted claims
    """
    text = (text or "").strip()
    if not text:
        return []

    # Prefer spaCy sentence-based claim detection
    if _NLP is not None:
        doc = _NLP(text)
        candidates = []
        for sent in doc.sents:
            s = sent.text.strip()
            if not s:
                continue
            score = _score_sentence(sent)
            if score > 1:  # threshold for likely factual statement
                candidates.append((score, s))

        # Dedup normalized sentences and keep top-k by score
        seen = set()
        claims = []
        for _, s in sorted(candidates, key=lambda x: x[0], reverse=True):
            key = " ".join(s.lower().split())
            if key not in seen:
                seen.add(key)
                claims.append(s[:240])
            if len(claims) >= max_claims:
                break

        if claims:
            return claims

    # Fallback to TextBlob noun phrases + sentence fallback
    blob = TextBlob(text)
    cands = list({p.strip() for p in blob.noun_phrases if len(p.split()) > 1})
    if len(cands) < 2:
        cands.extend([s.strip() for s in text.split(".") if len(s.split()) > 3][:3])

    seen = set()
    claims = []
    for c in cands:
        c = c[:240]
        k = c.lower()
        if k not in seen:
            seen.add(k)
            claims.append(c)
    return claims[:max_claims]


def ensure_nltk_corpora():
    """Ensure required NLTK corpora are downloaded."""
    try:
        from textblob.en import np_extractors  # triggers corpora check
        TextBlob("test").noun_phrases  # warm-up
    except Exception:
        import nltk
        for p in ['brown', 'punkt', 'averaged_perceptron_tagger', 'wordnet']:
            nltk.download(p, quiet=True)
