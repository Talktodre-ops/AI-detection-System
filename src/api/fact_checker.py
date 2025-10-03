import os
import math
import time
from typing import List, Dict, Optional

import tldextract
import httpx
from duckduckgo_search import DDGS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from src.utils.claim_extraction import extract_claims, ensure_nltk_corpora

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")

NLI_MODEL_NAME = os.environ.get("NLI_MODEL_NAME", "roberta-large-mnli")
_device = "cuda" if torch.cuda.is_available() else "cpu"

_tokenizer = None
_nli = None

def _load_nli():
	global _tokenizer, _nli
	if _tokenizer is None or _nli is None:
		_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
		_nli = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME).to(_device)
	return _tokenizer, _nli

ensure_nltk_corpora()

async def google_search(query: str, k: int = 5) -> List[Dict]:
	if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
		return []
	url = "https://www.googleapis.com/customsearch/v1"
	params = {"q": query, "cx": GOOGLE_CSE_ID, "key": GOOGLE_API_KEY, "num": min(k, 10)}
	try:
		async with httpx.AsyncClient(timeout=15) as client:
			r = await client.get(url, params=params)
			r.raise_for_status()
			items = r.json().get("items", []) or []
			return [{"title": it.get("title"), "link": it.get("link"), "snippet": it.get("snippet")} for it in items]
	except httpx.HTTPError:
		# any quota/auth/network error → trigger fallback
		return []

def ddg_search(query: str, k: int = 5) -> List[Dict]:
	results = []
	with DDGS() as ddgs:
		for i, r in enumerate(ddgs.text(query, max_results=k)):
			results.append({"title": r.get("title"), "link": r.get("href"), "snippet": r.get("body")})
			if i + 1 >= k:
				break
	return results

async def wiki_search(query: str, k: int = 3) -> List[Dict]:
	# Simple Wikipedia API
	search_url = "https://en.wikipedia.org/w/api.php"
	params = {"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": k}
	async with httpx.AsyncClient(timeout=15) as client:
		r = await client.get(search_url, params=params)
		r.raise_for_status()
		data = r.json().get("query", {}).get("search", []) or []
		out = []
		for item in data:
			title = item.get("title")
			out.append({
				"title": title,
				"link": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
				"snippet": item.get("snippet")
			})
		return out

def domain_score(url: str) -> float:
	domain = tldextract.extract(url).registered_domain.lower()
	if domain.endswith(".gov") or domain.endswith(".gov.ng") or domain.endswith(".edu"):
		return 1.0
	if "wikipedia.org" in domain:
		return 0.9
	major = {"reuters.com", "bbc.com", "apnews.com", "nature.com", "who.int", "un.org", "nih.gov"}
	if any(x in domain for x in major):
		return 0.85
	return 0.6

async def fetch_text(url: str) -> Optional[str]:
	try:
		import trafilatura  # lazy import
	except ImportError:
		return None
	try:
		downloaded = trafilatura.fetch_url(url)
		if not downloaded:
			return None
		# Extract more content for better NLI analysis
		text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
		if text and len(text) > 100:  # Only use if we got substantial content
			return text
		return None
	except Exception:
		return None

def nli_stance(claim: str, evidence: str) -> Dict:
	tokenizer, model = _load_nli()
	pair = tokenizer(claim, evidence, truncation=True, padding=True, return_tensors="pt").to(_device)
	with torch.no_grad():
		logits = model(**pair).logits.softmax(-1).squeeze().tolist()
	# For MNLI label order: [contradiction, neutral, entailment]
	return {"entailment": logits[2], "contradiction": logits[0], "neutral": logits[1]}

def verdict_from_scores(scores: Dict, thresh=0.3) -> str:  # Lower threshold from 0.55 to 0.3
	if scores["entailment"] >= thresh and scores["entailment"] > scores["contradiction"]:
		return "supported"
	if scores["contradiction"] >= thresh and scores["contradiction"] > scores["entailment"]:
		return "refuted"
	return "unclear"

async def verify_claims(text: str) -> Dict:
	start = time.time()
	claims = extract_claims(text)
	if not claims:
		return {"status": "No claims found", "claims": []}

	out = []
	for claim in claims:
		# retrieval
		results = []
		g = await google_search(claim, k=5) if GOOGLE_API_KEY and GOOGLE_CSE_ID else []
		if not g:
			w = await wiki_search(claim, k=3)
			ddg = ddg_search(claim, k=5)
			results = w + ddg
		else:
			results = g

		# analyze top-k
		citations = []
		best_support = 0.0
		best_refute = 0.0
		for rank, r in enumerate(results[:6], start=1):
			url = r.get("link") or r.get("url")
			if not url:
				continue
			main_text = await fetch_text(url)
			context = main_text or r.get("snippet") or ""
			if not context:
				continue
			scores = nli_stance(claim, context)
			ds = domain_score(url)
			retrieval_weight = 1.0 / math.log2(rank + 1.2)

			# Use the raw NLI scores more directly
			final_score = max(scores["entailment"], scores["contradiction"]) * ds * retrieval_weight

			# Update best scores with better weighting
			best_support = max(best_support, scores["entailment"] * ds * 0.8)  # Weight entailment higher
			best_refute = max(best_refute, scores["contradiction"] * ds * 0.8)
			citations.append({
				"title": r.get("title"),
				"url": url,
				"snippet": (context[:240] + "...") if len(context) > 240 else context,
				"stance": "entailment" if scores["entailment"] > scores["contradiction"] else "contradiction" if scores["contradiction"] > scores["neutral"] else "neutral",
				"score": round(final_score, 4)
			})

		# aggregate
		raw = {"entailment": best_support, "contradiction": best_refute, "neutral": 0.0}
		verdict = verdict_from_scores(raw, thresh=0.3)
		confidence = round(max(best_support, best_refute), 3)
		citations = sorted(citations, key=lambda x: x["score"], reverse=True)[:3]

		out.append({
			"claim": claim,
			"verdict": verdict,
			"confidence": confidence,
			"citations": citations
		})

	return {"status": "Claims verified", "elapsed_sec": round(time.time() - start, 2), "claims": out}