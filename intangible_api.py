import os
import logging
import random
import re
import json
import openai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("intangible_api")
logger.setLevel(logging.DEBUG)

OPENAI_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
MAX_PITCH_CHARS = 4000

def compute_intangible_llm(doc: dict) -> float:
    pitch_text = doc.get("pitch_deck_text", "") or ""
    pitch_text = pitch_text.strip()

    pitch_sent = doc.get("pitch_sentiment", {})

    logger.debug(f"[Intangible] Starting intangible for doc: {doc.get('name','N/A')}, length={len(pitch_text)}")

    if not pitch_text:
        logger.info("[Intangible] No pitch text => fallback triggered.")
        fallback_score = _investor_fallback(doc, pitch_sent)
        logger.info(f"[Intangible] Fallback => {fallback_score:.2f}")
        return fallback_score

    if len(pitch_text) > MAX_PITCH_CHARS:
        logger.info(f"[Intangible] Truncating from {len(pitch_text)} to {MAX_PITCH_CHARS} chars.")
        pitch_text = pitch_text[:MAX_PITCH_CHARS] + "\n[...Truncated...]"

    try:
        score = _call_deepseek_api_chat(pitch_text)
        logger.info(f"[Intangible] DeepSeek => {score:.2f}")
        return float(score)
    except Exception as e:
        logger.error(f"[Intangible] DeepSeek error => {e}")
        fallback_score = _investor_fallback(doc, pitch_sent)
        logger.info(f"[Intangible] Fallback => {fallback_score:.2f}")
        return fallback_score

def _call_deepseek_api_chat(pitch_text: str) -> float:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("[Intangible] Missing DEEPSEEK_API_KEY environment variable")

    openai.api_key = DEEPSEEK_API_KEY
    openai.api_base = OPENAI_BASE_URL

    system_prompt = (
        "You are an intangible rating assistant. Please output raw JSON with a single field: score in [0..100]."
    )
    user_prompt = (
        f"Startup pitch:\n{pitch_text}\n\n"
        "Reply ONLY with JSON like {\"score\": 55.0}\n"
    )

    response = openai.ChatCompletion.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        stream=False,
        max_tokens=200,
        temperature=0.0
    )

    content = response.choices[0].message.content.strip()
    logger.debug(f"[Intangible] Raw model response => {content}")

    # FIX: Improved regex to handle code blocks with triple backticks
    # This regex will extract JSON content either directly or from code blocks
    # Pattern looks for either direct JSON or content inside ```json ... ``` blocks
    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```|(\{.*?\})'
    match = re.search(json_pattern, content, re.DOTALL)
    
    if match:
        # Use the first matched group that isn't None (either direct JSON or inside code block)
        json_str = next(group for group in match.groups() if group is not None)
        try:
            data = json.loads(json_str.strip())
            raw_score = data.get("score", 50)
            logger.info(f"[Intangible] Successfully parsed JSON: {data}")
        except json.JSONDecodeError as e:
            logger.warning(f"[Intangible] JSON parsing error: {e} => fallback to 50")
            raw_score = 50
    else:
        logger.warning("[Intangible] No JSON found in response => fallback to 50")
        raw_score = 50

    if not (0 <= raw_score <= 100):
        logger.warning(f"[Intangible] out-of-range => {raw_score}, clamping to [0..100]")
        raw_score = max(0, min(100, raw_score))

    return float(raw_score)

def _investor_fallback(doc: dict, pitch_sent: dict) -> float:
    base = 50.0
    founder_exits = doc.get("founder_exits", 0)
    domain_exp = doc.get("founder_domain_exp_yrs", 0)

    if founder_exits >= 1:
        base += 5
    elif domain_exp >= 5:
        base += 3
    if domain_exp < 1 and founder_exits == 0:
        base -= 5

    base = _combine_with_sentiment(base, pitch_sent)
    final = max(0, min(100, base))
    if abs(final - 50) < 0.01:
        final += random.uniform(-5, 5)
    return float(final)

def _combine_with_sentiment(score: float, pitch_sent: dict) -> float:
    if not pitch_sent or "overall_sentiment" not in pitch_sent:
        return score

    overall = pitch_sent["overall_sentiment"]
    sentiment_val = float(overall.get("score", 0.0))
    if sentiment_val > 0.3:
        score += (sentiment_val * 5.0)
    elif sentiment_val < -0.3:
        score += (sentiment_val * 5.0)

    if len(pitch_sent.get("category_sentiments", {})) < 5:
        score -= 3
    return score
