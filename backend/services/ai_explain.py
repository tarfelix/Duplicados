"""AI-powered diff explanation — ported from src/services/ai_explain.py."""
import logging
from typing import Optional

from backend.config import get_settings

AI_TEXT_LIMIT = 5000


def is_ai_available() -> bool:
    settings = get_settings()
    return bool(settings.openai_api_key or settings.azure_openai_api_key)


def _get_client():
    from openai import OpenAI

    settings = get_settings()
    if settings.ai_provider.lower() == "azure":
        if not settings.azure_openai_endpoint or not settings.azure_openai_api_key:
            raise ValueError("Azure config incompleto")
        return OpenAI(api_key=settings.azure_openai_api_key, base_url=settings.azure_openai_endpoint.rstrip("/"))
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY não configurada")
    return OpenAI(api_key=settings.openai_api_key)


def _get_model() -> str:
    settings = get_settings()
    if settings.ai_provider.lower() == "azure":
        return settings.azure_openai_deployment
    return settings.openai_model


def explain_differences(text_a: str, text_b: str) -> Optional[str]:
    if not is_ai_available():
        return None
    a = (text_a or "").strip()
    b = (text_b or "").strip()
    if not a or not b:
        return None
    if len(a) > AI_TEXT_LIMIT:
        a = a[:AI_TEXT_LIMIT] + "\n[... texto truncado ...]"
    if len(b) > AI_TEXT_LIMIT:
        b = b[:AI_TEXT_LIMIT] + "\n[... texto truncado ...]"

    system = (
        "Você é um assistente que compara dois textos e explica as diferenças em linguagem simples. "
        "O leitor pode não ser da área jurídica. "
        "Responda em português do Brasil, em 2 a 4 tópicos curtos (bullets ou frases). "
        "Mencione o que difere: datas, números, trechos que só aparecem em um dos textos. "
        "Seja objetivo e claro."
    )
    user = f"""Compare os dois textos abaixo e liste as principais diferenças em linguagem simples.

--- TEXTO 1 ---
{a}

--- TEXTO 2 ---
{b}

Quais são as principais diferenças? (2 a 4 tópicos)"""

    try:
        client = _get_client()
        model = _get_model()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=500,
        )
        content = (resp.choices[0].message.content or "").strip()
        return content if content else None
    except Exception:
        logging.exception("AI explain_differences failed")
        return None
