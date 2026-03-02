"""
Explicação de diferenças entre dois textos em linguagem simples, via OpenAI ou Azure.
Usado no diálogo "Ver diferenças" do Verificador de Duplicidade.
Ativo apenas se OPENAI_API_KEY (ou variáveis Azure) estiverem configuradas.
"""
import os
from typing import Optional

# Limite de caracteres por texto para evitar custo/latência
AI_TEXT_LIMIT = 5000


def _get_api_key() -> Optional[str]:
    """Retorna a chave da API (OpenAI ou Azure) se configurada."""
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY")
    if key and key.strip():
        return key.strip()
    return None


def _get_client():
    """Cria cliente OpenAI (OpenAI ou Azure)."""
    from openai import OpenAI
    if (os.environ.get("AI_PROVIDER") or "").lower() == "azure":
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        key = os.environ.get("AZURE_OPENAI_API_KEY")
        if not endpoint or not key:
            raise ValueError("Azure config incompleto: AZURE_OPENAI_ENDPOINT e AZURE_OPENAI_API_KEY")
        return OpenAI(api_key=key, base_url=endpoint.rstrip("/"))
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY não configurada")
    return OpenAI(api_key=key)


def _get_model() -> str:
    if (os.environ.get("AI_PROVIDER") or "").lower() == "azure":
        return os.environ.get("AZURE_OPENAI_DEPLOYMENT") or "gpt-4o"
    return os.environ.get("OPENAI_MODEL") or "gpt-4o"


def explain_differences(text_a: str, text_b: str) -> Optional[str]:
    """
    Compara dois textos e retorna uma explicação em português simples (2-4 tópicos).
    Retorna None se a API não estiver configurada ou em caso de erro.
    """
    if not _get_api_key():
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
        "Seja objetivo e claro. Não use jargão desnecessário."
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
        return None


def is_ai_available() -> bool:
    """Retorna True se a API de IA estiver configurada e utilizável."""
    return _get_api_key() is not None
