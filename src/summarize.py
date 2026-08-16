"""
summarize.py
────────────
LLM-based call summarizer used in the production loop.

Bridges the gap between predictions: after each call, the transcript and
the action taken are summarized into a short narrative string that becomes
``prev_outcome`` for the next prediction.

This closes the loop between training (where ``prev_outcome`` was generated
by GPT-4o for the synthetic dataset) and production (where it would
otherwise be missing).

Usage:
    from src.summarize import summarize_call
    prev_outcome = summarize_call(transcript, action_taken)
"""
from __future__ import annotations

import os
from openai import OpenAI

_client: OpenAI | None = None

DEFAULT_MODEL = "gpt-4o-mini"   # ~$0.0001 / call

_SYSTEM_PROMPT = (
    "Eres un asistente que resume llamadas comerciales en español. "
    "Genera un único resumen narrativo en una sola frase, máximo 25 palabras, "
    "tono profesional, sin prefijos ni etiquetas."
)

_USER_PROMPT_TEMPLATE = (
    "Transcripción de la llamada:\n"
    "{transcript}\n\n"
    "Acción decidida tras la llamada: {action}\n\n"
    "Resumen:"
)


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY no está configurada. Expórtala o usa el modo "
                "fallback (template-based) en el módulo que llame a summarize_call."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def summarize_call(
    transcript: str,
    action_taken: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 80,
) -> str:
    """Genera un resumen narrativo de una llamada para usar como prev_outcome.

    Parameters
    ----------
    transcript    : transcripción de la llamada (estilo "Agente: … Contacto: …")
    action_taken  : acción comercial decidida tras la llamada
    model         : modelo OpenAI (default gpt-4o-mini, óptimo coste/calidad)
    temperature   : 0.3 da resúmenes consistentes pero no monótonos
    max_tokens    : 80 tokens ≈ 25 palabras (límite del prompt)

    Returns
    -------
    str con el resumen, ya stripped y sin saltos de línea.
    """
    client = _get_client()
    user_prompt = _USER_PROMPT_TEMPLATE.format(
        transcript=transcript.strip(),
        action=action_taken.strip(),
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip().replace("\n", " ")


# ── Quick standalone test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_transcript = (
        "Agente: Buenos días Alejandro, soy Laura de MoveUp. "
        "Le llamo para hablar de movilidad corporativa. "
        "Contacto: Hola, sí, cuénteme. Estamos buscando alternativas a las dietas."
    )
    sample_action = "Enviar documentación"

    print("Transcripción:")
    print(f"  {sample_transcript}\n")
    print(f"Acción: {sample_action}\n")
    print("Resumen generado:")
    print(f"  {summarize_call(sample_transcript, sample_action)}")
