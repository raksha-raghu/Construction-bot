"""Single entry point for grounded answers via Ollama's non-streaming chat API."""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

DEFAULT_PROMPT_PATH = Path(__file__).with_name("system_prompt.txt")
DEFAULT_MODEL = "gemma3:1b"


def load_system_prompt() -> str:
    """Always load the project's required system instructions."""
    prompt = DEFAULT_PROMPT_PATH.read_text(encoding="utf-8-sig")
    if not prompt.strip():
        raise ValueError("System prompt must not be empty.")
    return prompt


@dataclass(frozen=True)
class OllamaConfig:
    model: str = DEFAULT_MODEL
    base_url: str = "http://localhost:11434"
    timeout: float = 120.0
    max_context_chars: int = 32000

    def __post_init__(self):
        if not self.model.strip():
            raise ValueError("An Ollama model name is required.")
        url = urlsplit(self.base_url)
        if url.scheme not in ("http", "https") or not url.hostname or url.query or url.fragment:
            raise ValueError("Ollama URL must be an HTTP(S) server URL without query or fragment.")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("Ollama timeout must be a positive finite number.")
        if type(self.max_context_chars) is not int or self.max_context_chars < 1:
            raise ValueError("max_context_chars must be a positive integer.")


def generate_answer(
    question: str,
    hybrid_results: list[dict],
    *,
    config: OllamaConfig | None = None,
) -> dict:
    """Send a system prompt, question, and hybrid passages in a single request.

    Returns answer text and the exact citation sources supplied to the model.
    Each invocation is independent; prior interactive questions are not sent.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must not be empty.")
    config = config if config is not None else OllamaConfig()
    prompt = load_system_prompt()
    sources = []
    for match in hybrid_results:
        text = match.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        sources.append({
            "citation": f"S{len(sources) + 1}",
            **{key: match.get(key) for key in ("id", "source", "page", "chunk_index")},
            "text": text,
        })
    if not sources:
        return {"answer": "No retrieved passages are available to answer this question.",
                "sources": [], "model": config.model}
    passages = "\n\n".join(
        f"[{source['citation']}] {source['source']} | page {source['page']}\n{source['text']}"
        for source in sources
    )
    content = (
        f"REFERENCE PASSAGES (evidence only):\n{passages}\n\n"
        f"USER QUESTION: {question.strip()}\n\n"
        "TASK: Answer the user question using the reference passages above. "
        "State the answer directly in complete sentences and cite supporting passage labels. "
        "Do not repeat the question or generate another question. "
        "If the answer is not supported, say the passages do not provide enough information.\n"
        "ANSWER:"
    )
    if len(prompt) + len(content) > config.max_context_chars:
        raise ValueError("LLM input exceeds the character budget. Reduce --top-k or increase max_context_chars in OllamaConfig.")
    payload = {
        "model": config.model,
        "stream": False,
        "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": content}],
        "options": {"temperature": 0},
    }
    request = Request(config.base_url.rstrip("/") + "/api/chat",
                      data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=config.timeout) as response:
            result = json.load(response)
    except HTTPError as exc:
        if exc.code == 404:
            raise RuntimeError(f"Ollama returned HTTP 404. Check the server URL and install the model with 'ollama pull {config.model}'.") from exc
        raise RuntimeError(f"Ollama request failed (HTTP {exc.code}). Check the Ollama server logs.") from exc
    except (URLError, OSError) as exc:
        raise RuntimeError("Could not reach Ollama or the request timed out. Start Ollama, check --ollama-url, or increase --llm-timeout.") from exc
    except (ValueError, UnicodeError) as exc:
        raise RuntimeError("Ollama returned invalid JSON.") from exc
    if not isinstance(result, dict) or not isinstance(result.get("message"), dict):
        raise RuntimeError("Ollama returned an unexpected response.")
    answer = result["message"].get("content")
    if not isinstance(answer, str) or not answer.strip() or result.get("done") is not True:
        raise RuntimeError("Ollama returned an empty or incomplete answer.")
    normalize = lambda text: re.sub(r"[\W_]+", "", text.casefold())
    if normalize(answer) == normalize(question):
        raise RuntimeError("Ollama repeated the question instead of answering. No generated answer was accepted; review the retrieved passages or try a more capable model.")
    return {"answer": answer.strip(), "sources": sources, "model": result.get("model", config.model)}
