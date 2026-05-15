from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Iterable, Protocol
from urllib.error import URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class Document:
    id: str
    content: str


class ReadyModel(Protocol):
    def generate(self, prompt: str) -> str:
        ...


class EchoReadyModel:
    """Implementação de exemplo para um modelo pronto.

    Troque por integrações reais (OpenAI, Azure OpenAI, Ollama, etc).
    """

    def generate(self, prompt: str) -> str:
        return f"[Resposta do modelo]\n{prompt}"


class N8NWebhookModel:
    """Adapter ReadyModel que envia prompts para um workflow n8n via webhook HTTP."""

    def __init__(self, webhook_url: str, timeout: float = 10.0, token: str | None = None) -> None:
        normalized_url = webhook_url.strip()
        if not normalized_url.startswith(("http://", "https://")):
            raise ValueError(
                f"webhook_url do n8n precisa ser uma URL HTTP/HTTPS válida, recebido: {normalized_url}"
            )
        self._webhook_url = normalized_url
        self._timeout = timeout
        self._token = token

    def generate(self, prompt: str) -> str:
        payload = json.dumps({"prompt": prompt}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        request = Request(self._webhook_url, data=payload, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self._timeout) as response:
                charset = response.headers.get_content_charset("utf-8")
                if not isinstance(charset, str):
                    charset = "utf-8"
                response_body = response.read().decode(charset)
        except URLError as exc:
            raise RuntimeError(f"falha ao chamar webhook do n8n em {self._webhook_url}: {exc}") from exc

        if not response_body.strip():
            return ""

        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError:
            return response_body

        if isinstance(parsed, dict):
            for key in ("response", "answer", "text", "output"):
                value = parsed.get(key)
                if isinstance(value, str):
                    return value
            return json.dumps(parsed, ensure_ascii=False)

        return response_body


class OpenAIModel:
    """Adapter ReadyModel para OpenAI Chat Completions API."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", timeout: float = 30.0) -> None:
        normalized_key = api_key.strip()
        if not normalized_key:
            raise ValueError("api_key da OpenAI não pode ser vazia")
        self._api_key = normalized_key
        self._model = model
        self._timeout = timeout
        self._url = "https://api.openai.com/v1/chat/completions"

    def generate(self, prompt: str) -> str:
        payload = json.dumps({
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        request = Request(self._url, data=payload, headers=headers, method="POST")

        try:
            with urlopen(request, timeout=self._timeout) as response:
                charset = response.headers.get_content_charset("utf-8")
                response_body = response.read().decode(charset)
        except URLError as exc:
            raise RuntimeError(f"falha ao chamar OpenAI em {self._url}: {exc}") from exc

        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError:
            return response_body

        if isinstance(parsed, dict):
            choices = parsed.get("choices")
            if isinstance(choices, list) and choices:
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    message = first_choice.get("message")
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, str):
                            return content
            return json.dumps(parsed, ensure_ascii=False)

        return response_body


class SimpleRetriever:
    def __init__(self, documents: Iterable[Document]) -> None:
        self._documents = list(documents)

    @staticmethod
    def _normalize(text: str) -> set[str]:
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return {token for token in cleaned.split() if token}

    def retrieve(self, query: str, k: int = 3) -> list[Document]:
        query_tokens = self._normalize(query)
        if not query_tokens:
            return []

        scored: list[tuple[int, Document]] = []
        for doc in self._documents:
            score = len(query_tokens & self._normalize(doc.content))
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:k]]


class RAGAgent:
    def __init__(self, model: ReadyModel, retriever: SimpleRetriever) -> None:
        self.model = model
        self.retriever = retriever

    def ask(self, question: str) -> str:
        context_docs = self.retriever.retrieve(question)
        context = "\n\n".join(f"[{doc.id}] {doc.content}" for doc in context_docs)

        prompt = (
            "Você é um agente de IA para suporte de projetos. "
            "Responda com base no contexto recuperado.\n\n"
            f"Pergunta: {question}\n"
            f"Contexto:\n{context if context else '(sem contexto encontrado)'}\n"
        )
        return self.model.generate(prompt)
