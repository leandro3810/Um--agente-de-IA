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
    """Integração simples com workflows n8n via webhook HTTP."""

    def __init__(self, webhook_url: str, timeout: float = 10.0, token: str | None = None) -> None:
        normalized_url = webhook_url.strip()
        if not normalized_url.startswith(("http://", "https://")):
            raise ValueError("webhook_url do n8n precisa ser uma URL HTTP/HTTPS válida")
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
                response_body = response.read().decode("utf-8")
        except URLError as exc:
            raise RuntimeError("falha ao chamar webhook do n8n") from exc

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
