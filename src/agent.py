from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Protocol


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
