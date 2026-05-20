---
name: Custom issue template
about: Describe this issue template's purpose here.
title: ''
labels: enhancement
assignees: ''

---

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from src.agent import Document, EchoReadyModel, RAGAgent, ReadyModel, SimpleRetriever


VALID_PROBLEM_STATUSES = {"aberto", "investigando", "resolvido", "arquivado"}
VALID_PROBLEM_SEVERITIES = {"baixa", "media", "alta", "critica"}


@dataclass(frozen=True)
class Problem:
    id: str
    title: str
    description: str
    severity: str = "media"
    status: str = "aberto"

    def validate(self) -> None:
        if not self.id.strip():
            raise ValueError("id do problema não pode ser vazio")
        if not self.title.strip():
            raise ValueError("título do problema não pode ser vazio")
        if not self.description.strip():
            raise ValueError("descrição do problema não pode ser vazia")
        if self.severity not in VALID_PROBLEM_SEVERITIES:
            raise ValueError(f"severidade inválida: {self.severity}")
        if self.status not in VALID_PROBLEM_STATUSES:
            raise ValueError(f"status inválido: {self.status}")


class ProblemManager:
    def __init__(self, problems: Iterable[Problem] | None = None) -> None:
        self._problems: dict[str, Problem] = {}
        for problem in problems or []:
            self.add_problem(problem)

    def add_problem(self, problem: Problem) -> None:
        problem.validate()
        if problem.id in self._problems:
            raise ValueError(f"problema já existe: {problem.id}")
        self._problems[problem.id] = problem

    def get_problem(self, problem_id: str) -> Problem:
        if problem_id not in self._problems:
            raise KeyError(f"problema não encontrado: {problem_id}")
        return self._problems[problem_id]

    def list_problems(
        self, status: str | None = None, severity: str | None = None
    ) -> list[Problem]:
        if status is not None and status not in VALID_PROBLEM_STATUSES:
            raise ValueError(f"status inválido: {status}")
        if severity is not None and severity not in VALID_PROBLEM_SEVERITIES:
            raise ValueError(f"severidade inválida: {severity}")

        return [
            problem
            for problem in self._problems.values()
            if (status is None or problem.status == status)
            and (severity is None or problem.severity == severity)
        ]

    def update_status(self, problem_id: str, status: str) -> Problem:
        if status not in VALID_PROBLEM_STATUSES:
            raise ValueError(f"status inválido: {status}")
        problem = self.get_problem(problem_id)
        updated = replace(problem, status=status)
        self._problems[problem_id] = updated
        return updated

    def update_severity(self, problem_id: str, severity: str) -> Problem:
        if severity not in VALID_PROBLEM_SEVERITIES:
            raise ValueError(f"severidade inválida: {severity}")
        problem = self.get_problem(problem_id)
        updated = replace(problem, severity=severity)
        self._problems[problem_id] = updated
        return updated

    def remove_problem(self, problem_id: str) -> None:
        if problem_id not in self._problems:
            raise KeyError(f"problema não encontrado: {problem_id}")
        del self._problems[problem_id]

    def search(self, term: str) -> list[Problem]:
        normalized = term.strip().lower()
        if not normalized:
            return self.list_problems()
        return [
            problem
            for problem in self._problems.values()
            if normalized in problem.title.lower()
            or normalized in problem.description.lower()
        ]

    def _to_documents(self) -> list[Document]:
        return [
            Document(
                problem.id,
                (
                    f"Problema: {problem.title}. Severidade: {problem.severity}. "
                    f"Status: {problem.status}. Descrição: {problem.description}"
                ),
            )
            for problem in self._problems.values()
        ]

    def build_agent(self, model: ReadyModel | None = None) -> RAGAgent:
        ready_model = model if model is not None else EchoReadyModel()
        retriever = SimpleRetriever(self._to_documents())
        return RAGAgent(model=ready_model, retriever=retriever)

    def ask(self, question: str, model: ReadyModel | None = None) -> str:
        return self.build_agent(model=model).ask(question)
