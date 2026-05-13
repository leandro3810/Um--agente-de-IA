from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from src.agent import Document, EchoReadyModel, RAGAgent, ReadyModel, SimpleRetriever


VALID_STATUSES = {"planejado", "em_andamento", "pausado", "concluido", "cancelado"}


@dataclass(frozen=True)
class Project:
    id: str
    name: str
    description: str
    status: str = "planejado"

    def validate(self) -> None:
        if not self.id.strip():
            raise ValueError("id do projeto não pode ser vazio")
        if not self.name.strip():
            raise ValueError("nome do projeto não pode ser vazio")
        if not self.description.strip():
            raise ValueError("descrição do projeto não pode ser vazia")
        if self.status not in VALID_STATUSES:
            raise ValueError(f"status inválido: {self.status}")


class ProjectManager:
    def __init__(self, projects: Iterable[Project] | None = None) -> None:
        self._projects: dict[str, Project] = {}
        for project in projects or []:
            self.add_project(project)

    def add_project(self, project: Project) -> None:
        project.validate()
        if project.id in self._projects:
            raise ValueError(f"projeto já existe: {project.id}")
        self._projects[project.id] = project

    def get_project(self, project_id: str) -> Project:
        if project_id not in self._projects:
            raise KeyError(f"projeto não encontrado: {project_id}")
        return self._projects[project_id]

    def list_projects(self, status: str | None = None) -> list[Project]:
        if status is None:
            return list(self._projects.values())
        if status not in VALID_STATUSES:
            raise ValueError(f"status inválido: {status}")
        return [project for project in self._projects.values() if project.status == status]

    def update_status(self, project_id: str, status: str) -> Project:
        if status not in VALID_STATUSES:
            raise ValueError(f"status inválido: {status}")
        project = self.get_project(project_id)
        updated = replace(project, status=status)
        self._projects[project_id] = updated
        return updated

    def remove_project(self, project_id: str) -> None:
        if project_id not in self._projects:
            raise KeyError(f"projeto não encontrado: {project_id}")
        del self._projects[project_id]

    def search(self, term: str) -> list[Project]:
        normalized = term.strip().lower()
        if not normalized:
            return self.list_projects()
        return [
            project
            for project in self._projects.values()
            if normalized in project.name.lower() or normalized in project.description.lower()
        ]

    def _to_documents(self) -> list[Document]:
        return [
            Document(
                project.id,
                f"Projeto: {project.name}. Status: {project.status}. Descrição: {project.description}",
            )
            for project in self._projects.values()
        ]

    def build_agent(self, model: ReadyModel | None = None) -> RAGAgent:
        ready_model = model if model is not None else EchoReadyModel()
        retriever = SimpleRetriever(self._to_documents())
        return RAGAgent(model=ready_model, retriever=retriever)

    def ask(self, question: str, model: ReadyModel | None = None) -> str:
        return self.build_agent(model=model).ask(question)
