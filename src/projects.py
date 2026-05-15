from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from typing import Iterable

from src.agent import Document, EchoReadyModel, RAGAgent, ReadyModel, SimpleRetriever


VALID_STATUSES = {"planejado", "em_andamento", "pausado", "concluido", "cancelado"}


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    done: bool = False

    def validate(self) -> None:
        if not self.id.strip():
            raise ValueError("id da tarefa não pode ser vazio")
        if not self.title.strip():
            raise ValueError("título da tarefa não pode ser vazio")


@dataclass(frozen=True)
class Project:
    id: str
    name: str
    description: str
    status: str = "planejado"
    tasks: tuple[Task, ...] = ()

    def validate(self) -> None:
        if not self.id.strip():
            raise ValueError("id do projeto não pode ser vazio")
        if not self.name.strip():
            raise ValueError("nome do projeto não pode ser vazio")
        if not self.description.strip():
            raise ValueError("descrição do projeto não pode ser vazia")
        if self.status not in VALID_STATUSES:
            raise ValueError(f"status inválido: {self.status}")
        for task in self.tasks:
            task.validate()


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

    def add_task(self, project_id: str, task: Task) -> Project:
        task.validate()
        project = self.get_project(project_id)
        updated = replace(project, tasks=project.tasks + (task,))
        self._projects[project_id] = updated
        return updated

    def list_tasks(self, project_id: str) -> list[Task]:
        return list(self.get_project(project_id).tasks)

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
        def format_tasks(project: Project) -> str:
            if not project.tasks:
                return "Sem tarefas."
            items = [
                f"{task.id}: {task.title} ({'concluída' if task.done else 'pendente'})"
                for task in project.tasks
            ]
            return "Tarefas: " + "; ".join(items)

        return [
            Document(
                project.id,
                (
                    f"Projeto: {project.name}. Status: {project.status}. "
                    f"Descrição: {project.description}. {format_tasks(project)}"
                ),
            )
            for project in self._projects.values()
        ]

    def export_csv(self, filepath: str) -> None:
        fieldnames = ["id", "name", "description", "status"]
        with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for project in self._projects.values():
                writer.writerow({
                    "id": project.id,
                    "name": project.name,
                    "description": project.description,
                    "status": project.status,
                })

    def build_agent(self, model: ReadyModel | None = None) -> RAGAgent:
        ready_model = model if model is not None else EchoReadyModel()
        retriever = SimpleRetriever(self._to_documents())
        return RAGAgent(model=ready_model, retriever=retriever)

    def ask(self, question: str, model: ReadyModel | None = None) -> str:
        return self.build_agent(model=model).ask(question)
