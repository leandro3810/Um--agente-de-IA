from src.agent import Document, EchoReadyModel, RAGAgent, SimpleRetriever
from src.projects import Project, ProjectManager


def build_agent() -> RAGAgent:
    docs = [
        Document("1", "Pipeline CI valida lint, testes e segurança."),
        Document("2", "RAG combina recuperação de contexto com modelo pronto."),
        Document("3", "Segredos devem ficar em variáveis de ambiente."),
    ]
    return RAGAgent(model=EchoReadyModel(), retriever=SimpleRetriever(docs))


def build_project_manager() -> ProjectManager:
    return ProjectManager([
        Project("p1", "Site institucional", "Atualizar landing page e analytics", "em_andamento"),
        Project("p2", "App mobile", "Planejar backlog do próximo release", "planejado"),
    ])


if __name__ == "__main__":
    agent = build_agent()
    manager = build_project_manager()
    question = "O que é RAG?"
    print(agent.ask(question))
    print(manager.ask("Qual projeto está em andamento?"))
