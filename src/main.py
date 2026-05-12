from src.agent import Document, EchoReadyModel, RAGAgent, SimpleRetriever


def build_agent() -> RAGAgent:
    docs = [
        Document("1", "Pipeline CI valida lint, testes e segurança."),
        Document("2", "RAG combina recuperação de contexto com modelo pronto."),
        Document("3", "Segredos devem ficar em variáveis de ambiente."),
    ]
    return RAGAgent(model=EchoReadyModel(), retriever=SimpleRetriever(docs))


if __name__ == "__main__":
    agent = build_agent()
    question = "O que é RAG?"
    print(agent.ask(question))
