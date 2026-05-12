import unittest

from src.agent import Document, EchoReadyModel, RAGAgent, SimpleRetriever


class AgentTests(unittest.TestCase):
    def test_retriever_returns_relevant_document(self):
        retriever = SimpleRetriever([
            Document("1", "RAG usa recuperação de contexto."),
            Document("2", "Segredos ficam em variáveis de ambiente."),
        ])
        docs = retriever.retrieve("Como funciona RAG?", k=1)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].id, "1")

    def test_rag_agent_builds_answer(self):
        retriever = SimpleRetriever([Document("1", "RAG reduz alucinação.")])
        agent = RAGAgent(model=EchoReadyModel(), retriever=retriever)
        answer = agent.ask("Como reduzir alucinação?")
        self.assertIn("Contexto", answer)
        self.assertIn("RAG reduz alucinação", answer)


if __name__ == "__main__":
    unittest.main()
