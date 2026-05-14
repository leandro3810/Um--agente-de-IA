import unittest
from unittest.mock import patch
from urllib.error import URLError

from src.agent import Document, EchoReadyModel, N8NWebhookModel, RAGAgent, SimpleRetriever


class AgentTests(unittest.TestCase):
    def test_n8n_webhook_model_calls_webhook_and_reads_response(self):
        model = N8NWebhookModel("https://example.com/webhook", token="abc123")
        with patch("src.agent.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value.read.return_value = b'{"response":"ok n8n"}'

            answer = model.generate("Teste n8n")

            self.assertEqual(answer, "ok n8n")
            request = mock_urlopen.call_args.args[0]
            self.assertEqual(request.full_url, "https://example.com/webhook")
            self.assertEqual(request.get_method(), "POST")
            self.assertIn(b'"prompt": "Teste n8n"', request.data)
            self.assertEqual(request.headers["Authorization"], "Bearer abc123")

    def test_n8n_webhook_model_wraps_transport_errors(self):
        model = N8NWebhookModel("https://example.com/webhook")
        with patch("src.agent.urlopen", side_effect=URLError("offline")):
            with self.assertRaises(RuntimeError):
                model.generate("Teste")

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
