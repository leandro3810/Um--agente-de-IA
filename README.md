# Um Agente de IA (RAG + Modelos Prontos)

Projeto base para criar uma IA do zero com estrutura completa: sistemas, design, infraestrutura, segurança, código de conduta, licença, linguagem, bibliotecas, treinamento e avaliação.

## Objetivo
Construir um agente de IA modular para apoiar projetos em tempo real, usando **RAG** (Retrieval-Augmented Generation) e integração com **modelos prontos** (OpenAI, Azure OpenAI, Ollama, etc).

## Stack (linguagem e bibliotecas)
- **Linguagem:** Python 3.11+
- **Bibliotecas padrão usadas no código atual:** `dataclasses`, `typing`, `json`, `pathlib`, `re`, `urllib`
- **Modelos prontos:** via adapter (`ReadyModel`), permitindo trocar o provedor sem alterar o núcleo do agente

## Estrutura
```text
src/
  agent.py        # Núcleo RAG (ingestão, recuperação, prompt e resposta)
  main.py         # Exemplo de execução via CLI
  training.py     # Pipeline simples de preparo de dados
  evaluation.py   # Métricas básicas de avaliação
```

## Design e arquitetura
1. **Camada de conhecimento**: documentos em memória (`Document`) + índice simples.
2. **Camada de recuperação**: `SimpleRetriever` por sobreposição de termos.
3. **Camada de geração**: `ReadyModel` (interface) + implementação de exemplo (`EchoReadyModel`).
4. **Orquestração RAG**: `RAGAgent.ask()` monta contexto + prompt + chamada ao modelo.

## Integração com n8n
- O projeto agora inclui `N8NWebhookModel` em `src/agent.py`.
- Essa integração envia `{"prompt": "...texto..."}` para um webhook n8n via `POST` e usa o retorno para responder.
- Campos de resposta priorizados no JSON retornado: `response`, `answer`, `text`, `output`.
- Exemplo de uso:
```python
from src.agent import Document, N8NWebhookModel, RAGAgent, SimpleRetriever

docs = [Document("1", "Contexto do projeto")]
model = N8NWebhookModel(
    "https://n8n.example.com/webhook/rota",
    token="SEU_TOKEN_OPCIONAL",  # opcional: use apenas se o webhook exigir autenticação
)
agent = RAGAgent(model=model, retriever=SimpleRetriever(docs))
print(agent.ask("Me atualize sobre o projeto"))
```

## Infraestrutura sugerida
- Executar localmente com Python.
- Evolução natural:
  - API REST (FastAPI)
  - Banco vetorial (FAISS/Qdrant/pgvector)
  - Observabilidade (OpenTelemetry + logs estruturados)
  - CI/CD (lint/test/security scan)

## Segurança
- Ver `SECURITY.md`.
- Práticas aplicadas neste baseline:
  - separação entre recuperação e geração
  - sanitização básica de texto antes de recuperação
  - prompt com contexto controlado
  - sem hardcode de segredos

## Treinamento
- `src/training.py` descreve base para curadoria, limpeza e split de dados.
- Estratégia recomendada:
  - usar fine-tuning somente quando necessário
  - priorizar RAG para reduzir custo e risco de alucinação

## Avaliação
- `src/evaluation.py` inclui métricas iniciais:
  - `exact_match`
  - `context_recall`
- Evoluir com avaliações humanas e testes de segurança (prompt injection e data leakage).

## Sistema de controle de projetos
- `src/projects.py` adiciona um gerenciador de projetos com:
  - cadastro e remoção de projetos
  - atualização de status (`planejado`, `em_andamento`, `pausado`, `concluido`, `cancelado`)
  - busca por nome/descrição
  - perguntas ao agente com contexto dos projetos cadastrados

## Como executar
```bash
python3 -m src.main
```

## Página inicial
- Foi adicionada uma página estática em `index.html`.
- Para abrir localmente no navegador, execute um servidor local:
```bash
python3 -m http.server 8000
```
Depois acesse: `http://localhost:8000`

## Governança
- **Código de Conduta:** `CODE_OF_CONDUCT.md`
- **Licença:** `LICENSE`
