# Contribuindo para o projeto

Obrigado por querer contribuir com o **Um Agente de IA**.

## Como contribuir

1. Faça um fork do repositório.
2. Crie uma branch para sua mudança.
3. Faça alterações pequenas e focadas.
4. Execute os testes localmente.
5. Abra um Pull Request com descrição clara.

## Ambiente local

Pré-requisitos:
- Python 3.11+

Instalação:

```bash
pip install -r requirements.txt
```

## Executando testes

Antes de abrir PR, rode:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

## Padrões de contribuição

- Mantenha o código simples e legível.
- Evite mudanças não relacionadas ao objetivo do PR.
- Atualize documentação quando necessário.
- Não inclua segredos, tokens ou credenciais no código.

## Pull Requests

Ao abrir seu PR:
- Explique o problema e a solução adotada.
- Liste os testes executados.
- Referencie issues relacionadas, quando houver.

## Segurança e conduta

- Para reportar vulnerabilidades, siga `SECURITY.md`.
- Respeite o `CODE_OF_CONDUCT.md`.
