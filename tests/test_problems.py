import unittest

from src.problems import Problem, ProblemManager


class ProblemManagerTests(unittest.TestCase):
    def test_add_and_list_problems(self):
        manager = ProblemManager()
        manager.add_problem(Problem("pr1", "Erro de login", "Falha ao autenticar", "alta"))
        manager.add_problem(Problem("pr2", "Lentidão no dashboard", "Consulta pesada", "media"))

        self.assertEqual(len(manager.list_problems()), 2)
        self.assertEqual(len(manager.list_problems(severity="alta")), 1)

    def test_update_problem_status(self):
        manager = ProblemManager([Problem("pr1", "Erro de login", "Falha ao autenticar", "alta")])
        updated = manager.update_status("pr1", "investigando")
        self.assertEqual(updated.status, "investigando")
        self.assertEqual(manager.get_problem("pr1").status, "investigando")

    def test_remove_problem(self):
        manager = ProblemManager([
            Problem("pr1", "Erro de login", "Falha ao autenticar", "alta"),
            Problem("pr2", "Lentidão no dashboard", "Consulta pesada", "media"),
        ])
        manager.remove_problem("pr2")
        self.assertEqual(len(manager.list_problems()), 1)

    def test_search_problems(self):
        manager = ProblemManager([
            Problem("pr1", "Erro de login", "Falha ao autenticar no portal", "alta"),
            Problem("pr2", "Timeout API", "Resposta lenta na API de pedidos", "critica"),
        ])

        results = manager.search("autenticar")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "pr1")

    def test_search_with_empty_term_returns_all(self):
        manager = ProblemManager([
            Problem("pr1", "Erro de login", "Falha ao autenticar no portal", "alta"),
            Problem("pr2", "Timeout API", "Resposta lenta na API de pedidos", "critica"),
        ])

        self.assertEqual(len(manager.search("")), 2)
        self.assertEqual(len(manager.search("   ")), 2)

    def test_ask_problem_context(self):
        manager = ProblemManager([
            Problem("pr1", "Erro de login", "Falha ao autenticar no portal", "alta"),
            Problem("pr2", "Timeout API", "Resposta lenta na API de pedidos", "critica"),
        ])
        answer = manager.ask("Qual problema é crítico?")
        self.assertIn("Timeout API", answer)
        self.assertIn("Severidade: critica", answer)

    def test_ask_without_matching_context(self):
        manager = ProblemManager([
            Problem("pr1", "Erro de login", "Falha ao autenticar no portal", "alta"),
        ])

        answer = manager.ask("zzqv")
        self.assertIn("(sem contexto encontrado)", answer)

    def test_validation_errors(self):
        manager = ProblemManager([Problem("pr1", "Erro", "Descrição", "media")])

        with self.assertRaises(ValueError):
            manager.add_problem(Problem("pr1", "Duplicado", "Descrição", "media"))
        with self.assertRaises(ValueError):
            manager.update_status("pr1", "invalido")
        with self.assertRaises(ValueError):
            manager.update_severity("pr1", "urgente")
        with self.assertRaises(ValueError):
            Problem("", "Erro", "Descrição", "media").validate()
        with self.assertRaises(ValueError):
            Problem("pr3", " ", "Descrição", "media").validate()
        with self.assertRaises(ValueError):
            Problem("pr3", "Erro", " ", "media").validate()
        with self.assertRaises(ValueError):
            Problem("pr3", "Erro", "Descrição", "urgente").validate()
        with self.assertRaises(ValueError):
            Problem("pr3", "Erro", "Descrição", "media", "fechado").validate()


if __name__ == "__main__":
    unittest.main()
