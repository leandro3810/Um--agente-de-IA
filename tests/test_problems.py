import unittest

from src.problems import Problem, ProblemManager


class ProblemManagerTests(unittest.TestCase):
    def test_add_list_update_and_remove_problem(self):
        manager = ProblemManager()
        manager.add_problem(Problem("pr1", "Erro de login", "Falha ao autenticar", "alta"))
        manager.add_problem(Problem("pr2", "Lentidão no dashboard", "Consulta pesada", "media"))

        self.assertEqual(len(manager.list_problems()), 2)
        self.assertEqual(len(manager.list_problems(severity="alta")), 1)

        updated = manager.update_status("pr1", "investigando")
        self.assertEqual(updated.status, "investigando")
        self.assertEqual(manager.get_problem("pr1").status, "investigando")

        manager.remove_problem("pr2")
        self.assertEqual(len(manager.list_problems()), 1)

    def test_search_and_ask(self):
        manager = ProblemManager([
            Problem("pr1", "Erro de login", "Falha ao autenticar no portal", "alta"),
            Problem("pr2", "Timeout API", "Resposta lenta na API de pedidos", "critica"),
        ])

        results = manager.search("autenticar")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "pr1")

        answer = manager.ask("Qual problema é crítico?")
        self.assertIn("Timeout API", answer)
        self.assertIn("Severidade: critica", answer)

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
