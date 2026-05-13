import unittest

from src.projects import Project, ProjectManager


class ProjectManagerTests(unittest.TestCase):
    def test_build_agent_uses_explicit_model_even_if_falsy(self):
        class FalsyModel:
            def __bool__(self):
                return False

            def generate(self, prompt: str) -> str:
                return "ok"

        manager = ProjectManager([Project("1", "Projeto Alpha", "Criar API", "planejado")])
        model = FalsyModel()

        agent = manager.build_agent(model=model)
        self.assertIs(agent.model, model)

    def test_add_and_list_projects(self):
        manager = ProjectManager()
        manager.add_project(Project("1", "Projeto Alpha", "Criar API", "planejado"))
        manager.add_project(Project("2", "Projeto Beta", "Criar dashboard", "em_andamento"))

        self.assertEqual(len(manager.list_projects()), 2)
        self.assertEqual(len(manager.list_projects(status="em_andamento")), 1)

    def test_update_project_status(self):
        manager = ProjectManager([Project("1", "Projeto Alpha", "Criar API", "planejado")])
        updated = manager.update_status("1", "concluido")
        self.assertEqual(updated.status, "concluido")
        self.assertEqual(manager.get_project("1").status, "concluido")

    def test_remove_project(self):
        manager = ProjectManager([Project("1", "Projeto Alpha", "Criar API", "planejado")])
        manager.add_project(Project("2", "Projeto Beta", "Criar dashboard", "em_andamento"))
        manager.remove_project("2")
        self.assertEqual(len(manager.list_projects()), 1)

    def test_search_and_ask(self):
        manager = ProjectManager([
            Project("1", "Projeto ERP", "Automação financeira", "em_andamento"),
            Project("2", "Portal Cliente", "Novo fluxo de atendimento", "planejado"),
        ])
        results = manager.search("financeira")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "1")

        answer = manager.ask("Qual projeto está em andamento?")
        self.assertIn("Projeto ERP", answer)
        self.assertIn("Status: em_andamento", answer)

    def test_validation_errors(self):
        manager = ProjectManager()
        manager.add_project(Project("1", "Projeto", "Descrição", "planejado"))

        with self.assertRaises(ValueError):
            manager.add_project(Project("1", "Duplicado", "Descrição", "planejado"))

        with self.assertRaises(ValueError):
            manager.update_status("1", "desconhecido")

        with self.assertRaises(ValueError):
            manager.add_project(Project("3", "Inválido", "Descrição", "desconhecido"))

    def test_required_fields_validation(self):
        with self.assertRaises(ValueError):
            Project("", "Projeto", "Descrição", "planejado").validate()
        with self.assertRaises(ValueError):
            Project("1", " ", "Descrição", "planejado").validate()
        with self.assertRaises(ValueError):
            Project("1", "Projeto", " ", "planejado").validate()


if __name__ == "__main__":
    unittest.main()
