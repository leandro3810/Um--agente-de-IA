import csv
import io
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.projects import Project, ProjectManager, Task


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

    def test_add_and_list_tasks(self):
        manager = ProjectManager([Project("1", "Projeto ERP", "Automação financeira", "em_andamento")])
        manager.add_task("1", Task("t1", "Mapear integrações"))
        manager.add_task("1", Task("t2", "Publicar endpoint", done=True))

        tasks = manager.list_tasks("1")
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0].title, "Mapear integrações")
        self.assertTrue(tasks[1].done)

        answer = manager.ask("Qual tarefa é Publicar endpoint?")
        self.assertIn("Publicar endpoint", answer)

    def test_add_task_inexistent_project_raises(self):
        manager = ProjectManager()
        with self.assertRaises(KeyError):
            manager.add_task("nao-existe", Task("t1", "Tarefa"))

    def test_export_csv_with_projects(self):
        manager = ProjectManager([
            Project("1", "Projeto Alpha", "Criar API", "planejado"),
            Project("2", "Projeto Beta", "Criar dashboard", "em_andamento"),
        ])
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            temp_path = tmp.name
        try:
            manager.export_csv(temp_path)
            with open(temp_path, newline="", encoding="utf-8") as csvfile:
                rows = list(csv.DictReader(csvfile))

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["id"], "1")
            self.assertEqual(rows[0]["name"], "Projeto Alpha")
            self.assertEqual(rows[1]["status"], "em_andamento")
        finally:
            os.remove(temp_path)

    def test_export_csv_empty_manager_writes_only_header(self):
        manager = ProjectManager()
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            temp_path = tmp.name
        try:
            manager.export_csv(temp_path)
            with open(temp_path, newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
                self.assertEqual(rows, [])
                self.assertEqual(reader.fieldnames, ["id", "name", "description", "status"])
        finally:
            os.remove(temp_path)

    def test_export_csv_s3_uploads_correct_content(self):
        manager = ProjectManager([
            Project("1", "Projeto Alpha", "Criar API", "planejado"),
            Project("2", "Projeto Beta", "Criar dashboard", "em_andamento"),
        ])
        mock_boto3 = MagicMock()
        mock_s3 = mock_boto3.client.return_value

        with patch("src.projects.boto3", mock_boto3):
            manager.export_csv_s3("meu-bucket", "projetos/export.csv", region_name="us-east-1")

        mock_boto3.client.assert_called_once_with("s3", region_name="us-east-1")
        call_kwargs = mock_s3.put_object.call_args.kwargs
        self.assertEqual(call_kwargs["Bucket"], "meu-bucket")
        self.assertEqual(call_kwargs["Key"], "projetos/export.csv")
        self.assertEqual(call_kwargs["ContentType"], "text/csv")

        body_text = call_kwargs["Body"].decode("utf-8")
        reader = csv.DictReader(io.StringIO(body_text))
        rows = list(reader)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["id"], "1")
        self.assertEqual(rows[1]["status"], "em_andamento")

    def test_import_csv_s3_loads_projects(self):
        csv_content = "id,name,description,status\n3,Projeto Gama,Novo sistema,planejado\n"
        mock_boto3 = MagicMock()
        mock_s3 = mock_boto3.client.return_value
        mock_s3.get_object.return_value = {"Body": io.BytesIO(csv_content.encode("utf-8"))}

        manager = ProjectManager()
        with patch("src.projects.boto3", mock_boto3):
            manager.import_csv_s3("meu-bucket", "projetos/import.csv", region_name="sa-east-1")

        mock_boto3.client.assert_called_once_with("s3", region_name="sa-east-1")
        mock_s3.get_object.assert_called_once_with(Bucket="meu-bucket", Key="projetos/import.csv")
        self.assertEqual(len(manager.list_projects()), 1)
        self.assertEqual(manager.get_project("3").name, "Projeto Gama")

    def test_import_csv_s3_skips_existing_projects(self):
        csv_content = "id,name,description,status\n1,Projeto Alpha,Criar API,planejado\n4,Novo,Desc,concluido\n"
        mock_boto3 = MagicMock()
        mock_s3 = mock_boto3.client.return_value
        mock_s3.get_object.return_value = {"Body": io.BytesIO(csv_content.encode("utf-8"))}

        manager = ProjectManager([Project("1", "Projeto Alpha", "Criar API", "planejado")])
        with patch("src.projects.boto3", mock_boto3):
            manager.import_csv_s3("meu-bucket", "projetos/import.csv")

        self.assertEqual(len(manager.list_projects()), 2)
        self.assertEqual(manager.get_project("4").name, "Novo")

    def test_export_csv_s3_raises_when_boto3_unavailable(self):
        manager = ProjectManager()
        with patch("src.projects.boto3", None):
            with self.assertRaises(ImportError) as ctx:
                manager.export_csv_s3("bucket", "key")
        self.assertIn("boto3", str(ctx.exception))

    def test_import_csv_s3_raises_when_boto3_unavailable(self):
        manager = ProjectManager()
        with patch("src.projects.boto3", None):
            with self.assertRaises(ImportError) as ctx:
                manager.import_csv_s3("bucket", "key")
        self.assertIn("boto3", str(ctx.exception))

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
