"""
Day 12 tests: GitHub Actions workflow structural verification.

Validates that .github/workflows/ci.yml exists and contains the
expected jobs, triggers, and steps. Does not require GitHub Actions
to be running — purely structural YAML parsing.
"""

from pathlib import Path

import yaml

WORKFLOW_PATH = Path(".github/workflows/ci.yml")


class TestCIWorkflow:
    def _load(self) -> dict:
        with open(WORKFLOW_PATH) as f:
            return yaml.safe_load(f)

    def test_workflow_file_exists(self):
        assert WORKFLOW_PATH.exists()

    def test_valid_yaml(self):
        parsed = self._load()
        assert isinstance(parsed, dict)

    def test_triggers_on_push_to_dev(self):
        parsed = self._load()
        # "on" is parsed as boolean True by PyYAML (reserved YAML keyword)
        push = parsed[True]["push"]["branches"]
        assert "dev" in push

    def test_triggers_on_push_to_main(self):
        parsed = self._load()
        # "on" is parsed as boolean True by PyYAML (reserved YAML keyword)
        push = parsed[True]["push"]["branches"]
        assert "main" in push

    def test_triggers_on_pr_to_main(self):
        parsed = self._load()
        pr = parsed[True]["pull_request"]["branches"]
        assert "main" in pr

    def test_has_lint_job(self):
        assert "lint" in self._load()["jobs"]

    def test_has_test_job(self):
        assert "test" in self._load()["jobs"]

    def test_has_docker_build_job(self):
        assert "docker-build" in self._load()["jobs"]

    def test_test_job_needs_lint(self):
        jobs = self._load()["jobs"]
        needs = jobs["test"].get("needs", [])
        assert "lint" in str(needs)

    def test_docker_build_needs_test(self):
        jobs = self._load()["jobs"]
        needs = jobs["docker-build"].get("needs", [])
        assert "test" in str(needs)

    def test_lint_runs_ruff_check(self):
        jobs = self._load()["jobs"]
        steps = jobs["lint"]["steps"]
        step_names = [s.get("name", "") for s in steps]
        assert any("ruff" in n.lower() for n in step_names)

    def test_test_job_runs_pytest(self):
        jobs = self._load()["jobs"]
        steps = jobs["test"]["steps"]
        has_pytest = any("pytest" in str(s.get("run", "")) for s in steps)
        assert has_pytest

    def test_docker_build_runs_compose_build(self):
        jobs = self._load()["jobs"]
        steps = jobs["docker-build"]["steps"]
        has_compose_build = any(
            "docker compose build" in str(s.get("run", "")) for s in steps
        )
        assert has_compose_build

    def test_all_jobs_run_on_ubuntu(self):
        jobs = self._load()["jobs"]
        for job_name, job in jobs.items():
            assert "ubuntu" in job.get("runs-on", ""), (
                f"Job '{job_name}' must run on ubuntu-latest"
            )

    def test_uses_uv_setup_action(self):
        jobs = self._load()["jobs"]
        # At least the lint job should use the uv setup action
        lint_steps = jobs["lint"]["steps"]
        has_uv_action = any(
            "astral-sh/setup-uv" in str(s.get("uses", "")) for s in lint_steps
        )
        assert has_uv_action

    def test_uses_python_313(self):
        jobs = self._load()["jobs"]
        lint_steps = jobs["lint"]["steps"]
        # Python version may appear in 'run' commands (uv python install)
        # or in 'with' blocks (actions/setup-python). Check both.
        step_cmds = " ".join(str(s.get("run", "")) for s in lint_steps)
        step_withs = " ".join(str(s.get("with", "")) for s in lint_steps)
        assert "3.13" in step_cmds or "3.13" in step_withs
