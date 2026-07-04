"""
Day 11 tests: Docker configuration file structure.

Validates that Dockerfile, compose.yml, .dockerignore, and
monitoring/prometheus.yml exist and contain the expected content.
These are structural/linting tests — they do not require Docker
to be installed or running.
"""

from pathlib import Path

import yaml

DOCKERFILE = Path("Dockerfile")
COMPOSEFILE = Path("compose.yml")
DOCKERIGNORE = Path(".dockerignore")
PROMETHEUS_CONFIG = Path("monitoring/prometheus.yml")


# ---------------------------------------------------------------------------
# .dockerignore tests
# ---------------------------------------------------------------------------


class TestDockerIgnore:
    def test_file_exists(self):
        assert DOCKERIGNORE.exists()

    def test_excludes_venv(self):
        content = DOCKERIGNORE.read_text()
        assert ".venv" in content

    def test_excludes_data_raw(self):
        content = DOCKERIGNORE.read_text()
        assert "data/raw/" in content

    def test_excludes_model_artifacts(self):
        content = DOCKERIGNORE.read_text()
        assert "artifacts/*.joblib" in content or "artifacts/*.pt" in content

    def test_excludes_pycache(self):
        content = DOCKERIGNORE.read_text()
        assert "__pycache__" in content


# ---------------------------------------------------------------------------
# Dockerfile tests
# ---------------------------------------------------------------------------


class TestDockerfile:
    def _content(self) -> str:
        return DOCKERFILE.read_text()

    def test_file_exists(self):
        assert DOCKERFILE.exists()

    def test_uses_python_313(self):
        assert "python:3.13" in self._content()

    def test_installs_uv(self):
        content = self._content()
        assert "uv" in content

    def test_copies_dependency_files_before_source(self):
        content = self._content()
        lock_pos = content.find("uv.lock")
        src_pos = content.find("COPY src/")
        assert lock_pos < src_pos, (
            "pyproject.toml/uv.lock must be copied before source code "
            "to enable Docker layer caching of the dependency install step"
        )

    def test_runs_uv_sync_frozen(self):
        assert "uv sync --frozen" in self._content()

    def test_copies_api_directory(self):
        assert "COPY api/" in self._content()

    def test_copies_dashboard_directory(self):
        assert "COPY dashboard/" in self._content()

    def test_exposes_port_8000(self):
        assert "8000" in self._content()

    def test_exposes_port_8501(self):
        assert "8501" in self._content()

    def test_installs_curl_for_healthcheck(self):
        assert "curl" in self._content()


# ---------------------------------------------------------------------------
# compose.yml tests
# ---------------------------------------------------------------------------


class TestComposeFile:
    def _parsed(self) -> dict:
        with open(COMPOSEFILE) as f:
            return yaml.safe_load(f)

    def test_file_exists(self):
        assert COMPOSEFILE.exists()

    def test_valid_yaml(self):
        parsed = self._parsed()
        assert isinstance(parsed, dict)

    def test_has_services_key(self):
        assert "services" in self._parsed()

    def test_api_service_defined(self):
        assert "api" in self._parsed()["services"]

    def test_dashboard_service_defined(self):
        assert "dashboard" in self._parsed()["services"]

    def test_api_port_mapping(self):
        api = self._parsed()["services"]["api"]
        ports = api.get("ports", [])
        assert any("8000" in str(p) for p in ports)

    def test_dashboard_port_mapping(self):
        dash = self._parsed()["services"]["dashboard"]
        ports = dash.get("ports", [])
        assert any("8501" in str(p) for p in ports)

    def test_api_has_healthcheck(self):
        api = self._parsed()["services"]["api"]
        assert "healthcheck" in api

    def test_healthcheck_polls_health_endpoint(self):
        api = self._parsed()["services"]["api"]
        hc = api["healthcheck"]
        test_cmd = str(hc.get("test", ""))
        assert "/health" in test_cmd

    def test_api_has_artifacts_volume(self):
        api = self._parsed()["services"]["api"]
        volumes = api.get("volumes", [])
        assert any("artifacts" in str(v) for v in volumes)

    def test_dashboard_depends_on_api(self):
        dash = self._parsed()["services"]["dashboard"]
        assert "api" in str(dash.get("depends_on", ""))

    def test_dashboard_has_api_url_env_var(self):
        dash = self._parsed()["services"]["dashboard"]
        env = dash.get("environment", [])
        assert any("CLOUDDRIFT_API_URL" in str(e) for e in env)

    def test_api_url_points_to_api_service(self):
        dash = self._parsed()["services"]["dashboard"]
        env = dash.get("environment", [])
        env_str = str(env)
        assert "http://api:" in env_str

    def test_both_services_on_same_network(self):
        services = self._parsed()["services"]
        api_nets = set(str(services["api"].get("networks", "")).split())
        dash_nets = set(str(services["dashboard"].get("networks", "")).split())
        assert api_nets & dash_nets, "api and dashboard must share at least one network"

    def test_restart_policy_configured(self):
        services = self._parsed()["services"]
        for svc_name in ["api", "dashboard"]:
            svc = services[svc_name]
            assert "restart" in svc, f"{svc_name} missing restart policy"

    def test_prometheus_is_optional_profile(self):
        services = self._parsed().get("services", {})
        if "prometheus" in services:
            prom = services["prometheus"]
            profiles = prom.get("profiles", [])
            assert "monitoring" in profiles, (
                "prometheus service must be in the 'monitoring' profile "
                "so it doesn't start by default"
            )


# ---------------------------------------------------------------------------
# Prometheus configuration tests
# ---------------------------------------------------------------------------


class TestPrometheusConfig:
    def test_file_exists(self):
        assert PROMETHEUS_CONFIG.exists()

    def test_valid_yaml(self):
        with open(PROMETHEUS_CONFIG) as f:
            parsed = yaml.safe_load(f)
        assert isinstance(parsed, dict)

    def test_scrapes_api_service(self):
        with open(PROMETHEUS_CONFIG) as f:
            content = f.read()
        assert "api" in content

    def test_scrapes_metrics_path(self):
        with open(PROMETHEUS_CONFIG) as f:
            content = f.read()
        assert "/metrics" in content

    def test_has_global_scrape_interval(self):
        with open(PROMETHEUS_CONFIG) as f:
            parsed = yaml.safe_load(f)
        assert "global" in parsed
        assert "scrape_interval" in parsed["global"]
