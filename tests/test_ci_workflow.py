"""Tests for the GitHub Pages deploy job in .github/workflows/ci.yaml.

UI-UI-041/042/043/055/056. The workflow is parsed as data; the site-assembly
logic it calls is tested in test_assemble_site.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yaml"


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


@pytest.fixture(scope="module")
def deploy(workflow) -> dict:
    jobs = workflow.get("jobs", {})
    assert "deploy" in jobs, "ci.yaml must define a deploy job"
    return jobs["deploy"]


def _triggers(workflow: dict) -> dict:
    # PyYAML (YAML 1.1) reads the bare key `on` as boolean True.
    return workflow.get("on") or workflow.get(True) or {}


def _steps(job: dict) -> list[dict]:
    return job.get("steps", [])


def _step_using(job: dict, action: str) -> dict:
    matches = [s for s in _steps(job) if str(s.get("uses", "")).startswith(action + "@")]
    assert len(matches) == 1, f"expected exactly one {action} step"
    return matches[0]


def _assemble_step(job: dict) -> dict:
    matches = [s for s in _steps(job) if "bin/assemble_site.py" in str(s.get("run", ""))]
    assert len(matches) == 1, "deploy job must assemble the site with bin/assemble_site.py"
    return matches[0]


# @spec UI-UI-056
def test_deploy_needs_test_job(deploy):
    needs = deploy.get("needs")
    assert needs == "test" or needs == ["test"]


# @spec UI-UI-056
def test_workflow_can_be_dispatched_manually(workflow):
    assert "workflow_dispatch" in _triggers(workflow)


# @spec UI-UI-056
def test_deploy_runs_only_for_push_or_dispatch_on_main(deploy):
    cond = " ".join(str(deploy.get("if", "")).split())
    assert "github.ref == 'refs/heads/main'" in cond
    assert "github.event_name == 'push'" in cond
    assert "github.event_name == 'workflow_dispatch'" in cond
    assert "pull_request" not in cond.replace("!= 'pull_request'", "")


# @spec UI-UI-041, UI-UI-042
def test_deploy_assembles_site_with_repo_variable_endpoint(deploy):
    run = " ".join(str(_assemble_step(deploy)["run"]).split())
    assert "--endpoint" in run
    assert "${{ vars.PHENO_API_ENDPOINT }}" in run
    assert "--out" in run


# @spec UI-UI-055
def test_assembly_failure_fails_the_job(deploy):
    step = _assemble_step(deploy)
    assert not step.get("continue-on-error"), "a missing result must fail the job"
    assert "|| true" not in str(step["run"])
    assert not deploy.get("continue-on-error")


# @spec UI-UI-043
def test_deploy_uploads_assembled_site_and_deploys_pages(deploy):
    out = str(_assemble_step(deploy)["run"]).split("--out", 1)[1].split()[0]
    upload = _step_using(deploy, "actions/upload-pages-artifact")
    assert upload.get("with", {}).get("path") == out, "upload the directory that was assembled"
    _step_using(deploy, "actions/deploy-pages")
    steps = _steps(deploy)
    assert steps.index(_assemble_step(deploy)) < steps.index(upload) < steps.index(
        _step_using(deploy, "actions/deploy-pages")
    )


# @spec UI-UI-043
def test_deploy_has_pages_permissions_and_environment(deploy, workflow):
    perms = deploy.get("permissions") or workflow.get("permissions") or {}
    assert perms.get("pages") == "write"
    assert perms.get("id-token") == "write"
    env = deploy.get("environment")
    name = env.get("name") if isinstance(env, dict) else env
    assert name == "github-pages"
