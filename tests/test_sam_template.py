"""Tests for the SAM template + samconfig — DEPLOY-BE-010/013/014/015/019/020.

These tests parse ``template.yaml`` and ``samconfig.toml`` as data and assert
structural properties. They do NOT call AWS. ``test_template_validates_via_sam``
shells out to ``sam validate`` if the SAM CLI is installed (skipped otherwise).
"""
from __future__ import annotations

import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = REPO_ROOT / "template.yaml"
SAMCONFIG_PATH = REPO_ROOT / "samconfig.toml"


def _sam_yaml_loader():
    """SAM templates use !Ref / !GetAtt / !Sub short-form tags that PyYAML refuses by default."""
    class _SamLoader(yaml.SafeLoader):
        pass

    def _passthrough(loader, tag_suffix, node):
        if isinstance(node, yaml.ScalarNode):
            return loader.construct_scalar(node)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node)
        return loader.construct_mapping(node)

    _SamLoader.add_multi_constructor("!", _passthrough)
    return _SamLoader


@pytest.fixture(scope="module")
def template() -> dict:
    if not TEMPLATE_PATH.exists():
        pytest.fail(f"SAM template not found at {TEMPLATE_PATH}")
    return yaml.load(TEMPLATE_PATH.read_text(), Loader=_sam_yaml_loader())


@pytest.fixture(scope="module")
def resources(template) -> dict:
    return template.get("Resources", {})


def _resources_of_type(resources: dict, type_name: str) -> dict:
    return {k: v for k, v in resources.items() if v.get("Type") == type_name}


# @spec DEPLOY-BE-014
def test_template_parses_as_sam_transform(template):
    assert template.get("Transform") in (
        "AWS::Serverless-2016-10-31",
        ["AWS::Serverless-2016-10-31"],
    ), "template.yaml must declare the AWS::Serverless transform"


# @spec DEPLOY-BE-014, DEPLOY-BE-008
def test_template_declares_inference_lambda_function(resources):
    fns = _resources_of_type(resources, "AWS::Serverless::Function")
    assert fns, "template.yaml must declare at least one AWS::Serverless::Function"
    fn = next(iter(fns.values()))
    props = fn.get("Properties", {})
    assert props.get("PackageType") == "Image", "Inference Lambda must be a container image"
    assert props.get("MemorySize") == 3008
    assert props.get("Timeout") == 29, (
        "Lambda timeout must be 29 s, under API Gateway HTTP API's 30 s integration cutoff"
    )


# @spec DEPLOY-BE-010, DEPLOY-BE-014
def test_template_uses_http_api_not_rest_api(resources):
    rest = _resources_of_type(resources, "AWS::Serverless::Api")
    http = _resources_of_type(resources, "AWS::Serverless::HttpApi")
    fns = _resources_of_type(resources, "AWS::Serverless::Function")

    has_explicit_http = bool(http)
    has_inline_http_events = False
    for fn in fns.values():
        events = fn.get("Properties", {}).get("Events", {}) or {}
        for ev in events.values():
            if ev.get("Type") == "HttpApi":
                has_inline_http_events = True

    assert has_explicit_http or has_inline_http_events, (
        "template.yaml must front the inference Lambda with an HTTP API"
        " (AWS::Serverless::HttpApi or inline HttpApi events)"
    )
    assert not rest, (
        "template.yaml must NOT use AWS::Serverless::Api (REST API);"
        " DEPLOY-BE-010 mandates HTTP API"
    )


# @spec DEPLOY-BE-010
def test_template_routes_post_predict(resources):
    fns = _resources_of_type(resources, "AWS::Serverless::Function")
    routes: set[tuple[str, str]] = set()
    for fn in fns.values():
        events = fn.get("Properties", {}).get("Events", {}) or {}
        for ev in events.values():
            if ev.get("Type") != "HttpApi":
                continue
            props = ev.get("Properties", {})
            method = (props.get("Method") or "").upper()
            path = props.get("Path") or ""
            routes.add((method, path))
    assert ("POST", "/predict") in routes, f"Expected POST /predict route; got {routes}"
    assert ("GET", "/labels") not in routes, f"GET /labels route should be removed; got {routes}"


# @spec DEPLOY-BE-015
def test_template_exposes_model_run_id_and_bucket_as_parameters(template):
    params = template.get("Parameters") or {}
    assert "ModelRunId" in params or "MODEL_RUN_ID" in params or "MODELRUNID" in params, (
        "template.yaml must expose MODEL_RUN_ID as a CloudFormation parameter"
    )
    assert "PhenoS3Bucket" in params or "PHENO_S3_BUCKET" in params, (
        "template.yaml must expose PHENO_S3_BUCKET as a CloudFormation parameter"
    )


# @spec DEPLOY-BE-015
def test_lambda_function_receives_model_run_id_and_bucket_env_vars(resources):
    fns = _resources_of_type(resources, "AWS::Serverless::Function")
    fn = next(iter(fns.values()))
    env = fn.get("Properties", {}).get("Environment", {}).get("Variables", {}) or {}
    assert "MODEL_RUN_ID" in env, "Lambda must receive MODEL_RUN_ID env var"
    assert "PHENO_S3_BUCKET" in env, "Lambda must receive PHENO_S3_BUCKET env var"


# @spec DEPLOY-BE-020
def test_template_declares_log_group_with_14_day_retention(resources):
    log_groups = _resources_of_type(resources, "AWS::Logs::LogGroup")
    assert log_groups, "template.yaml must declare an AWS::Logs::LogGroup for the inference Lambda"
    assert any(
        lg.get("Properties", {}).get("RetentionInDays") == 14
        for lg in log_groups.values()
    ), "Log group must set RetentionInDays: 14"


# @spec DEPLOY-BE-019
def test_samconfig_pins_us_east_1():
    assert SAMCONFIG_PATH.exists(), f"samconfig.toml must exist at {SAMCONFIG_PATH}"
    cfg = tomllib.loads(SAMCONFIG_PATH.read_text())
    # samconfig.toml layout: [<env>.<command>.parameters] region = "us-east-1"
    # Top-level keys may be scalars (e.g. `version = 0.1`) — skip those.
    found = False
    for env_cfg in cfg.values():
        if not isinstance(env_cfg, dict):
            continue
        for cmd_cfg in env_cfg.values():
            if not isinstance(cmd_cfg, dict):
                continue
            params = cmd_cfg.get("parameters", {})
            if isinstance(params, dict) and params.get("region") == "us-east-1":
                found = True
    assert found, "samconfig.toml must pin region=us-east-1 (no other region acceptable)"


# @spec DEPLOY-BE-014
@pytest.mark.skipif(not shutil.which("sam"), reason="aws-sam-cli not installed")
def test_template_validates_via_sam():
    result = subprocess.run(
        ["sam", "validate", "--lint", "--template-file", str(TEMPLATE_PATH)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"sam validate failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


# ── HTTP API: /health route, CORS, throttling ─────────────────────────────────


def _http_routes(resources: dict) -> dict[tuple[str, str], dict]:
    """Map (METHOD, path) → HttpApi event properties across all functions."""
    routes: dict[tuple[str, str], dict] = {}
    for fn_name, fn in _resources_of_type(resources, "AWS::Serverless::Function").items():
        for ev in (fn.get("Properties", {}).get("Events", {}) or {}).values():
            if ev.get("Type") != "HttpApi":
                continue
            props = ev.get("Properties", {})
            key = ((props.get("Method") or "").upper(), props.get("Path") or "")
            routes[key] = {**props, "_function": fn_name}
    return routes


@pytest.fixture(scope="module")
def http_api(resources) -> tuple[str, dict]:
    apis = _resources_of_type(resources, "AWS::Serverless::HttpApi")
    assert len(apis) == 1, (
        "template.yaml must declare exactly one explicit AWS::Serverless::HttpApi"
        " (needed to carry CorsConfiguration and DefaultRouteSettings)"
    )
    name, api = next(iter(apis.items()))
    return name, api.get("Properties", {}) or {}


# @spec DEPLOY-BE-028
def test_template_routes_get_health_to_same_function_as_predict(resources):
    routes = _http_routes(resources)
    assert ("GET", "/health") in routes, f"Expected GET /health route; got {sorted(routes)}"
    assert routes[("GET", "/health")]["_function"] == routes[("POST", "/predict")]["_function"], (
        "GET /health must integrate with the same Lambda function as POST /predict"
    )


def test_all_routes_attach_to_the_explicit_http_api(resources, http_api):
    """Routes must hang off the declared HttpApi, or its CORS/throttling would not apply."""
    api_name, _ = http_api
    for key, props in _http_routes(resources).items():
        assert props.get("ApiId") == api_name, (
            f"Route {key} must set ApiId: !Ref {api_name}; got {props.get('ApiId')!r}"
        )


def _cors(http_api) -> dict:
    _, props = http_api
    cors = props.get("CorsConfiguration")
    assert isinstance(cors, dict), "HttpApi must declare a CorsConfiguration mapping"
    return cors


# @spec DEPLOY-BE-030
def test_cors_allows_exactly_pages_and_localhost_8000_origins(http_api):
    assert sorted(_cors(http_api).get("AllowOrigins") or []) == [
        "http://localhost:8000",
        "https://mattjtravers.github.io",
    ]


# @spec DEPLOY-BE-031
def test_cors_allows_exactly_post_and_get(http_api):
    methods = [m.upper() for m in (_cors(http_api).get("AllowMethods") or [])]
    assert sorted(methods) == ["GET", "POST"]


# @spec DEPLOY-BE-032
def test_cors_allows_exactly_content_type_header(http_api):
    headers = [h.lower() for h in (_cors(http_api).get("AllowHeaders") or [])]
    assert headers == ["content-type"]


# @spec DEPLOY-BE-033
def test_cors_does_not_allow_credentials(http_api):
    assert _cors(http_api).get("AllowCredentials") is False


# @spec DEPLOY-BE-034
def test_cors_preflight_max_age_is_600(http_api):
    assert _cors(http_api).get("MaxAge") == 600


# @spec DEPLOY-BE-036
def test_http_api_default_route_throttling_is_2_rps_burst_10(http_api):
    _, props = http_api
    settings = props.get("DefaultRouteSettings") or {}
    assert settings.get("ThrottlingRateLimit") == 2
    assert settings.get("ThrottlingBurstLimit") == 10
    assert not props.get("RouteSettings"), (
        "Per-route RouteSettings would override the default throttle for some routes"
    )


def test_api_endpoint_output_references_the_explicit_http_api(template, http_api):
    api_name, _ = http_api
    value = (template.get("Outputs", {}).get("ApiEndpoint", {}) or {}).get("Value", "")
    assert f"${{{api_name}}}" in value, (
        f"ApiEndpoint output must reference {api_name}, not the implicit ServerlessHttpApi"
    )
