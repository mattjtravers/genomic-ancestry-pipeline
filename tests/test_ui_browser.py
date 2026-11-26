"""Browser tests for the static site in site/ — UI-UI-* specs.

Each test assembles the site with bin/assemble_site.py into a temp directory,
serves it on a local port, and points the endpoint at ``{origin}/api`` on the
same origin, so API calls are intercepted with ``page.route`` and CORS never
comes into play (CORS is verified against the deployed API instead).

Timing-dependent behaviour uses a test-only ``fetch`` wrapper (``HOLD_SCRIPT``)
that can park ``/api/predict`` calls until the test resolves them, and
Playwright's clock for the 35-second client timeout.
"""
from __future__ import annotations

import functools
import json
import subprocess
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from playwright.sync_api import Page, Route, expect

REPO = Path(__file__).resolve().parent.parent
ASSEMBLE = REPO / "bin" / "assemble_site.py"
SAMPLES = REPO / "samples"
LIVE_RESULT = json.loads((SAMPLES / "sample_snp_2.result.json").read_text())
PRECOMPUTED_1 = json.loads((SAMPLES / "sample_snp_1.result.json").read_text())

# Display form per UI-UI-011/060: full 1000 Genomes name, then the code in parentheses.
YRI_DISPLAY = "Yoruba in Ibadan, Nigeria (YRI)"
CHB_DISPLAY = "Han Chinese in Beijing, China (CHB)"
assert PRECOMPUTED_1["predicted_phenotype"] == "YRI"
assert LIVE_RESULT["predicted_phenotype"] == "CHB"

GENERIC = "The prediction request failed — please try again."
UNAVAILABLE = "The prediction service is starting up or unavailable — please try again in a moment."
NOT_CONFIGURED = "The prediction service is not configured."
BUSY = "The demo is busy right now — please try again in a few seconds."

HOLD_SCRIPT = """
(() => {
  const realFetch = window.fetch.bind(window);
  window.__holdPredict = false;
  window.__holdMatch = null;
  window.__held = [];
  window.fetch = (input, init = {}) => {
    const url = typeof input === "string" ? input : input.url;
    const held = (window.__holdPredict && url.includes("/api/predict"))
      || (window.__holdMatch !== null && url.includes(window.__holdMatch));
    if (held) {
      return new Promise((resolve, reject) => {
        const entry = { url, init, resolve, reject };
        if (init.signal) {
          init.signal.addEventListener("abort", () =>
            reject(new DOMException("The operation was aborted.", "AbortError")));
        }
        window.__held.push(entry);
      });
    }
    return realFetch(input, init);
  };
})();
"""


def _vcf(n_samples: int = 1, rows: int = 3) -> str:
    names = "\t".join(f"s{i}" for i in range(1, n_samples + 1))
    header = (
        "##fileformat=VCFv4.1\n"
        f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{names}\n"
    )
    gts = "\t".join(["0|1"] * n_samples)
    body = "".join(f"15\t{28513871 + i}\t.\tA\tG\t.\tPASS\t.\tGT\t{gts}\n" for i in range(rows))
    return header + body


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):  # keep pytest output clean
        pass


class Site:
    """A served, assembled copy of site/ plus recorded API traffic."""

    def __init__(self, page: Page, tmp_path: Path):
        self.page = page
        self.dir = tmp_path / "served"
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), functools.partial(_QuietHandler, directory=str(self.dir))
        )
        self.origin = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.endpoint = f"{self.origin}/api"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.requests: list[dict] = []
        self.logs: list[dict] = []
        self.predict_response: dict = {"status": 200, "json": LIVE_RESULT}
        self.health_response: dict = {"status": 200, "json": {"status": "ok"}}
        page.on("console", self._on_console)
        page.route(f"{self.endpoint}/**", self._handle_api)

    def assemble(self, *, endpoint: str | None = "same-origin", samples: Path = SAMPLES,
                 config: bool = True) -> None:
        args = [sys.executable, str(ASSEMBLE), "--out", str(self.dir), "--samples", str(samples),
                "--endpoint", self.endpoint if endpoint == "same-origin" else (endpoint or "")]
        proc = subprocess.run(args, capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr
        if not config:
            (self.dir / "config.js").unlink()

    def open(self, **assemble_kwargs) -> Page:
        self.assemble(**assemble_kwargs)
        self.page.goto(f"{self.origin}/index.html")
        return self.page

    def _on_console(self, msg) -> None:
        try:
            record = json.loads(msg.text)
        except ValueError:
            return
        if isinstance(record, dict) and "event" in record:
            self.logs.append({**record, "_console": msg.type})

    def _handle_api(self, route: Route) -> None:
        req = route.request
        path = req.url[len(self.endpoint):]
        self.requests.append({"method": req.method, "path": path, "headers": req.headers,
                              "body": req.post_data})
        spec = self.health_response if path.startswith("/health") else self.predict_response
        if spec.get("abort"):
            route.abort()
            return
        body = json.dumps(spec["json"]) if "json" in spec else spec.get("text", "")
        route.fulfill(status=spec["status"], body=body,
                      headers={"Content-Type": "application/json"})

    def api_calls(self, path: str) -> list[dict]:
        return [r for r in self.requests if r["path"].startswith(path)]

    def events(self, name: str) -> list[dict]:
        return [r for r in self.logs if r["event"] == name]

    def close(self) -> None:
        self.server.shutdown()


@pytest.fixture
def site(page: Page, tmp_path: Path):
    s = Site(page, tmp_path)
    yield s
    s.close()


def _upload(page: Page, name: str = "mine.vcf", content: str | bytes | None = None) -> None:
    data = content if isinstance(content, bytes) else (content or _vcf()).encode()
    page.set_input_files("#vcf-file", files=[{"name": name, "mimeType": "text/plain",
                                              "buffer": data}])


def _select_sample(page: Page, label: str) -> None:
    page.get_by_label(label, exact=True).check()


def _hold(page: Page) -> None:
    page.evaluate("window.__holdPredict = true")


def _release(page: Page, index: int, body: dict, status: int = 200) -> None:
    page.evaluate(
        """([i, body, status]) => window.__held[i].resolve(new Response(JSON.stringify(body),
             {status, headers: {"Content-Type": "application/json"}}))""",
        [index, body, status],
    )


# ── Warm-up ────────────────────────────────────────────────────────────────────


# @spec UI-UI-027
def test_page_load_sends_one_health_request_without_blocking_interaction(site):
    site.page.add_init_script(HOLD_SCRIPT)
    page = site.open()
    expect(page.locator("#vcf-file")).to_be_enabled()
    page.wait_for_timeout(300)
    assert len(site.api_calls("/health")) == 1
    assert site.api_calls("/health")[0]["method"] == "GET"


# @spec UI-UI-028, UI-UI-049
def test_failed_health_request_is_not_retried_and_changes_nothing(site):
    site.health_response = {"abort": True}
    page = site.open()
    page.wait_for_timeout(500)
    assert len(site.api_calls("/health")) == 1
    expect(page.locator("#result-message")).to_be_empty()
    expect(page.locator("#result")).to_be_hidden()
    assert not [e for e in site.logs if "health" in json.dumps(e).lower()]


# @spec UI-UI-029, UI-UI-048
@pytest.mark.parametrize("variant", ["empty-endpoint", "no-config-js"])
def test_unconfigured_endpoint_skips_warm_up_and_logs_error(site, variant):
    page = site.open(endpoint=None) if variant == "empty-endpoint" else site.open(config=False)
    page.wait_for_timeout(300)
    assert site.api_calls("/health") == []
    errors = site.events("api_endpoint_not_configured")
    assert errors and errors[0]["level"] == "ERROR" and errors[0]["_console"] == "error"


# ── Input and validation ───────────────────────────────────────────────────────


# @spec UI-UI-001
def test_file_input_accepts_vcf(site):
    page = site.open()
    expect(page.locator("#vcf-file")).to_have_attribute("type", "file")
    expect(page.locator("#vcf-file")).to_have_attribute("accept", ".vcf")


# @spec UI-UI-051
def test_run_disabled_until_a_file_is_chosen_even_with_sample_selected(site):
    page = site.open()
    expect(page.locator("#run-button")).to_be_disabled()
    _select_sample(page, "Sample SNP 1")
    expect(page.locator("#run-button")).to_be_disabled()
    _upload(page)
    expect(page.locator("#run-button")).to_be_enabled()


# @spec UI-UI-003, UI-UI-044
def test_non_vcf_extension_shows_inline_error_and_sends_nothing(site):
    page = site.open()
    _upload(page, name="mine.txt")
    page.click("#run-button")
    expect(page.locator("#upload-error")).not_to_be_empty()
    assert site.api_calls("/predict") == []
    warn = site.events("upload_invalid")
    assert warn and warn[0]["level"] == "WARN" and warn[0]["filename"] == "mine.txt"
    assert warn[0]["errors"] and warn[0]["_console"] == "warning"


# @spec UI-UI-003
def test_extension_check_is_case_insensitive(site):
    page = site.open()
    _upload(page, name="MINE.VCF")
    page.click("#run-button")
    expect(page.locator("#result")).to_be_visible()
    assert len(site.api_calls("/predict")) == 1


# @spec UI-UI-004
def test_file_over_4_mib_is_rejected(site):
    page = site.open()
    _upload(page, content=b"#" * (4 * 1024 * 1024 + 1))
    page.click("#run-button")
    expect(page.locator("#upload-error")).not_to_be_empty()
    assert site.api_calls("/predict") == []


# @spec UI-UI-004
def test_file_of_exactly_4_mib_is_accepted(site):
    vcf = _vcf().encode()
    padded = b"##" + b"x" * (4 * 1024 * 1024 - len(vcf) - 3) + b"\n" + vcf
    assert len(padded) == 4 * 1024 * 1024
    page = site.open()
    _upload(page, content=padded)
    page.click("#run-button")
    expect(page.locator("#result")).to_be_visible()
    assert len(site.api_calls("/predict")) == 1


# @spec UI-UI-005
@pytest.mark.parametrize(
    "content",
    [_vcf(n_samples=2), _vcf(n_samples=0).replace("\tFORMAT\t\n", "\n"),
     "##fileformat=VCFv4.1\n15\t1\t.\tA\tG\t.\tPASS\t.\tGT\t0|1\n"],
    ids=["two-samples", "sites-only", "no-chrom-line"],
)
def test_not_exactly_one_sample_is_rejected(site, content):
    page = site.open()
    _upload(page, content=content)
    page.click("#run-button")
    expect(page.locator("#upload-error")).not_to_be_empty()
    assert site.api_calls("/predict") == []


# ── Live prediction request ────────────────────────────────────────────────────


# @spec UI-UI-008, UI-UI-024
def test_run_posts_vcf_json_and_logs_request(site):
    page = site.open()
    content = _vcf()
    _upload(page, content=content)
    page.click("#run-button")
    expect(page.locator("#result")).to_be_visible()

    (call,) = site.api_calls("/predict")
    assert call["method"] == "POST"
    assert call["headers"]["content-type"] == "application/json"
    assert json.loads(call["body"]) == {"vcf": content}
    sent = site.events("predict_request_sent")
    assert sent and sent[0]["level"] == "INFO"
    assert (sent[0]["input"], sent[0]["filename"], sent[0]["size_bytes"]) == (
        "upload", "mine.vcf", len(content.encode()))


# @spec UI-UI-038
def test_run_submits_upload_even_when_a_sample_is_selected(site):
    page = site.open()
    _select_sample(page, "Sample SNP 1")
    expect(page.locator("#result-source")).to_have_text("Precomputed result for Sample SNP 1")
    content = _vcf(rows=5)
    _upload(page, content=content)
    page.click("#run-button")
    expect(page.locator("#result-source")).to_have_text("Live prediction")
    assert json.loads(site.api_calls("/predict")[0]["body"]) == {"vcf": content}


# @spec UI-UI-030
def test_run_without_endpoint_shows_not_configured_and_sends_nothing(site):
    page = site.open(config=False)
    _upload(page)
    page.click("#run-button")
    expect(page.locator("#result-message")).to_have_text(NOT_CONFIGURED)
    assert site.api_calls("/predict") == []


# @spec UI-UI-009, UI-UI-046
def test_loading_indicator_and_disabled_button_while_request_in_flight(site):
    site.page.add_init_script(HOLD_SCRIPT)
    page = site.open()
    _hold(page)
    _upload(page)
    page.click("#run-button")
    expect(page.locator("#loading")).to_be_visible()
    expect(page.locator("#run-button")).to_be_disabled()

    _release(page, 0, LIVE_RESULT)
    expect(page.locator("#loading")).to_be_hidden()
    expect(page.locator("#run-button")).to_be_enabled()
    shown = site.events("live_result_rendered")
    assert shown and shown[0]["level"] == "INFO"
    assert shown[0]["phenotype"] == LIVE_RESULT["predicted_phenotype"]
    assert shown[0]["markers"] == len(LIVE_RESULT["top_markers"])


# @spec UI-UI-015, UI-UI-025
@pytest.mark.parametrize(
    ("status", "code", "message"),
    [
        (503, "MODEL_UNAVAILABLE", UNAVAILABLE),
        (400, "INVALID_VCF", "The uploaded VCF could not be processed (multi-sample file, "
                             "malformed format, or no usable variants)."),
        (400, "INVALID_INPUT",
         "The request was rejected as malformed — please reload and try again."),
        (500, "INFERENCE_FAILED", "The prediction service encountered an error — please try again "
                                  "or contact support."),
        (500, "INTERNAL_ERROR", "The prediction service encountered an error — please try again "
                                "or contact support."),
        (500, "SOMETHING_NEW", GENERIC),
    ],
)
def test_contract_error_codes_map_to_messages_and_are_logged(site, status, code, message):
    site.predict_response = {"status": status, "json": {"error": code, "detail": "d"}}
    page = site.open()
    _upload(page)
    page.click("#run-button")
    expect(page.locator("#result-message")).to_have_text(message)
    expect(page.locator("#run-button")).to_be_enabled()
    err = site.events("predict_http_error")
    assert err and err[0]["level"] == "ERROR" and err[0]["_console"] == "error"
    assert (err[0]["status"], err[0]["error_code"], err[0]["detail"]) == (status, code, "d")


# @spec UI-UI-025
def test_http_error_without_contract_body_logs_nulls(site):
    site.predict_response = {"status": 500, "text": "<html>oops</html>"}
    page = site.open()
    _upload(page)
    page.click("#run-button")
    expect(page.locator("#result-message")).to_have_text(GENERIC)
    err = site.events("predict_http_error")[0]
    assert (err["status"], err["error_code"], err["detail"]) == (500, None, None)


# @spec UI-UI-035
@pytest.mark.parametrize("status", [503, 504])
def test_gateway_503_504_without_contract_body_shows_unavailable(site, status):
    site.predict_response = {"status": status, "json": {"message": "Service Unavailable"}}
    page = site.open()
    _upload(page)
    page.click("#run-button")
    expect(page.locator("#result-message")).to_have_text(UNAVAILABLE)


# @spec UI-UI-036
def test_429_shows_busy_message(site):
    site.predict_response = {"status": 429, "json": {"message": "Too Many Requests"}}
    page = site.open()
    _upload(page)
    page.click("#run-button")
    expect(page.locator("#result-message")).to_have_text(BUSY)


# @spec UI-UI-010, UI-UI-026
def test_network_error_shows_generic_message_and_logs_stack(site):
    site.predict_response = {"abort": True}
    page = site.open()
    _upload(page)
    page.click("#run-button")
    expect(page.locator("#result-message")).to_have_text(GENERIC)
    expect(page.locator("#run-button")).to_be_enabled()
    err = site.events("predict_failed")
    assert err and err[0]["level"] == "ERROR" and err[0]["error"] and "stack" in err[0]


# @spec UI-UI-010
def test_client_aborts_after_35_seconds(site):
    site.page.clock.install()
    site.page.add_init_script(HOLD_SCRIPT)
    page = site.open()
    _hold(page)
    _upload(page)
    page.click("#run-button")
    expect(page.locator("#loading")).to_be_visible()
    page.clock.run_for(34_000)
    expect(page.locator("#loading")).to_be_visible()
    page.clock.run_for(1_500)
    expect(page.locator("#result-message")).to_have_text(GENERIC)
    expect(page.locator("#run-button")).to_be_enabled()
    assert page.evaluate("window.__held[0].init.signal.aborted")


# @spec UI-UI-059
def test_error_hides_previously_rendered_result(site):
    page = site.open()
    _select_sample(page, "Sample SNP 1")
    expect(page.locator("#result")).to_be_visible()
    site.predict_response = {"status": 500, "json": {"error": "INFERENCE_FAILED", "detail": "x"}}
    _upload(page)
    page.click("#run-button")
    expect(page.locator("#result-message")).not_to_be_empty()
    expect(page.locator("#result")).to_be_hidden()


# ── Latest action wins ─────────────────────────────────────────────────────────


# @spec UI-UI-034, UI-UI-052, UI-UI-053
def test_sample_selection_supersedes_in_flight_live_prediction(site):
    site.page.add_init_script(HOLD_SCRIPT)
    page = site.open()
    _hold(page)
    _upload(page)
    page.click("#run-button")
    expect(page.locator("#loading")).to_be_visible()

    _select_sample(page, "Sample SNP 1")
    expect(page.locator("#loading")).to_be_hidden()            # UI-UI-034
    expect(page.locator("#run-button")).to_be_enabled()
    assert page.evaluate("window.__held[0].init.signal.aborted")  # UI-UI-052
    expect(page.locator("#result-source")).to_have_text("Precomputed result for Sample SNP 1")

    _release(page, 0, LIVE_RESULT)                              # UI-UI-053
    page.wait_for_timeout(200)
    expect(page.locator("#result-source")).to_have_text("Precomputed result for Sample SNP 1")


# @spec UI-UI-034, UI-UI-052, UI-UI-053
def test_run_supersedes_pending_sample_fetch(site):
    """Run is disabled during a live request (UI-UI-009), so the Run-side supersede
    case is a Run while a sample's precomputed result is still loading."""
    site.page.add_init_script(HOLD_SCRIPT)
    page = site.open()
    page.evaluate("window.__holdMatch = 'sample_snp_1.result.json'")
    _select_sample(page, "Sample SNP 1")
    page.wait_for_function("window.__held.length === 1")
    _upload(page)
    expect(page.locator("#run-button")).to_be_enabled()
    page.click("#run-button")
    expect(page.locator("#result-source")).to_have_text("Live prediction")
    assert page.evaluate("window.__held[0].init.signal.aborted")
    _release(page, 0, {**PRECOMPUTED_1, "predicted_phenotype": "STALE"})
    page.wait_for_timeout(200)
    expect(page.locator("#result-source")).to_have_text("Live prediction")
    expect(page.locator("#result-phenotype")).not_to_have_text("STALE")


# @spec UI-UI-039
def test_selecting_none_leaves_results_and_pending_request_alone(site):
    site.page.add_init_script(HOLD_SCRIPT)
    page = site.open()
    _select_sample(page, "Sample SNP 1")
    expect(page.locator("#result-source")).to_have_text("Precomputed result for Sample SNP 1")
    _hold(page)
    _upload(page)
    page.click("#run-button")
    _select_sample(page, "None — use uploaded file")
    expect(page.locator("#loading")).to_be_visible()
    assert not page.evaluate("window.__held[0].init.signal.aborted")
    _release(page, 0, LIVE_RESULT)
    expect(page.locator("#result-source")).to_have_text("Live prediction")


# ── Samples ────────────────────────────────────────────────────────────────────


# @spec UI-UI-016, UI-UI-018, UI-UI-019
def test_sample_section_lists_manifest_entries_with_none_selected(site):
    page = site.open()
    section = page.locator("#sample-section")
    expect(section).to_be_visible()
    expect(section).to_have_attribute("open", "")
    expect(section.locator("summary")).to_have_text("Don't have a file? Try a sample")
    labels = page.locator("#sample-options label").all_inner_texts()
    assert [t.strip() for t in labels] == [
        "None — use uploaded file", "Sample SNP 1", "Sample SNP 2", "Sample SNP 3"]
    expect(page.get_by_label("None — use uploaded file", exact=True)).to_be_checked()
    upload_box = page.locator("#vcf-file").bounding_box()
    assert section.bounding_box()["y"] < upload_box["y"], "samples section sits above the upload"


# @spec UI-UI-017
@pytest.mark.parametrize("variant", ["missing", "unparseable", "empty"])
def test_bad_manifest_hides_sample_section_silently(site, variant):
    site.assemble()
    manifest = site.dir / "samples" / "manifest.json"
    if variant == "missing":
        manifest.unlink()
    else:
        manifest.write_text("{not json" if variant == "unparseable" else "[]")
    site.page.goto(f"{site.origin}/index.html")
    expect(site.page.locator("#vcf-file")).to_be_visible()
    expect(site.page.locator("#sample-section")).to_be_hidden()
    expect(site.page.locator("#result-message")).to_be_empty()


# @spec UI-UI-031, UI-UI-032, UI-UI-045
def test_selecting_sample_renders_precomputed_result_without_api_call(site):
    page = site.open()
    _select_sample(page, "Sample SNP 1")
    expect(page.locator("#result-source")).to_have_text("Precomputed result for Sample SNP 1")
    expect(page.locator("#result-phenotype")).to_have_text(YRI_DISPLAY)
    assert site.api_calls("/predict") == []
    shown = site.events("precomputed_result_rendered")
    assert shown and shown[0]["level"] == "INFO" and shown[0]["sample"] == "Sample SNP 1"
    assert shown[0]["phenotype"] == PRECOMPUTED_1["predicted_phenotype"]
    assert shown[0]["confidence"] == PRECOMPUTED_1["confidence_score"]


# @spec UI-UI-033, UI-UI-047, UI-UI-032
def test_missing_precomputed_result_falls_back_to_live_prediction(site):
    page = site.page
    page.route(f"{site.origin}/samples/sample_snp_1.result.json",
               lambda route: route.fulfill(status=404, body="not found"))
    site.open()
    _select_sample(page, "Sample SNP 1")
    expect(page.locator("#result-source")).to_have_text("Live prediction")
    (call,) = site.api_calls("/predict")
    assert json.loads(call["body"]) == {"vcf": (SAMPLES / "sample_snp_1.vcf").read_text()}
    warn = site.events("precomputed_result_unavailable")
    assert warn and warn[0]["level"] == "WARN" and warn[0]["sample"] == "Sample SNP 1"
    sent = site.events("predict_request_sent")[0]
    assert (sent["input"], sent["filename"]) == ("sample", "sample_snp_1.vcf")


# @spec UI-UI-022
def test_sample_fallback_skips_browser_validation(site, tmp_path):
    samples = tmp_path / "odd_samples"
    samples.mkdir()
    (samples / "sample_snp_1.vcf").write_text(_vcf(n_samples=2))  # would fail UI-UI-005
    (samples / "sample_snp_1.result.json").write_text("{not json")  # forces the fallback
    page = site.open(samples=samples)
    _select_sample(page, "Sample SNP 1")
    expect(page.locator("#result-source")).to_have_text("Live prediction")
    assert len(site.api_calls("/predict")) == 1
    assert site.events("upload_invalid") == []


# @spec UI-UI-054
def test_missing_precomputed_result_without_endpoint_shows_not_configured(site):
    page = site.page
    page.route(f"{site.origin}/samples/sample_snp_1.result.json",
               lambda route: route.fulfill(status=404, body="not found"))
    site.open(config=False)
    _select_sample(page, "Sample SNP 1")
    expect(page.locator("#result-message")).to_have_text(NOT_CONFIGURED)
    assert site.api_calls("/predict") == []


# ── Results display ────────────────────────────────────────────────────────────


# @spec UI-UI-011, UI-UI-060
def test_result_shows_label_rounded_confidence_and_progress(site):
    site.predict_response = {"status": 200, "json": {**LIVE_RESULT, "confidence_score": 0.825}}
    page = site.open()
    _upload(page)
    page.click("#run-button")
    expect(page.locator("#result-phenotype")).to_have_text(CHB_DISPLAY)
    expect(page.locator("#result-confidence")).to_have_text("83%")
    expect(page.locator("#result-confidence-bar")).to_have_attribute("value", "83")


# @spec UI-UI-012
def test_marker_table_rows_follow_rank(site):
    page = site.open()
    _select_sample(page, "Sample SNP 1")
    rows = page.locator("#marker-table tbody tr")
    expect(rows).to_have_count(len(PRECOMPUTED_1["top_markers"]))
    first = rows.nth(0).locator("td").all_inner_texts()
    m = min(PRECOMPUTED_1["top_markers"], key=lambda m: m["rank"])
    assert first[:4] == [str(m["rank"]), m["variant_id"], m["chrom"], str(m["pos"])]
    assert float(first[4]) == pytest.approx(m["shap_contribution"], abs=1e-4)


def _chart_markers() -> list[dict]:
    values = [0.30, -0.60, 0.15, -0.05, 0.45]
    return [
        {"variant_id": f"15_{1000 + i}_A_G", "chrom": "15", "pos": 1000 + i, "ref": "A",
         "alt": "G", "shap_contribution": v, "rank": i + 1}
        for i, v in enumerate(values)
    ]


# @spec UI-UI-014
def test_chart_bars_sorted_by_absolute_shap_and_labelled(site):
    site.predict_response = {"status": 200,
                             "json": {**LIVE_RESULT, "top_markers": _chart_markers()}}
    page = site.open()
    _upload(page)
    page.click("#run-button")
    bars = page.locator("#marker-chart g.bar")
    expect(bars).to_have_count(5)
    order = [bars.nth(i).get_attribute("data-variant") for i in range(5)]
    assert order == ["15_1001_A_G", "15_1004_A_G", "15_1000_A_G", "15_1002_A_G", "15_1003_A_G"]
    ys = [float(bars.nth(i).locator("rect").get_attribute("y")) for i in range(5)]
    assert ys == sorted(ys), "largest |SHAP| at the top"
    assert bars.nth(0).locator("text").text_content() == "15_1001_A_G"


# @spec UI-UI-037
def test_chart_bars_diverge_from_zero_axis_and_scale_to_largest(site):
    site.predict_response = {"status": 200,
                             "json": {**LIVE_RESULT, "top_markers": _chart_markers()}}
    page = site.open()
    _upload(page)
    page.click("#run-button")
    axis = page.locator("#marker-chart #zero-axis")
    zero = float(axis.get_attribute("x1"))
    half = float(page.locator("#marker-chart").get_attribute("data-half-width"))
    by_variant = {}
    bars = page.locator("#marker-chart g.bar")
    for i in range(bars.count()):
        rect = bars.nth(i).locator("rect")
        by_variant[bars.nth(i).get_attribute("data-variant")] = (
            float(rect.get_attribute("x")), float(rect.get_attribute("width")))
    for m in _chart_markers():
        x, w = by_variant[m["variant_id"]]
        assert w == pytest.approx(abs(m["shap_contribution"]) / 0.60 * half, rel=1e-3)
        if m["shap_contribution"] > 0:
            assert x == pytest.approx(zero)
        else:
            assert x + w == pytest.approx(zero)
    assert by_variant["15_1001_A_G"][1] == pytest.approx(half)


# ── Log helper ─────────────────────────────────────────────────────────────────


# @spec UI-UI-023
def test_every_log_event_is_one_json_object_on_the_matching_console_method(site):
    site.predict_response = {"status": 500, "json": {"error": "INFERENCE_FAILED", "detail": "x"}}
    page = site.open(config=False)              # ERROR: endpoint not configured
    _upload(page, name="bad.txt")
    page.click("#run-button")                    # WARN: upload invalid
    _select_sample(page, "Sample SNP 1")         # INFO: precomputed rendered
    expect(page.locator("#result")).to_be_visible()
    assert site.logs, "no structured log events captured"
    method = {"INFO": "info", "WARN": "warning", "ERROR": "error"}
    for rec in site.logs:
        assert set(rec) >= {"level", "event"}
        assert rec["_console"] == method[rec["level"]], rec
    assert {r["level"] for r in site.logs} == {"INFO", "WARN", "ERROR"}


# ── Population names — UI-UI-060 ───────────────────────────────────────────────


def _population_table(page: Page) -> dict:
    return page.evaluate("async () => (await import('./populations.js')).POPULATIONS")


# @spec UI-UI-060
def test_population_table_covers_exactly_the_models_26_codes(site):
    page = site.open()
    table = _population_table(page)
    model_codes = set(PRECOMPUTED_1["class_probabilities"])
    assert len(model_codes) == 26
    assert set(table) == model_codes
    assert all(isinstance(v, str) and v and v != k for k, v in table.items())


# @spec UI-UI-060
@pytest.mark.parametrize(
    ("code", "name"),
    [
        ("YRI", "Yoruba in Ibadan, Nigeria"),
        ("CHB", "Han Chinese in Beijing, China"),
        ("CEU", "Utah residents with Northern and Western European ancestry"),
        ("JPT", "Japanese in Tokyo, Japan"),
        ("GBR", "British in England and Scotland"),
        ("TSI", "Toscani in Italia"),
    ],
)
def test_population_names_match_1000_genomes(site, code, name):
    assert _population_table(site.open())[code] == name


# @spec UI-UI-060
@pytest.mark.parametrize(("code", "shown"), [("JPT", "Japanese in Tokyo, Japan (JPT)"),
                                              ("XYZ", "XYZ")])
def test_rendered_population_uses_name_and_code_or_code_alone(site, code, shown):
    site.predict_response = {"status": 200,
                             "json": {**LIVE_RESULT, "predicted_phenotype": code}}
    page = site.open()
    _upload(page)
    page.click("#run-button")
    expect(page.locator("#result-phenotype")).to_have_text(shown)
