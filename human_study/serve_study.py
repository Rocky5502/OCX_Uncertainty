#!/usr/bin/env python3
"""Dependency-free local server for the OpenCoderX human developer study."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import html
import json
import os
import re
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "human_study"
FROZEN = STUDY / "frozen"
CONFIG = json.loads((STUDY / "study_config.json").read_text(encoding="utf-8"))
PARTICIPANT_CODE = re.compile(r"^H[0-9]{3}$")
FREQUENCY_LABELS = [
    "Never", "Monthly or less", "Several times per month", "Weekly",
    "Several times per week", "Daily",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_schedule() -> list[dict[str, str]]:
    with (FROZEN / "human_randomization_schedule.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def field(form: dict[str, list[str]], key: str, default: str = "") -> str:
    return str(form.get(key, [default])[0]).strip()


def int_field(form: dict[str, list[str]], key: str, low: int, high: int) -> int:
    value = int(field(form, key))
    if not low <= value <= high:
        raise ValueError(f"{key} must be between {low} and {high}")
    return value


def float_field(form: dict[str, list[str]], key: str, low: float, high: float) -> float:
    value = float(field(form, key))
    if not low <= value <= high:
        raise ValueError(f"{key} must be between {low} and {high}")
    return value


def options(values: list[str], labels: list[str] | None = None) -> str:
    labels = labels or values
    return "".join(
        f'<option value="{html.escape(value)}">{html.escape(label)}</option>'
        for value, label in zip(values, labels)
    )


STYLE = """
:root { color-scheme: light; --ink:#171a1f; --muted:#667085; --line:#d7dce2;
  --blue:#1859a9; --blue-light:#eef5fd; --gold:#8b6508; --gold-light:#fff8df;
  --surface:#fff; --page:#f5f7f9; }
* { box-sizing:border-box; }
body { margin:0; font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  color:var(--ink); background:var(--page); letter-spacing:0; }
header { background:#fff; border-bottom:1px solid var(--line); padding:14px 24px; }
header strong { font-size:17px; } header span { float:right; color:var(--muted); }
main { max-width:1180px; margin:0 auto; padding:24px; }
.band { background:var(--surface); border:1px solid var(--line); border-radius:6px;
  padding:20px; margin-bottom:16px; }
.grid { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:16px; }
h1 { font-size:24px; margin:0 0 12px; } h2 { font-size:17px; margin:0 0 10px; }
p { margin:6px 0 12px; } .muted { color:var(--muted); } .small { font-size:13px; }
label { display:block; font-weight:600; margin:13px 0 5px; }
input,select,textarea { width:100%; border:1px solid #aeb6c2; border-radius:4px;
  background:#fff; color:var(--ink); padding:9px 10px; font:inherit; }
textarea.code { min-height:430px; resize:vertical; font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; tab-size:4; }
pre { white-space:pre-wrap; overflow-wrap:anywhere; background:#f7f8fa; border:1px solid var(--line);
  padding:12px; max-height:330px; overflow:auto; font-size:12px; }
button { border:0; border-radius:4px; background:var(--blue); color:#fff; padding:10px 15px;
  font-weight:650; cursor:pointer; }
button.secondary { background:#596273; } button.danger { background:#9d2a2a; }
.actions { display:flex; gap:10px; align-items:center; margin-top:18px; }
.risk { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; }
.risk div { background:var(--blue-light); border-left:3px solid var(--blue); padding:9px; }
.guidance { background:var(--gold-light); border-left:4px solid var(--gold); padding:12px; }
.progress { height:7px; background:#e5e8ec; margin-top:10px; }
.progress span { display:block; height:100%; background:var(--blue); }
.checks label { font-weight:400; display:flex; gap:8px; align-items:flex-start; }
.checks input { width:auto; margin-top:5px; }
.error { color:#9d2a2a; font-weight:600; }
@media (max-width:800px) { .grid { grid-template-columns:1fr; } .risk { grid-template-columns:1fr 1fr; }
  main { padding:12px; } header span { float:none; display:block; } }
"""


def page(title: str, body: str, mode: str) -> bytes:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>{STYLE}</style></head><body><header><strong>OpenCoderX Review Study</strong>
<span>{html.escape(mode.replace('_', ' ').title())}</span></header><main>{body}</main></body></html>""".encode("utf-8")


class StudyState:
    def __init__(self, data_dir: Path, mode: str, ethics_path: Path | None):
        self.data_dir = data_dir
        self.mode = mode
        self.participants_path = data_dir / "participants.jsonl"
        self.episodes_path = data_dir / "episodes.jsonl"
        self.starts_path = data_dir / "episode_starts.jsonl"
        self.tutorials_path = data_dir / "tutorials.jsonl"
        self.poststudy_path = data_dir / "poststudy.jsonl"
        self.withdrawals_path = data_dir / "withdrawals.jsonl"
        self.mode_label = "SIMULATED_DRY_RUN" if mode == "dry-run" else "EMPIRICAL"
        if mode == "empirical":
            if ethics_path is None or not ethics_path.is_file():
                raise RuntimeError("empirical mode requires --ethics-approval-file")
            approval = json.loads(ethics_path.read_text(encoding="utf-8"))
            required = {
                "status": "APPROVED_FOR_RECRUITMENT",
                "protocol_version": CONFIG["protocol_version"],
            }
            for key, expected in required.items():
                if approval.get(key) != expected:
                    raise RuntimeError(f"ethics approval field {key!r} does not match {expected!r}")
            for key in ("institution", "ethics_body_or_authority", "protocol_identifier", "determination", "determination_date", "principal_investigator", "contact", "compensation", "retention_period"):
                if not str(approval.get(key) or "").strip() or str(approval[key]).startswith("REPLACE"):
                    raise RuntimeError(f"ethics approval field {key!r} is incomplete")
        self.schedule = read_schedule()
        self.stimuli = {row["task_id"]: row for row in read_jsonl(FROZEN / "stimuli_public.jsonl")}

    def participant(self, code: str) -> dict[str, Any] | None:
        return next((row for row in reversed(read_jsonl(self.participants_path)) if row["participant_code"] == code), None)

    def completed(self, code: str) -> list[dict[str, Any]]:
        return [row for row in read_jsonl(self.episodes_path) if row["participant_code"] == code]

    def tutorial_complete(self, code: str) -> bool:
        return any(row["participant_code"] == code and row.get("passed") is True for row in read_jsonl(self.tutorials_path))

    def completed_participant_codes(self) -> set[str]:
        return {str(row["participant_code"]) for row in read_jsonl(self.poststudy_path)}

    def recruitment_closed(self) -> bool:
        return len(self.completed_participant_codes()) >= int(CONFIG["participants_planned"])

    def scheduled(self, code: str) -> list[dict[str, str]]:
        return sorted(
            (row for row in self.schedule if row["participant_code"] == code),
            key=lambda row: int(row["episode_index"]),
        )

    def start_time(self, code: str, episode_index: int) -> tuple[str, float]:
        existing = next((
            row for row in read_jsonl(self.starts_path)
            if row["participant_code"] == code and int(row["episode_index"]) == episode_index
        ), None)
        if existing:
            return str(existing["started_at_utc"]), float(existing["started_epoch"])
        row = {
            "participant_code": code,
            "episode_index": episode_index,
            "started_at_utc": utc_now(),
            "started_epoch": time.time(),
            "study_mode": self.mode_label,
        }
        append_jsonl(self.starts_path, row)
        return str(row["started_at_utc"]), float(row["started_epoch"])


class Handler(BaseHTTPRequestHandler):
    state: StudyState

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def send_html(self, title: str, body: str, status: int = 200) -> None:
        payload = page(title, body, self.state.mode_label)
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(payload)

    def redirect(self, location: str) -> None:
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", location)
        self.end_headers()

    def form(self) -> dict[str, list[str]]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 200_000:
            raise ValueError("submission is too large")
        return parse_qs(self.rfile.read(length).decode("utf-8"), keep_blank_values=True)

    def do_GET(self) -> None:
        route = urlparse(self.path)
        query = parse_qs(route.query)
        if route.path in {"/", "/start"}:
            note = "Synthetic software-validation mode; no human evidence will be created." if self.state.mode == "dry-run" else "Empirical collection is unlocked under the supplied ethics determination."
            if self.state.recruitment_closed():
                note += " The completion target has been reached; only previously enrolled participants may resume."
            self.send_html("Start", f"""
<section class="band"><h1>Repository Code Review Study</h1><p>{html.escape(note)}</p><p class="small muted">On a shared computer, each person must use their own assigned participant code.</p>
<form method="post" action="/start"><label for="participant_code">Participant code</label>
<input id="participant_code" name="participant_code" required pattern="H[0-9]{{3}}" placeholder="H001">
<div class="actions"><button type="submit">Continue</button></div></form></section>""")
            return
        code = str(query.get("participant", [""])[0]).upper()
        if not PARTICIPANT_CODE.fullmatch(code) or not self.state.scheduled(code):
            self.send_html("Invalid participant", '<section class="band"><p class="error">Unknown participant code.</p></section>', 400)
            return
        if route.path == "/task":
            self.render_task(code)
            return
        if route.path == "/tutorial":
            self.render_tutorial(code)
            return
        if route.path == "/poststudy":
            self.render_poststudy(code)
            return
        self.send_html("Not found", '<section class="band"><p>Page not found.</p></section>', 404)

    def do_POST(self) -> None:
        try:
            form = self.form()
            if self.path == "/start":
                code = field(form, "participant_code").upper()
                if not PARTICIPANT_CODE.fullmatch(code) or not self.state.scheduled(code):
                    raise ValueError("unknown participant code")
                if self.state.participant(code):
                    destination = "task" if self.state.tutorial_complete(code) else "tutorial"
                    self.redirect(f"/{destination}?participant={code}")
                else:
                    if self.state.recruitment_closed():
                        raise ValueError("the completion target has been reached; new enrollment is closed")
                    self.render_enrollment(code)
                return
            if self.path == "/enroll":
                self.enroll(form)
                return
            if self.path == "/submit":
                self.submit(form)
                return
            if self.path == "/tutorial-submit":
                self.submit_tutorial(form)
                return
            if self.path == "/finish":
                self.finish_study(form)
                return
            if self.path == "/withdraw":
                code = field(form, "participant_code").upper()
                append_jsonl(self.state.withdrawals_path, {"participant_code": code, "withdrawn_at_utc": utc_now(), "study_mode": self.state.mode_label})
                self.send_html("Withdrawn", '<section class="band"><h1>Participation ended</h1><p>Your withdrawal has been recorded. Contact the research team using the consent information if you want previously collected data removed where applicable.</p><div class="actions"><a href="/start"><button type="button" class="secondary">Next participant</button></a></div></section>')
                return
            self.send_html("Not found", '<section class="band"><p>Page not found.</p></section>', 404)
        except (ValueError, KeyError, TypeError) as exc:
            self.send_html("Submission error", f'<section class="band"><p class="error">{html.escape(str(exc))}</p><p>Please return and check the submitted fields.</p></section>', 400)

    def render_enrollment(self, code: str) -> None:
        role_options = options(
            ["software_engineer", "ai_ml_researcher", "phd_student", "other"],
            ["Software engineer", "AI/ML researcher", "PhD student", "Other"],
        )
        freq = options([str(i) for i in range(6)], FREQUENCY_LABELS)
        self.send_html("Consent and background", f"""
<section class="band"><h1>Consent and Background</h1><p>Review the approved participant information before continuing. Do not enter private code, employer information, credentials, or identifying details.</p>
<form method="post" action="/enroll"><input type="hidden" name="participant_code" value="{code}">
<div class="checks">
<label><input type="checkbox" name="adult" value="yes" required>I am at least 18 years old.</label>
<label><input type="checkbox" name="informed" value="yes" required>I have read the study information and had an opportunity to ask questions.</label>
<label><input type="checkbox" name="voluntary" value="yes" required>I understand participation is voluntary and I may withdraw.</label>
<label><input type="checkbox" name="data_consent" value="yes" required>I understand the data collection and de-identified research use.</label>
<label><input type="checkbox" name="consent" value="yes" required>I consent to participate.</label></div>
<div class="grid"><div><label>Primary role</label><select name="primary_role" required>{role_options}</select>
<label>Total programming years</label><input type="number" step="0.5" min="1" max="60" name="programming_years" required>
<label>Python years</label><input type="number" step="0.5" min="1" max="40" name="python_years" required>
<label>AI coding-tool usage</label><select name="ai_tool_usage" required>{freq}</select></div>
<div><label>Additional roles</label><div class="checks"><label><input type="checkbox" name="additional_roles" value="software_engineer">Software engineer</label><label><input type="checkbox" name="additional_roles" value="ai_ml_researcher">AI/ML researcher</label><label><input type="checkbox" name="additional_roles" value="phd_student">PhD student</label><label><input type="checkbox" name="additional_roles" value="other">Other</label></div>
<label>Code-review frequency</label><select name="code_review_frequency" required>{freq}</select>
<label>Repository-development frequency</label><select name="repository_development_frequency" required>{freq}</select>
<label>Unit-testing familiarity (1--5)</label><input type="number" min="1" max="5" name="testing_familiarity" required>
<label>Dependency-tracing familiarity (1--5)</label><input type="number" min="1" max="5" name="dependency_tracing_familiarity" required></div></div>
<div class="actions"><button type="submit">Begin study</button></div></form></section>""")

    def enroll(self, form: dict[str, list[str]]) -> None:
        code = field(form, "participant_code").upper()
        if not PARTICIPANT_CODE.fullmatch(code) or not self.state.scheduled(code):
            raise ValueError("unknown participant code")
        if self.state.recruitment_closed() and not self.state.participant(code):
            raise ValueError("the completion target has been reached; new enrollment is closed")
        for key in ("adult", "informed", "voluntary", "data_consent", "consent"):
            if field(form, key) != "yes":
                raise ValueError("all consent statements are required")
        record = {
            "record_type": "participant", "study_mode": self.state.mode_label,
            "participant_code": code,
            "assignment_group": int(self.state.scheduled(code)[0]["assignment_group"]),
            "consent": True, "adult": True,
            "primary_role": field(form, "primary_role"),
            "additional_roles": sorted(set(form.get("additional_roles", []))),
            "programming_years": float_field(form, "programming_years", 1, 60),
            "python_years": float_field(form, "python_years", 1, 40),
            "ai_tool_usage": int_field(form, "ai_tool_usage", 0, 5),
            "code_review_frequency": int_field(form, "code_review_frequency", 0, 5),
            "repository_development_frequency": int_field(form, "repository_development_frequency", 0, 5),
            "testing_familiarity": int_field(form, "testing_familiarity", 1, 5),
            "dependency_tracing_familiarity": int_field(form, "dependency_tracing_familiarity", 1, 5),
            "started_at_utc": utc_now(), "completed_at_utc": None, "withdrawn": False,
        }
        if record["primary_role"] not in {"software_engineer", "ai_ml_researcher", "phd_student", "other"}:
            raise ValueError("invalid primary role")
        familiar = (
            record["code_review_frequency"] >= 1
            or record["repository_development_frequency"] >= 1
            or record["testing_familiarity"] >= 2
        )
        if not familiar:
            raise ValueError(
                "eligibility requires familiarity with code review, testing, or repository development"
            )
        append_jsonl(self.state.participants_path, record)
        self.redirect(f"/tutorial?participant={code}")

    def render_tutorial(self, code: str, error: str = "") -> None:
        if not self.state.participant(code):
            self.redirect("/")
            return
        if self.state.tutorial_complete(code):
            self.redirect(f"/task?participant={code}")
            return
        error_html = f'<p class="error">{html.escape(error)}</p>' if error else ""
        self.send_html("Tutorial", f"""
<section class="band"><h1>Tutorial and Practice</h1><p>For every scored task, decide whether the starting function is correct, edit it when needed, and submit one complete Python function. Executable correctness remains hidden until submission.</p>{error_html}</section>
<form method="post" action="/tutorial-submit"><input type="hidden" name="participant_code" value="{code}">
<div class="grid"><section class="band"><h2>Practice 1</h2><p>Repository context specifies <code>safe_divide(a, b, default=0)</code>, which returns <code>default</code> when <code>b == 0</code>.</p><pre>def ratio(a, b):\n    return a / b</pre><label>Is the implementation correct for the repository requirement?</label><select name="practice_one" required><option value="">Select</option><option value="false">No</option><option value="true">Yes</option></select></section>
<section class="band"><h2>Practice 2</h2><p>The target must return a lowercase string after trimming surrounding whitespace.</p><pre>def normalize_name(value):\n    return value.strip().lower()</pre><label>Is the implementation correct?</label><select name="practice_two" required><option value="">Select</option><option value="false">No</option><option value="true">Yes</option></select></section></div>
<section class="band"><p class="small muted">Practice responses are not part of the empirical outcomes.</p><div class="actions"><button type="submit">Check practice</button></div></section></form>""")

    def submit_tutorial(self, form: dict[str, list[str]]) -> None:
        code = field(form, "participant_code").upper()
        if not self.state.participant(code):
            raise ValueError("participant enrollment is required")
        passed = field(form, "practice_one") == "false" and field(form, "practice_two") == "true"
        if not passed:
            self.render_tutorial(code, "Review both examples and try again.")
            return
        append_jsonl(self.state.tutorials_path, {
            "participant_code": code, "passed": True,
            "completed_at_utc": utc_now(), "study_mode": self.state.mode_label,
        })
        self.redirect(f"/task?participant={code}")

    def render_task(self, code: str) -> None:
        if not self.state.participant(code):
            self.redirect("/")
            return
        if not self.state.tutorial_complete(code):
            self.redirect(f"/tutorial?participant={code}")
            return
        completed = self.state.completed(code)
        completed_indices = {int(row["episode_index"]) for row in completed}
        schedule = self.state.scheduled(code)
        next_row = next((row for row in schedule if int(row["episode_index"]) not in completed_indices), None)
        if next_row is None:
            self.redirect(f"/poststudy?participant={code}")
            return
        index = int(next_row["episode_index"])
        stimulus = self.state.stimuli[next_row["task_id"]]
        condition = next_row["condition"]
        started_at, _ = self.state.start_time(code, index)
        condition_panel = ""
        if condition in {"uncertainty_display", "targeted_guidance"}:
            risks = stimulus["source_risks"]
            condition_panel = '<section class="band"><h2>Uncertainty trace</h2><div class="risk">' + "".join(
                f'<div><strong>{html.escape(name.replace("_", " ").title())}</strong><br>{100*float(risks[name]):.1f}% risk</div>'
                for name in ("api", "context", "similar_code", "generation")
            ) + f'</div><p class="small muted">Aggregate review risk: {100*float(stimulus["aggregate_risk"]):.1f}%.</p></section>'
        if condition == "targeted_guidance":
            condition_panel += f'<section class="band"><h2>Recommended review action</h2><div class="guidance">{html.escape(stimulus["targeted_guidance"])}</div></section>'
        usefulness = '<option value="">Not shown</option>' if condition == "generic_review" else options([str(i) for i in range(1, 6)])
        progress = 100.0 * (index - 1) / 12.0
        self.send_html(f"Task {index}", f"""
<section class="band"><h1>Task {index} of 12</h1><p class="muted">Repository: {html.escape(stimulus['repository'])} · Target: {html.escape(stimulus['function_name'])}</p><div class="progress"><span style="width:{progress:.1f}%"></span></div></section>
{condition_panel}
<div class="grid"><div><section class="band"><h2>Target function</h2><pre>{html.escape(stimulus['task_text'])}</pre></section><section class="band"><h2>Repository context</h2><pre>{html.escape(stimulus['repository_context'] or '(No additional context block)')}</pre></section><section class="band"><h2>Retrieved evidence</h2><pre>{html.escape(stimulus['retrieved_evidence'] or '(No retrieved evidence)')}</pre></section></div>
<div><section class="band"><form method="post" action="/submit"><input type="hidden" name="participant_code" value="{code}"><input type="hidden" name="episode_index" value="{index}"><input type="hidden" name="task_id" value="{html.escape(stimulus['task_id'])}"><input type="hidden" name="condition" value="{condition}"><input type="hidden" name="started_at_utc" value="{html.escape(started_at)}">
<label>Starting implementation</label><textarea class="code" name="final_code" required spellcheck="false">{html.escape(stimulus['starting_code'])}</textarea>
<label>Was the starting implementation correct?</label><select name="starting_judgment_correct" required><option value="false">No</option><option value="true">Yes</option></select>
<label>Confidence in starting judgment (0--100)</label><input type="number" min="0" max="100" name="starting_confidence" required>
<label>Confidence in submitted implementation (0--100)</label><input type="number" min="0" max="100" name="final_confidence" required>
<div class="grid"><div><label>Difficulty (1--5)</label><input type="number" min="1" max="5" name="difficulty" required></div><div><label>Guidance usefulness (1--5)</label><select name="guidance_usefulness" {'required' if condition != 'generic_review' else ''}>{usefulness}</select></div></div>
<label>Main action</label><select name="action_category" required>{options(['accepted','edited_api','used_context','rejected_similar_code','changed_logic','other'], ['Accepted unchanged','Edited API use','Used repository context','Rejected similar-code assumptions','Changed implementation logic','Other'])}</select>
<label>Optional note</label><textarea name="client_notes" maxlength="1000" rows="3"></textarea>
<div class="actions"><button type="submit">Submit task</button></div></form>
<form method="post" action="/withdraw"><input type="hidden" name="participant_code" value="{code}"><div class="actions"><button type="submit" class="danger">Withdraw</button></div></form></section></div></div>""")

    def submit(self, form: dict[str, list[str]]) -> None:
        code = field(form, "participant_code").upper()
        index = int_field(form, "episode_index", 1, 12)
        scheduled = next((row for row in self.state.scheduled(code) if int(row["episode_index"]) == index), None)
        if scheduled is None:
            raise ValueError("episode is not scheduled")
        if scheduled["task_id"] != field(form, "task_id") or scheduled["condition"] != field(form, "condition"):
            raise ValueError("submitted task or condition does not match frozen schedule")
        if any(int(row["episode_index"]) == index for row in self.state.completed(code)):
            raise ValueError("episode has already been submitted")
        stimulus = self.state.stimuli[scheduled["task_id"]]
        started_at, started_epoch = self.state.start_time(code, index)
        usefulness_raw = field(form, "guidance_usefulness")
        usefulness = None if usefulness_raw == "" else int(usefulness_raw)
        if scheduled["condition"] != "generic_review" and usefulness is None:
            raise ValueError("guidance usefulness is required when guidance is shown")
        final_code = field(form, "final_code")
        if not final_code:
            raise ValueError("final code is required")
        record = {
            "record_type": "episode", "study_mode": self.state.mode_label,
            "participant_code": code, "episode_index": index,
            "task_id": scheduled["task_id"], "condition": scheduled["condition"],
            "assignment_group": int(scheduled["assignment_group"]),
            "starting_code_sha256": stimulus["starting_code_sha256"],
            "starting_judgment_correct": field(form, "starting_judgment_correct").lower() == "true",
            "starting_confidence": int_field(form, "starting_confidence", 0, 100),
            "final_code": final_code,
            "final_confidence": int_field(form, "final_confidence", 0, 100),
            "difficulty": int_field(form, "difficulty", 1, 5),
            "guidance_usefulness": usefulness,
            "action_category": field(form, "action_category"),
            "started_at_utc": started_at, "submitted_at_utc": utc_now(),
            "elapsed_seconds": min(720.0, max(0.0, time.time() - started_epoch)),
            "client_notes": field(form, "client_notes")[:1000],
        }
        if record["action_category"] not in {"accepted", "edited_api", "used_context", "rejected_similar_code", "changed_logic", "other"}:
            raise ValueError("invalid action category")
        append_jsonl(self.state.episodes_path, record)
        self.redirect(f"/task?participant={code}")

    def render_poststudy(self, code: str) -> None:
        if len(self.state.completed(code)) < 12:
            self.redirect(f"/task?participant={code}")
            return
        existing = next((row for row in read_jsonl(self.state.poststudy_path) if row["participant_code"] == code), None)
        if existing:
            self.send_html("Complete", '<section class="band"><h1>Session complete</h1><p>Your responses have been recorded. Thank you.</p><div class="actions"><a href="/start"><button type="button" class="secondary">Next participant</button></a></div></section>')
            return
        self.send_html("Post-study questionnaire", f"""
<section class="band"><h1>Post-Study Questionnaire</h1><form method="post" action="/finish"><input type="hidden" name="participant_code" value="{code}">
<div class="grid"><div><label>Mental demand (0--20)</label><input type="number" min="0" max="20" name="mental_demand" required><label>Effort (0--20)</label><input type="number" min="0" max="20" name="effort" required></div><div><label>Frustration (0--20)</label><input type="number" min="0" max="20" name="frustration" required><label>Temporal demand (0--20)</label><input type="number" min="0" max="20" name="temporal_demand" required></div></div>
<label>Which information was most useful and why?</label><textarea name="most_useful" maxlength="2000" required></textarea>
<label>When did uncertainty information feel misleading?</label><textarea name="misleading" maxlength="2000" required></textarea>
<label>Would you use this guidance during real code review? (1--5)</label><input type="number" min="1" max="5" name="adoption_intent" required>
<div class="actions"><button type="submit">Finish study</button></div></form></section>""")

    def finish_study(self, form: dict[str, list[str]]) -> None:
        code = field(form, "participant_code").upper()
        if len(self.state.completed(code)) < 12:
            raise ValueError("all twelve tasks must be completed first")
        record = {
            "record_type": "poststudy", "study_mode": self.state.mode_label,
            "participant_code": code,
            "mental_demand": int_field(form, "mental_demand", 0, 20),
            "effort": int_field(form, "effort", 0, 20),
            "frustration": int_field(form, "frustration", 0, 20),
            "temporal_demand": int_field(form, "temporal_demand", 0, 20),
            "most_useful": field(form, "most_useful")[:2000],
            "misleading": field(form, "misleading")[:2000],
            "adoption_intent": int_field(form, "adoption_intent", 1, 5),
            "submitted_at_utc": utc_now(),
        }
        append_jsonl(self.state.poststudy_path, record)
        append_jsonl(self.state.participants_path, {
            **self.state.participant(code), "completed_at_utc": utc_now()
        })
        self.send_html("Complete", '<section class="band"><h1>Session complete</h1><p>Your responses have been recorded. Thank you.</p><div class="actions"><a href="/start"><button type="button" class="secondary">Next participant</button></a></div></section>')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", type=Path, default=STUDY / "data")
    parser.add_argument("--mode", choices=("dry-run", "empirical"), default="dry-run")
    parser.add_argument("--ethics-approval-file", type=Path)
    args = parser.parse_args()
    state = StudyState(args.data_dir, args.mode, args.ethics_approval_file)
    Handler.state = state
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"OpenCoderX human study ({state.mode_label}) at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
