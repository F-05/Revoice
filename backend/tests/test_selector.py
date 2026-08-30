"""Tests for the constrained suggestion-first selector.

The deployed policy: the selector NEVER silently replaces the production
transcript. It either keeps A0 (KEEP_A0) or keeps A0 while suggesting a real
alternative hypothesis (UNCERTAIN + suggested_text). Automatic SWITCH does not
exist in new-user mode. Every surfaced string must be a member of H1-H5.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ARTIFACT = Path(__file__).resolve().parent.parent / "app/assets/revoice_selector_v1.json"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def make_settings(tmp_path, **overrides):
    from app.config import Settings

    values = {"repair_backend": "selector",
              "repair_model_path": str(ARTIFACT)}
    values.update(overrides)
    return Settings(_env_file=None, **values)


def make_service(tmp_path, **overrides):
    from app.services.selector_repair import SelectorRepairService

    return SelectorRepairService(make_settings(tmp_path, **overrides))


def candidates(*texts, a0_index=0, scores=None):
    from app.services.nbest import Candidate, normalize_text

    out = []
    for i, t in enumerate(texts):
        out.append(Candidate(
            display=t, normalized=normalize_text(t), is_a0=(i == a0_index),
            ct2_score=None if i == a0_index else (scores[i] if scores else -1.0 - i)))
    return out


GLASSES = ["bring me my classes", "bring me my glasses",
           "bring me my glass", "bring me the glasses"]


# --------------------------------------------------------------------------
# decision policy
# --------------------------------------------------------------------------
def test_artifact_loads_and_declares_suggestion_first(tmp_path):
    service = make_service(tmp_path)
    assert service.active
    art = json.loads(ARTIFACT.read_text())
    assert art["policy"]["auto_switch_enabled"] is False
    assert art["policy"]["suggestion_tau"] == 0.35
    assert art["whisper_model"] == "medium.en"


def test_single_candidate_keeps_a0(tmp_path):
    service = make_service(tmp_path)
    out = service.decide(candidates("turn on the light"), a0_conf=0.9)
    assert out.decision == "KEEP_A0"
    assert out.final_normalized == "turn on the light"
    assert out.suggested_text is None


def test_never_emits_switch_even_with_max_margin(tmp_path):
    """No selector condition silently changes the user's speech."""
    service = make_service(tmp_path)
    for conf in (None, 0.05, 0.5, 0.95):
        out = service.decide(candidates(*GLASSES), a0_conf=conf)
        assert out.decision in {"KEEP_A0", "UNCERTAIN"}
        # the textual output is ALWAYS the A0 sentence
        assert out.final_normalized == "bring me my classes"


def test_uncertain_keeps_a0_and_suggests_from_list(tmp_path):
    service = make_service(tmp_path)
    out = service.decide(candidates(*GLASSES), a0_conf=0.3)
    from app.services.nbest import normalize_text

    norms = {normalize_text(t) for t in GLASSES}
    assert normalize_text(out.final_display) == "bring me my classes"
    if out.decision == "UNCERTAIN":
        assert out.suggested_text is not None
        assert normalize_text(out.suggested_text) in norms          # H1-H5 only
        assert out.alternatives, "recommendation must be in alternatives"
        # EXACT string identity: the suggestion IS alternatives[0], and any
        # confirmed final output equals one of these canonical strings.
        assert out.alternatives[0] == out.suggested_text
        assert out.suggestion_strength in {"strong", "weak"}


def test_all_surfaced_strings_belong_to_the_list(tmp_path):
    from app.services.nbest import normalize_text

    service = make_service(tmp_path)
    lists = [GLASSES, ["hello there", "hello here"],
             ["one", "won", "wan", "own", "on"]]
    for texts in lists:
        out = service.decide(candidates(*texts), a0_conf=0.4)
        norms = {normalize_text(t) for t in texts}
        assert out.final_normalized in norms
        displays = {c.display for c in candidates(*texts)}
        for surfaced in ([out.suggested_text] if out.suggested_text else []) + out.alternatives:
            assert normalize_text(surfaced) in norms
            # no post-selection rewriting: surfaced strings are candidate
            # display strings verbatim
            assert surfaced in displays or surfaced == out.final_display


def test_duplicate_free_and_short_lists(tmp_path):
    service = make_service(tmp_path)
    out = service.decide(candidates("go home", "go rome"), a0_conf=0.5)
    assert out.n_candidates == 2
    assert out.decision in {"KEEP_A0", "UNCERTAIN"}


# --------------------------------------------------------------------------
# fail-closed artifact validation
# --------------------------------------------------------------------------
def test_missing_artifact_disables_selector(tmp_path):
    service = make_service(tmp_path, repair_model_path=str(tmp_path / "missing.json"))
    assert not service.active
    assert service.select(tmp_path / "x.wav", None, None) is None


@pytest.mark.parametrize("mutation", [
    lambda a: a.update(model_version="revoice_selector_v99"),
    lambda a: a.update(feature_version="d3-22f-v2"),
    lambda a: a.update(features=a["features"][:-1]),                # count
    lambda a: a.update(features=list(reversed(a["features"]))),    # order
    lambda a: a.update(scaler_mean=a["scaler_mean"][:-1]),         # dims
    lambda a: a["policy"].update(suggestion_tau=7.0),              # invalid tau
    lambda a: a["policy"].update(auto_switch_enabled=True),        # unsupported
])
def test_malformed_artifact_fails_closed(tmp_path, mutation):
    art = json.loads(ARTIFACT.read_text())
    mutation(art)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(art))
    service = make_service(tmp_path, repair_model_path=str(bad))
    assert not service.active


def test_unparseable_artifact_fails_closed(tmp_path):
    bad = tmp_path / "garbage.json"
    bad.write_text("{not json")
    assert not make_service(tmp_path, repair_model_path=str(bad)).active


# --------------------------------------------------------------------------
# route-level behaviour (mock ASR: selector must fall back cleanly)
# --------------------------------------------------------------------------
def _post(client, wav_bytes):
    response = client.post("/process-speech",
                           files={"audio": ("r.wav", wav_bytes, "audio/wav")})
    assert response.status_code == 200, response.text
    return response.json()


def test_selector_backend_with_mock_asr_still_transcribes(
        client, wav_bytes, monkeypatch):
    """Selector cannot run against the mock engine -> legacy behaviour, no error."""
    from app.config import get_settings
    from app.dependencies import reset_asr_service

    monkeypatch.setenv("REPAIR_BACKEND", "selector")
    monkeypatch.setenv("REPAIR_MODEL_PATH", str(ARTIFACT))
    get_settings.cache_clear(); reset_asr_service()

    body = _post(client, wav_bytes)
    assert body["status"] == "success"
    assert body["raw_transcript"] == "could you get me some water"
    assert body["repair_available"] is False


def test_repair_backend_none_reproduces_a0_only(client, wav_bytes, monkeypatch):
    from app.config import get_settings
    from app.dependencies import reset_asr_service

    monkeypatch.setenv("REPAIR_BACKEND", "none")
    get_settings.cache_clear(); reset_asr_service()

    body = _post(client, wav_bytes)
    assert body["status"] == "success"
    assert body["repaired_text"] is None      # NoOp leaves the transcript alone
    assert body["repair_available"] is False
    assert body["alternatives"] == []


def test_selector_exception_never_breaks_transcription(
        client, wav_bytes, monkeypatch):
    from app.config import get_settings
    from app.dependencies import reset_asr_service
    from app.services import selector_repair

    monkeypatch.setenv("REPAIR_BACKEND", "selector")
    monkeypatch.setenv("REPAIR_MODEL_PATH", str(ARTIFACT))
    get_settings.cache_clear(); reset_asr_service()

    def boom(self, *a, **k):
        raise RuntimeError("synthetic selector failure")

    monkeypatch.setattr(selector_repair.SelectorRepairService, "decide", boom)
    body = _post(client, wav_bytes)
    assert body["status"] == "success"
    assert body["raw_transcript"] == "could you get me some water"


def test_medium_en_is_the_documented_repair_configuration():
    env = (Path(__file__).resolve().parent.parent / ".env").read_text()
    assert "WHISPER_MODEL=medium.en" in env
    assert "REPAIR_BACKEND=selector" in env


def test_margin_override_is_logged(tmp_path, caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        service = make_service(tmp_path, repair_switch_margin=0.5)
    assert service.active
    assert any("override active" in r.message for r in caplog.records)
