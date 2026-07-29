"""Verdict reading, terminal outcome, and the run log."""
import json

from agent_framework import JudgeVerdict

from reflection_demo.judging import RunLog, Verdict, read_verdict, summarize


def test_structured_verdict_is_used_verbatim():
    v = JudgeVerdict(answered=True, reasoning="covers the mandate")
    assert read_verdict(v, "ignored text") == (True, "covers the mandate")


def test_marker_fallback_reads_done():
    assert read_verdict(None, "Looks complete.\nVERDICT: DONE") == (
        True, "Looks complete.\nVERDICT: DONE")


def test_marker_fallback_reads_more():
    answered, _reasoning = read_verdict(None, "Missing the Azure conflict.\nVERDICT: MORE")
    assert answered is False


def test_more_wins_when_both_markers_appear():
    # Ambiguity must keep the loop running, never stop it.
    answered, _ = read_verdict(None, "VERDICT: DONE ... on reflection VERDICT: MORE")
    assert answered is False


def test_markerless_reply_keeps_looping():
    # Fail OPEN: an unreadable verdict costs a pass, it does not end the run.
    # (reflexion fails CLOSED and rejects - see the design spec, section 8.)
    answered, reasoning = read_verdict(None, "I am not sure what to do here.")
    assert answered is False
    assert reasoning == "I am not sure what to do here."


def test_empty_reply_keeps_looping():
    assert read_verdict(None, "") == (False, "")


def test_summarize_answered_on_early_exit():
    verdicts = [Verdict(1, False, "thin"), Verdict(2, True, "good")]
    assert summarize(verdicts, max_passes=5) == ("answered", 2)


def test_summarize_unjudged_when_capped():
    # Cap 3: passes 1 and 2 were judged, pass 3 ran and the judge was never
    # called (max_iterations short-circuits before should_continue).
    verdicts = [Verdict(1, False, "a"), Verdict(2, False, "b")]
    assert summarize(verdicts, max_passes=3) == ("unjudged", 3)


def test_summarize_unjudged_with_no_verdicts_at_all():
    # --max-passes 1: the single pass runs and is never judged.
    assert summarize([], max_passes=1) == ("unjudged", 1)


def test_run_log_records_each_verdict_then_the_outcome(tmp_path):
    path = tmp_path / "reflection_log.jsonl"
    log = RunLog(path)
    log.record(Verdict(1, False, "missing the Azure mandate"))
    log.record(Verdict(2, True, "now cited"))
    outcome, passes = log.finish(max_passes=5, report_path=tmp_path / "report.md")

    assert (outcome, passes) == ("answered", 2)
    lines = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()]
    assert lines[0] == {"pass": 1, "answered": False,
                        "reasoning": "missing the Azure mandate", "judged": True}
    assert lines[1]["answered"] is True
    assert lines[-1]["outcome"] == "answered"
    assert lines[-1]["passes"] == 2


def test_run_log_writes_the_unjudged_pass_when_capped(tmp_path):
    path = tmp_path / "reflection_log.jsonl"
    log = RunLog(path)
    log.record(Verdict(1, False, "thin"))
    outcome, passes = log.finish(max_passes=2, report_path=tmp_path / "report.md")

    assert (outcome, passes) == ("unjudged", 2)
    lines = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()]
    assert lines[1] == {"pass": 2, "answered": None, "reasoning": None, "judged": False}
    assert lines[-1]["outcome"] == "unjudged"
