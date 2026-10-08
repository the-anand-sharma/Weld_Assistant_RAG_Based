"""Tests for the agent's tools. They call the tool functions directly,
so no Gemini API key is needed. Run with:  pytest
"""

from types import SimpleNamespace

from google.genai import types

from weld_agent.tools import analyze_weld_csv, check_weld_parameters, search_wps


def chat_with(*parts):
    """A minimal stand-in for ADK's ToolContext: a chat session whose only
    event is one user message containing the given parts."""
    event = SimpleNamespace(content=types.Content(role="user", parts=list(parts)))
    return SimpleNamespace(session=SimpleNamespace(events=[event]))


def test_check_parameters_all_in_range():
    result = check_weld_parameters("WPS-2026-001", "root", 180, 22, 15)
    assert result["overall"] == "PASS"


def test_check_parameters_flags_high_current():
    result = check_weld_parameters("WPS-2026-001", "root", 230, 22, 15)
    assert result["overall"] == "FAIL"
    assert result["checks"]["current_A"]["result"] == "FAIL"
    assert result["checks"]["voltage_V"]["result"] == "PASS"


def test_check_parameters_unknown_wps_is_an_error():
    result = check_weld_parameters("WPS-9999-999", "root", 180, 22, 15)
    assert result["status"] == "error"


def test_analyze_sample_csv_finds_out_of_range_samples():
    result = analyze_weld_csv("sample_weld_log.csv", "WPS-2026-001", "root", chat_with())
    assert result["status"] == "success"
    assert result["total_samples"] == 120
    assert result["signals"]["current_A"]["samples_out_of_range"] > 0
    assert result["signals"]["voltage_V"]["samples_out_of_range"] > 0


def test_analyze_uploaded_csv():
    csv_bytes = b"timestamp_s,current_A,voltage_V\n0.0,180,22\n0.5,250,22\n"
    upload = types.Part.from_bytes(data=csv_bytes, mime_type="text/csv")
    result = analyze_weld_csv("uploaded", "WPS-2026-001", "root", chat_with(upload))
    assert result["total_samples"] == 2
    assert result["signals"]["current_A"]["out_of_range_times_s"] == [0.5]


def test_analyze_without_an_upload_is_an_error():
    result = analyze_weld_csv("uploaded", "WPS-2026-001", "root", chat_with())
    assert result["status"] == "error"


def test_search_wps_finds_preheat_section():
    result = search_wps("What is the minimum preheat temperature?")
    sources = [match["source"] for match in result["results"]]
    assert "weld_data.txt > PREHEAT AND INTERPASS" in sources
