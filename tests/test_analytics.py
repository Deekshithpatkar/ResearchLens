import pytest
import json
from backend.profile_utils import clean_json_response, parse_with_repair
from backend.analytics_utils import determine_relevant_fields, execute_global_analytics

@pytest.fixture
def anyio_backend():
    return 'asyncio'

def test_clean_json_response_markdown_fences():
    raw_markdown = "```json\n{\n  \"title\": \"BERT\"\n}\n```"
    cleaned = clean_json_response(raw_markdown)
    assert cleaned == "{\n  \"title\": \"BERT\"\n}"

def test_clean_json_response_no_fences():
    raw_text = "{\n  \"title\": \"BERT\"\n}"
    cleaned = clean_json_response(raw_text)
    assert cleaned == raw_text

def test_parse_with_repair_trailing_comma():
    malformed_json = "{\n  \"keys\": [\"dataset1\", \"dataset2\"],\n}"
    parsed = parse_with_repair(malformed_json)
    assert parsed == {"keys": ["dataset1", "dataset2"]}

def test_determine_relevant_fields_fallback():
    # If LLM classification fails/times out, it should default to all allowed fields
    fields = determine_relevant_fields("unclassified query that will fail mockup")
    assert isinstance(fields, list)
    assert len(fields) > 0
    assert "methodology" in fields

