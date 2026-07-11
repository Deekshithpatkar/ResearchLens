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

@pytest.mark.anyio
async def test_execute_cluster_analysis_grouping():
    import numpy as np
    from unittest.mock import patch, AsyncMock
    
    # Mock data
    mock_metadata = {
        "papers": {
            "Paper_A": {"filename": "Paper_A.pdf"},
            "Paper_B": {"filename": "Paper_B.pdf"}
        }
    }
    
    # Create mock embeddings with high cosine similarity
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.95, 0.05, 0.0])
    
    with patch("backend.analytics_utils.load_metadata", return_value=mock_metadata), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("numpy.load") as mock_load, \
         patch("backend.analytics_utils.load_profile", return_value={"title": "Mock Paper"}), \
         patch("google.generativeai.GenerativeModel") as mock_model_class:
         
        mock_load.side_effect = [np.array([v1]), np.array([v2])]
        
        mock_response = AsyncMock()
        mock_response.text = '{"name": "Mock Theme", "description": "Mock Description"}'
        
        mock_model = AsyncMock()
        mock_model.generate_content_async.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        from backend.analytics_utils import execute_cluster_analysis
        result = await execute_cluster_analysis()
        
        assert "clusters" in result
        assert len(result["clusters"]) == 1
        cluster = result["clusters"][0]
        assert cluster["name"] == "Mock Theme"
        assert sorted(cluster["papers"]) == ["Paper_A", "Paper_B"]
