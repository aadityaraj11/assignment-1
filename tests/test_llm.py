import json
import pytest
from unittest.mock import patch, MagicMock
import requests

from app.schemas import MetadataExtraction
from app.llm.extractor import build_extraction_prompt
from app.retrieval.training_data import TrainingExample
from app.llm.provider import get_provider, MockProvider, GeminiRESTProvider


def test_metadata_extraction_schema():
    m = MetadataExtraction()
    assert m.agreement_value is None
    assert m.party_one is None
    
    m2 = MetadataExtraction(agreement_end_date="31.11.2009")
    assert m2.agreement_end_date == "31.11.2009"


def test_build_extraction_prompt():
    ex = TrainingExample(
        file_name="f1",
        document_text="contract 1",
        metadata={"agreement_value": "100"}
    )
    prompt = build_extraction_prompt("new target doc", [ex])
    
    assert "FIELD DEFINITIONS" in prompt
    assert "contract 1" in prompt
    assert '"agreement_value": "100"' in prompt
    assert "new target doc" in prompt


def test_mock_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    provider = get_provider()
    assert isinstance(provider, MockProvider)
    
    res = provider.generate_structured("prompt", MetadataExtraction)
    assert res["agreement_value"] == "12000"
    assert "party_one" in res


def test_gemini_rest_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY environment variable not set"):
        GeminiRESTProvider()


def test_gemini_rest_provider_check_availability_success(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "TEST_KEY_123")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    provider = GeminiRESTProvider()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "OK"

    with patch.object(provider.session, "get", return_value=mock_resp):
        ok, status, msg = provider.check_availability()
        assert ok is True
        assert status == 200
        assert msg == "OK"


def test_gemini_rest_provider_check_availability_failure(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "SECRET_API_KEY_456")
    provider = GeminiRESTProvider()

    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.text = "Model not found for key SECRET_API_KEY_456"

    with patch.object(provider.session, "get", return_value=mock_resp):
        ok, status, msg = provider.check_availability()
        assert ok is False
        assert status == 404
        assert "[REDACTED]" in msg
        assert "SECRET_API_KEY_456" not in msg


def test_gemini_rest_provider_success(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "TEST_SECRET_KEY")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    provider = GeminiRESTProvider()
    assert provider.model == "gemini-2.5-flash"
    
    mock_payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "```json\n{\n  \"agreement_value\": \"5000\",\n  \"agreement_start_date\": \"01.01.2024\",\n  \"agreement_end_date\": \"31.12.2024\",\n  \"renewal_notice_days\": \"60\",\n  \"party_one\": \"Alice\",\n  \"party_two\": \"Bob\"\n}\n```"
                        }
                    ]
                }
            }
        ]
    }
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = json.dumps(mock_payload).encode("utf-8")
    mock_resp.json.return_value = mock_payload

    with patch.object(provider.session, "post", return_value=mock_resp):
        res = provider.generate_structured("Test prompt", MetadataExtraction)
        assert res["agreement_value"] == "5000"
        assert res["party_one"] == "Alice"
        assert res["party_two"] == "Bob"


def test_gemini_rest_provider_http_error_redacts_secrets(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "SECRET_KEY_999")
    provider = GeminiRESTProvider()
    
    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_resp.text = '{"error": "Invalid API key SECRET_KEY_999"}'
    
    with patch.object(provider.session, "post", return_value=mock_resp):
        with pytest.raises(RuntimeError) as exc_info:
            provider.generate_structured("Test prompt", MetadataExtraction)
        
        err_msg = str(exc_info.value)
        assert "[REDACTED]" in err_msg
        assert "SECRET_KEY_999" not in err_msg


def test_gemini_rest_provider_timeout(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "SECRET_KEY_123")
    provider = GeminiRESTProvider()
    
    with patch.object(provider.session, "post", side_effect=requests.Timeout("timed out with secret SECRET_KEY_123")):
        with pytest.raises(RuntimeError) as exc_info:
            provider.generate_structured("Test prompt", MetadataExtraction)
        err_msg = str(exc_info.value)
        assert "Request Timeout" in err_msg
        assert "[REDACTED]" in err_msg
        assert "SECRET_KEY_123" not in err_msg


def test_gemini_rest_provider_invalid_json(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "TEST_KEY")
    provider = GeminiRESTProvider()
    
    mock_payload = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Not a valid JSON output"}]
                }
            }
        ]
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = json.dumps(mock_payload).encode("utf-8")
    mock_resp.json.return_value = mock_payload
    
    with patch.object(provider.session, "post", return_value=mock_resp):
        with pytest.raises(ValueError, match="Failed to decode JSON"):
            provider.generate_structured("Test prompt", MetadataExtraction)

