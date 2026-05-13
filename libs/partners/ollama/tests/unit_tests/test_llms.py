"""Test Ollama Chat API wrapper."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from langchain_ollama import OllamaLLM

MODEL_NAME = "llama3.1"


def test_initialization() -> None:
    """Test integration initialization."""
    OllamaLLM(model=MODEL_NAME)


def test_model_params() -> None:
    """Test standard tracing params"""
    llm = OllamaLLM(model=MODEL_NAME)
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "ollama",
        "ls_model_type": "llm",
        "ls_model_name": MODEL_NAME,
    }

    llm = OllamaLLM(model=MODEL_NAME, num_predict=3)
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "ollama",
        "ls_model_type": "llm",
        "ls_model_name": MODEL_NAME,
        "ls_max_tokens": 3,
    }


@patch("langchain_ollama.llms.validate_model")
def test_validate_model_on_init(mock_validate_model: Any) -> None:
    """Test that the model is validated on initialization when requested."""
    OllamaLLM(model=MODEL_NAME, validate_model_on_init=True)
    mock_validate_model.assert_called_once()
    mock_validate_model.reset_mock()

    OllamaLLM(model=MODEL_NAME, validate_model_on_init=False)
    mock_validate_model.assert_not_called()
    OllamaLLM(model=MODEL_NAME)
    mock_validate_model.assert_not_called()


def test_reasoning_aggregation() -> None:
    """Test that reasoning chunks are aggregated into final response."""
    llm = OllamaLLM(model=MODEL_NAME, reasoning=True)
    prompts = ["some prompt"]
    mock_stream = [
        {"thinking": "I am thinking.", "done": False},
        {"thinking": " Still thinking.", "done": False},
        {"response": "Final Answer.", "done": True},
    ]

    with patch.object(llm, "_create_generate_stream") as mock_stream_method:
        mock_stream_method.return_value = iter(mock_stream)
        result = llm.generate(prompts)

    assert result.generations[0][0].generation_info is not None
    assert (
        result.generations[0][0].generation_info["thinking"]
        == "I am thinking. Still thinking."
    )


def test_create_generate_stream_raises_when_client_none() -> None:
    """Test that _create_generate_stream raises RuntimeError when client is None."""
    with patch("langchain_ollama.llms.Client") as mock_client_class:
        mock_client_class.return_value = MagicMock()
        llm = OllamaLLM(model="test-model")
        llm._client = None  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="sync client is not initialized"):
            list(llm._create_generate_stream("Hello"))


async def test_acreate_generate_stream_raises_when_client_none() -> None:
    """Test that _acreate_generate_stream raises RuntimeError when client is None."""
    with patch("langchain_ollama.llms.AsyncClient") as mock_client_class:
        mock_client_class.return_value = MagicMock()
        llm = OllamaLLM(model="test-model")
        llm._async_client = None  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="async client is not initialized"):
            async for _ in llm._acreate_generate_stream("Hello"):
                pass


def test_stream_finish_reason_only_on_final_chunk() -> None:
    """`finish_reason` is set only on the final chunk and uses `done_reason`."""
    llm = OllamaLLM(model=MODEL_NAME)
    mock_stream = [
        {"response": "Hi", "done": False},
        {"response": " there", "done": False},
        {"response": "!", "done": True, "done_reason": "stop"},
    ]
    with patch.object(llm, "_create_generate_stream") as mock_stream_method:
        mock_stream_method.return_value = iter(mock_stream)
        chunks = list(llm._stream("prompt"))

    assert len(chunks) == 3
    assert "finish_reason" not in (chunks[0].generation_info or {})
    assert "finish_reason" not in (chunks[1].generation_info or {})
    assert chunks[2].generation_info is not None
    assert chunks[2].generation_info["finish_reason"] == "stop"


def test_stream_finish_reason_uses_done_reason_not_stop_tokens() -> None:
    """`finish_reason` reflects Ollama's `done_reason`, not the configured `stop`."""
    llm = OllamaLLM(model=MODEL_NAME, stop=["END"])
    mock_stream = [
        {"response": "Hi", "done": True, "done_reason": "length"},
    ]
    with patch.object(llm, "_create_generate_stream") as mock_stream_method:
        mock_stream_method.return_value = iter(mock_stream)
        chunks = list(llm._stream("prompt"))

    assert chunks[0].generation_info is not None
    assert chunks[0].generation_info["finish_reason"] == "length"


async def test_astream_finish_reason_only_on_final_chunk() -> None:
    """Async: `finish_reason` is set only on the final chunk and uses `done_reason`."""
    llm = OllamaLLM(model=MODEL_NAME)
    mock_stream = [
        {"response": "Hi", "done": False},
        {"response": " there", "done": False},
        {"response": "!", "done": True, "done_reason": "stop"},
    ]

    async def aiter_mock() -> Any:
        for item in mock_stream:
            yield item

    with patch.object(llm, "_acreate_generate_stream") as mock_stream_method:
        mock_stream_method.return_value = aiter_mock()
        chunks = [chunk async for chunk in llm._astream("prompt")]

    assert len(chunks) == 3
    assert "finish_reason" not in (chunks[0].generation_info or {})
    assert "finish_reason" not in (chunks[1].generation_info or {})
    assert chunks[2].generation_info is not None
    assert chunks[2].generation_info["finish_reason"] == "stop"
