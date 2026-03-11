"""Tests for retry logic with exponential backoff."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from deepseek_ocr.backends.base import Backend, TransientError
from deepseek_ocr.backends.ollama import OllamaBackend


class TestTransientError:
    """Tests for TransientError exception."""

    def test_transient_error_message(self) -> None:
        err = TransientError("timeout happened")
        assert str(err) == "timeout happened"

    def test_transient_error_with_original(self) -> None:
        original = TimeoutError("conn timeout")
        err = TransientError("wrapped", original=original)
        assert err.original is original


class TestRetryLogic:
    """Tests for Backend._retry method."""

    def _make_backend(self, max_retries: int = 3, retry_delay: float = 0.0) -> OllamaBackend:
        """Create an OllamaBackend with retry delay=0 for fast tests."""
        with patch.object(OllamaBackend, "__init__", lambda self, **kw: None):
            backend = OllamaBackend.__new__(OllamaBackend)
        backend.model_name = "test"
        backend.max_dimension = None
        backend.max_retries = max_retries
        backend.retry_delay = retry_delay
        backend.model = True
        backend.ollama_url = "http://localhost:11434"
        return backend

    def test_retry_on_transient_then_success(self) -> None:
        """Should retry on transient errors and return on eventual success."""
        backend = self._make_backend()
        func = MagicMock(
            side_effect=[
                TransientError("timeout"),
                TransientError("connection lost"),
                "success result",
            ]
        )

        result = backend._retry(func, "arg1", key="val")

        assert result == "success result"
        assert func.call_count == 3
        func.assert_called_with("arg1", key="val")

    def test_max_retries_exhausted(self) -> None:
        """Should raise RuntimeError after max_retries + 1 attempts."""
        backend = self._make_backend(max_retries=2)
        func = MagicMock(side_effect=TransientError("always fails"))

        with pytest.raises(RuntimeError, match="Max retries.*exhausted"):
            backend._retry(func)

        # max_retries=2 means 3 total attempts (initial + 2 retries)
        assert func.call_count == 3

    def test_no_retry_on_permanent_error(self) -> None:
        """Should not retry on non-TransientError exceptions."""
        backend = self._make_backend()
        func = MagicMock(side_effect=RuntimeError("bad request"))

        with pytest.raises(RuntimeError, match="bad request"):
            backend._retry(func)

        assert func.call_count == 1

    def test_no_retry_needed_on_success(self) -> None:
        """Should return immediately on first success."""
        backend = self._make_backend()
        func = MagicMock(return_value="immediate success")

        result = backend._retry(func)

        assert result == "immediate success"
        assert func.call_count == 1

    @patch("deepseek_ocr.backends.base.time.sleep")
    @patch("deepseek_ocr.backends.base.random.uniform", return_value=0.25)
    def test_exponential_backoff_delays(self, mock_uniform, mock_sleep) -> None:
        """Should sleep with exponential backoff + jitter between retries."""
        backend = self._make_backend(max_retries=3, retry_delay=1.0)
        func = MagicMock(
            side_effect=[
                TransientError("fail 1"),
                TransientError("fail 2"),
                TransientError("fail 3"),
                "ok",
            ]
        )

        result = backend._retry(func)

        assert result == "ok"
        # Delays: 1.0*2^0 + 0.25 = 1.25, 1.0*2^1 + 0.25 = 2.25, 1.0*2^2 + 0.25 = 4.25
        assert mock_sleep.call_count == 3
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays == [1.25, 2.25, 4.25]


class TestOllamaRetry:
    """Tests for Ollama-specific transient error handling."""

    def _make_backend(self) -> OllamaBackend:
        with patch.object(OllamaBackend, "__init__", lambda self, **kw: None):
            backend = OllamaBackend.__new__(OllamaBackend)
        backend.model_name = "test"
        backend.max_dimension = None
        backend.max_retries = 2
        backend.retry_delay = 0.0
        backend.model = True
        backend.ollama_url = "http://localhost:11434"
        return backend

    def test_timeout_is_transient(self) -> None:
        """requests.Timeout should raise TransientError."""
        backend = self._make_backend()

        with patch("deepseek_ocr.backends.ollama.requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.Timeout("timed out")
            with pytest.raises(TransientError, match="timed out"):
                backend._call_ollama_api("base64img", "prompt")

    def test_connection_error_is_transient(self) -> None:
        """requests.ConnectionError should raise TransientError."""
        backend = self._make_backend()

        with patch("deepseek_ocr.backends.ollama.requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.ConnectionError("refused")
            with pytest.raises(TransientError, match="Lost connection"):
                backend._call_ollama_api("base64img", "prompt")

    @pytest.mark.parametrize("status_code", [429, 500, 502, 503, 504])
    def test_transient_http_status_is_transient(self, status_code: int) -> None:
        """HTTP 429/5xx should raise TransientError."""
        backend = self._make_backend()

        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.text = "server error"

        with patch("deepseek_ocr.backends.ollama.requests.post", return_value=mock_resp):
            with pytest.raises(TransientError, match=str(status_code)):
                backend._call_ollama_api("base64img", "prompt")

    @pytest.mark.parametrize("status_code", [400, 404, 422])
    def test_permanent_http_status_not_retried(self, status_code: int) -> None:
        """HTTP 4xx (not 429) should raise RuntimeError, not TransientError."""
        backend = self._make_backend()

        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.text = "client error"

        with patch("deepseek_ocr.backends.ollama.requests.post", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="Ollama API error"):
                backend._call_ollama_api("base64img", "prompt")
