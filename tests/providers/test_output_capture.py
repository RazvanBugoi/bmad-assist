"""Tests for stdout/stderr capture functionality in ClaudeSubprocessProvider.

Tests AC1-AC9 from Story 4.5: Stdout/Stderr Capture.
All tests mock Popen - no real subprocess calls.

Note: These tests specifically test the subprocess-based provider's output
capture behavior. The SDK provider (ClaudeSDKProvider) handles output
differently via typed message objects.

Test Structure:
    - TestOutputCaptureSeparation: AC1, AC5, AC6 (stdout/stderr separation)
    - TestOutputCaptureEncoding: AC2, AC8 (UTF-8 handling)
    - TestOutputCaptureLarge: AC3 (large output)
    - TestOutputCaptureEmpty: AC7 (empty output)
    - TestOutputCaptureConfig: AC9 (configuration)

"""

from unittest.mock import patch

import pytest

from bmad_assist.providers import ClaudeSubprocessProvider
from bmad_assist.providers.base import ProviderResult

from .conftest import create_mock_process


@pytest.fixture
def claude_provider() -> ClaudeSubprocessProvider:
    """Create ClaudeSubprocessProvider instance for testing."""
    return ClaudeSubprocessProvider()


class TestOutputCaptureSeparation:
    """Test AC1, AC5, AC6: stdout and stderr captured separately."""

    def test_separate_stdout_stderr_capture(
        self, claude_provider: ClaudeSubprocessProvider
    ) -> None:
        """Test AC1: stdout and stderr captured separately as strings."""
        with patch("bmad_assist.providers.claude.Popen") as mock_popen:
            mock_popen.return_value = create_mock_process(
                response_text="Main response here",
                stderr_content="Warning: something minor\n",
            )

            result = claude_provider.invoke("Test prompt", timeout=5)

            assert result.stdout == "Main response here"
            assert "Warning: something minor" in result.stderr
            assert isinstance(result.stdout, str)
            assert isinstance(result.stderr, str)

    def test_stdout_not_mixed_with_stderr(self, claude_provider: ClaudeSubprocessProvider) -> None:
        """Test AC1: stdout content separate from stderr content."""
        with patch("bmad_assist.providers.claude.Popen") as mock_popen:
            mock_popen.return_value = create_mock_process(
                response_text="STDOUT_CONTENT",
                stderr_content="STDERR_CONTENT\n",
            )

            result = claude_provider.invoke("Test", timeout=5)

            assert "STDERR_CONTENT" not in result.stdout
            assert "STDOUT_CONTENT" not in result.stderr

    def test_stdout_only_response(self, claude_provider: ClaudeSubprocessProvider) -> None:
        """Test AC5: main LLM response extracted from stdout."""
        with patch("bmad_assist.providers.claude.Popen") as mock_popen:
            mock_popen.return_value = create_mock_process(
                response_text="Hello world",
                stderr_content="",
            )

            result = claude_provider.invoke("Hello", timeout=5)
            response = claude_provider.parse_output(result)

            assert response == "Hello world"

    def test_stderr_contains_warnings(self, claude_provider: ClaudeSubprocessProvider) -> None:
        """Test AC6: stderr contains warnings separate from response."""
        with patch("bmad_assist.providers.claude.Popen") as mock_popen:
            mock_popen.return_value = create_mock_process(
                response_text="Response text",
                stderr_content="Warning: deprecation notice\nWarning: rate limit approaching\n",
            )

            result = claude_provider.invoke("Test", timeout=5)

            assert "Warning" in result.stderr
            assert "deprecation" in result.stderr
            assert "Warning" not in result.stdout

    def test_both_response_and_errors_available(
        self, claude_provider: ClaudeSubprocessProvider
    ) -> None:
        """Test AC6: both stdout response and stderr errors available."""
        with patch("bmad_assist.providers.claude.Popen") as mock_popen:
            mock_popen.return_value = create_mock_process(
                response_text="The answer is 42",
                stderr_content="Error: context limit warning\n",
            )

            result = claude_provider.invoke("Test", timeout=5)

            assert result.stdout == "The answer is 42"
            assert "context limit warning" in result.stderr

    def test_output_not_truncated(self, claude_provider: ClaudeSubprocessProvider) -> None:
        """Test AC1: output is complete (not truncated)."""
        # 10KB of content
        content = "x" * 10240
        with patch("bmad_assist.providers.claude.Popen") as mock_popen:
            mock_popen.return_value = create_mock_process(
                response_text=content,
                stderr_content="",
            )

            result = claude_provider.invoke("Test", timeout=5)

            assert len(result.stdout) == 10240
            assert result.stdout == content


class TestOutputCaptureEncoding:
    """Test AC2, AC8: UTF-8 encoding handling."""

    def test_utf8_japanese_characters(self, claude_provider: ClaudeSubprocessProvider) -> None:
        """Test AC2: Japanese UTF-8 characters preserved."""
        with patch("bmad_assist.providers.claude.Popen") as mock_popen:
            mock_popen.return_value = create_mock_process(
                response_text="日本語テスト",
            )

            result = claude_provider.invoke("Test", timeout=5)

            assert "日本語" in result.stdout
            assert result.stdout == "日本語テスト"

    def test_utf8_polish_characters(self, claude_provider: ClaudeSubprocessProvider) -> None:
        """Test AC2: Polish UTF-8 characters preserved."""
        with patch("bmad_assist.providers.claude.Popen") as mock_popen:
            mock_popen.return_value = create_mock_process(
                response_text="Cześć, świat! Zażółć gęślą jaźń",
            )

            result = claude_provider.invoke("Test", timeout=5)

            assert "Cześć" in result.stdout
            assert "ż" in result.stdout
            assert "ś" in result.stdout

    def test_utf8_emoji(self, claude_provider: ClaudeSubprocessProvider) -> None:
        """Test AC2: Emoji UTF-8 characters preserved."""
        with patch("bmad_assist.providers.claude.Popen") as mock_popen:
            mock_popen.return_value = create_mock_process(
                response_text="🚀 rocket launch 🎉 celebration 🔥 fire",
            )

            result = claude_provider.invoke("Test", timeout=5)

            assert "🚀" in result.stdout
            assert "🎉" in result.stdout
            assert "🔥" in result.stdout

    def test_utf8_international_mixed(self, claude_provider: ClaudeSubprocessProvider) -> None:
        """Test AC2: Mixed international UTF-8 characters preserved."""
        international = "日本語テスト Cześć 🚀 émoji café"
        with patch("bmad_assist.providers.claude.Popen") as mock_popen:
            mock_popen.return_value = create_mock_process(
                response_text=international,
            )

            result = claude_provider.invoke("Test", timeout=5)

            assert result.stdout == international
            assert "日本語" in result.stdout
            assert "Cześć" in result.stdout
            assert "🚀" in result.stdout
            assert "émoji" in result.stdout
            assert "café" in result.stdout

    def test_replacement_character_preserved(
        self, claude_provider: ClaudeSubprocessProvider
    ) -> None:
        """Test AC2, AC8: Replacement character (U+FFFD) preserved in output.

        When Popen with errors='replace' encounters invalid UTF-8,
        it substitutes with U+FFFD. This test verifies the replacement is
        preserved in ProviderResult.
        """
        # Simulate Popen returning string with replacement character
        # (Popen already did the decoding with errors='replace')
        output_with_replacement = "valid \ufffd invalid \ufffd data"
        with patch("bmad_assist.providers.claude.Popen") as mock_popen:
            mock_popen.return_value = create_mock_process(
                response_text=output_with_replacement,
            )

            result = claude_provider.invoke("Test", timeout=5)

            assert result.stdout == output_with_replacement
            assert "\ufffd" in result.stdout
            # Count replacement characters
            assert result.stdout.count("\ufffd") == 2

    def test_no_unicode_decode_error(self, claude_provider: ClaudeSubprocessProvider) -> None:
        """Test AC2, AC8: No UnicodeDecodeError raised."""
        # Simulate Popen handling invalid UTF-8 gracefully
        with patch("bmad_assist.providers.claude.Popen") as mock_popen:
            mock_popen.return_value = create_mock_process(
                response_text="partially valid \ufffd content",
            )

            # Should not raise any exception
            result = claude_provider.invoke("Test", timeout=5)

            assert isinstance(result.stdout, str)

    def test_replacement_character_in_stderr(
        self, claude_provider: ClaudeSubprocessProvider
    ) -> None:
        """Test AC2, AC8: Replacement character in stderr preserved."""
        stderr_with_replacement = "warning: encoding \ufffd issue detected\n"
        with patch("bmad_assist.providers.claude.Popen") as mock_popen:
            mock_popen.return_value = create_mock_process(
                response_text="Response text",
                stderr_content=stderr_with_replacement,
            )

            result = claude_provider.invoke("Test", timeout=5)

            assert "encoding \ufffd issue" in result.stderr
            assert "\ufffd" in result.stderr

    def test_replacement_character_in_both_streams(
        self, claude_provider: ClaudeSubprocessProvider
    ) -> None:
        """Test AC2, AC8: Replacement character in both stdout and stderr."""
        with patch("bmad_assist.providers.claude.Popen") as mock_popen:
            mock_popen.return_value = create_mock_process(
                response_text="response \ufffd data",
                stderr_content="warning \ufffd issue\n",
            )

            result = claude_provider.invoke("Test", timeout=5)

            assert "\ufffd" in result.stdout
            assert "\ufffd" in result.stderr


class TestOutputCaptureLarge:
    """Test AC3: Large output capture without truncation."""

    def test_1mb_output_captured_completely(
        self, claude_provider: ClaudeSubprocessProvider
    ) -> None:
        """Test AC3: 1MB output captured without truncation."""
        # Generate exactly 1MB of content
        large_content = "x" * (1024 * 1024)  # 1MB
        with patch("bmad_assist.providers.claude.Popen") as mock_popen:
            mock_popen.return_value = create_mock_process(
                response_text=large_content,
            )

            result = claude_provider.invoke("Generate large output", timeout=60)

            assert len(result.stdout) == 1024 * 1024
            assert result.stdout == large_content

    def test_large_output_content_integrity(
        self, claude_provider: ClaudeSubprocessProvider
    ) -> None:
        """Test AC3: Large output maintains content integrity."""
        # Pattern that would show if truncation happened
        content = "".join(f"LINE_{i:06d}\n" for i in range(50000))
        with patch("bmad_assist.providers.claude.Popen") as mock_popen:
            mock_popen.return_value = create_mock_process(
                response_text=content,
            )

            result = claude_provider.invoke("Test", timeout=60)

            # Verify first and last lines intact
            assert "LINE_000000" in result.stdout
            assert "LINE_049999" in result.stdout
            assert result.stdout.count("\n") == 50000

    def test_10mb_output_captured(self, claude_provider: ClaudeSubprocessProvider) -> None:
        """Test AC3: 10MB output captured without truncation."""
        # 10MB - edge of typical use
        large_content = "y" * (10 * 1024 * 1024)  # 10MB
        with patch("bmad_assist.providers.claude.Popen") as mock_popen:
            mock_popen.return_value = create_mock_process(
                response_text=large_content,
            )

            result = claude_provider.invoke("Generate large output", timeout=120)

            assert len(result.stdout) == 10 * 1024 * 1024

    def test_large_stderr_captured(self, claude_provider: ClaudeSubprocessProvider) -> None:
        """Test AC3: Large stderr also captured completely."""
        large_stderr = "warning: " * 100000  # ~900KB
        with patch("bmad_assist.providers.claude.Popen") as mock_popen:
            mock_popen.return_value = create_mock_process(
                response_text="Response",
                stderr_content=large_stderr + "\n",
            )

            result = claude_provider.invoke("Test", timeout=60)

            assert large_stderr in result.stderr


class TestOutputCaptureEmpty:
    """Test AC7: Empty output handling."""

    def test_empty_stdout_returns_empty_string(
        self, claude_provider: ClaudeSubprocessProvider
    ) -> None:
        """Test AC7: Empty stdout is empty string, not None."""
        with patch("bmad_assist.providers.claude.Popen") as mock_popen:
            mock_popen.return_value = create_mock_process(
                stdout_content="",
                stderr_content="warning: no output generated\n",
            )

            result = claude_provider.invoke("Test", timeout=5)

            assert result.stdout == ""
            assert result.stdout is not None
            assert isinstance(result.stdout, str)

    def test_empty_stderr_returns_empty_string(
        self, claude_provider: ClaudeSubprocessProvider
    ) -> None:
        """Test AC7: Empty stderr is empty string, not None."""
        with patch("bmad_assist.providers.claude.Popen") as mock_popen:
            mock_popen.return_value = create_mock_process(
                response_text="response",
                stderr_content="",
            )

            result = claude_provider.invoke("Test", timeout=5)

            assert result.stderr == ""
            assert result.stderr is not None
            assert isinstance(result.stderr, str)

    def test_both_empty(self, claude_provider: ClaudeSubprocessProvider) -> None:
        """Test AC7: Both stdout and stderr can be empty strings."""
        with patch("bmad_assist.providers.claude.Popen") as mock_popen:
            mock_popen.return_value = create_mock_process(
                stdout_content="",
                stderr_content="",
            )

            result = claude_provider.invoke("Test", timeout=5)

            assert result.stdout == ""
            assert result.stderr == ""

    def test_parse_output_empty_stdout(self, claude_provider: ClaudeSubprocessProvider) -> None:
        """Test AC7: parse_output returns empty string for empty stdout."""
        with patch("bmad_assist.providers.claude.Popen") as mock_popen:
            mock_popen.return_value = create_mock_process(
                stdout_content="",
                stderr_content="warning message\n",
            )

            result = claude_provider.invoke("Test", timeout=5)
            response = claude_provider.parse_output(result)

            assert response == ""

    def test_stdout_only_whitespace(self, claude_provider: ClaudeSubprocessProvider) -> None:
        """Test AC7: Whitespace-only stdout stripped to empty by parse_output."""
        with patch("bmad_assist.providers.claude.Popen") as mock_popen:
            mock_popen.return_value = create_mock_process(
                response_text="   \n\t\n   ",
            )

            result = claude_provider.invoke("Test", timeout=5)
            response = claude_provider.parse_output(result)

            assert response == ""


class TestOutputCaptureConfig:
    """Test AC9: Capture configuration is explicit."""

    def test_popen_uses_text_mode(self, claude_provider: ClaudeSubprocessProvider) -> None:
        """Test AC9: text=True is always set in Popen call."""
        with patch("bmad_assist.providers.claude.Popen") as mock_popen:
            mock_popen.return_value = create_mock_process(
                response_text="Response",
            )

            claude_provider.invoke("Test", timeout=5)

            call_kwargs = mock_popen.call_args.kwargs
            assert call_kwargs.get("text") is True

    def test_popen_uses_utf8_encoding(self, claude_provider: ClaudeSubprocessProvider) -> None:
        """Test AC9: encoding='utf-8' is explicitly set in Popen call."""
        with patch("bmad_assist.providers.claude.Popen") as mock_popen:
            mock_popen.return_value = create_mock_process(
                response_text="Response",
            )

            claude_provider.invoke("Test", timeout=5)

            call_kwargs = mock_popen.call_args.kwargs
            assert call_kwargs.get("encoding") == "utf-8"

    def test_popen_uses_errors_replace(self, claude_provider: ClaudeSubprocessProvider) -> None:
        """Test AC9: errors='replace' is explicitly set in Popen call."""
        with patch("bmad_assist.providers.claude.Popen") as mock_popen:
            mock_popen.return_value = create_mock_process(
                response_text="Response",
            )

            claude_provider.invoke("Test", timeout=5)

            call_kwargs = mock_popen.call_args.kwargs
            assert call_kwargs.get("errors") == "replace"

    def test_popen_uses_pipe_for_stdout_stderr(
        self, claude_provider: ClaudeSubprocessProvider
    ) -> None:
        """Test AC9: PIPE is used for stdout and stderr."""
        from subprocess import PIPE

        with patch("bmad_assist.providers.claude.Popen") as mock_popen:
            mock_popen.return_value = create_mock_process(
                response_text="Response",
            )

            claude_provider.invoke("Test", timeout=5)

            call_kwargs = mock_popen.call_args.kwargs
            assert call_kwargs.get("stdout") == PIPE
            assert call_kwargs.get("stderr") == PIPE

    def test_all_encoding_params_together(self, claude_provider: ClaudeSubprocessProvider) -> None:
        """Test AC9: All encoding parameters set together in Popen call."""
        from subprocess import PIPE

        with patch("bmad_assist.providers.claude.Popen") as mock_popen:
            mock_popen.return_value = create_mock_process(
                response_text="Response",
            )

            claude_provider.invoke("Test", timeout=5)

            call_kwargs = mock_popen.call_args.kwargs
            # Verify all four parameters are set correctly
            assert call_kwargs.get("stdout") == PIPE
            assert call_kwargs.get("stderr") == PIPE
            assert call_kwargs.get("text") is True
            assert call_kwargs.get("encoding") == "utf-8"
            assert call_kwargs.get("errors") == "replace"


class TestTimeoutEncodingHandling:
    """Test encoding handling during timeout scenarios."""

    def test_timeout_with_partial_output(
        self, claude_provider: ClaudeSubprocessProvider, accelerated_time
    ) -> None:
        """Test timeout partial output with streamed content."""
        from bmad_assist.core.exceptions import ProviderTimeoutError

        with patch("bmad_assist.providers.claude.Popen") as mock_popen:
            # Create mock with partial stream output before timeout
            mock_popen.return_value = create_mock_process(
                response_text="partial output before timeout",
                stderr_content="error output\n",
                never_finish=True,
            )

            with pytest.raises(ProviderTimeoutError) as exc_info:
                claude_provider.invoke("Test", timeout=5)

            # Verify partial result captured
            partial = exc_info.value.partial_result
            assert partial is not None
            assert isinstance(partial.stdout, str)
            assert isinstance(partial.stderr, str)
            assert "partial output" in partial.stdout
            assert "error output" in partial.stderr

    def test_timeout_with_str_partial_output(
        self, claude_provider: ClaudeSubprocessProvider, accelerated_time
    ) -> None:
        """Test timeout partial output handles str correctly."""
        from bmad_assist.core.exceptions import ProviderTimeoutError

        with patch("bmad_assist.providers.claude.Popen") as mock_popen:
            mock_popen.return_value = create_mock_process(
                response_text="partial string output",
                stderr_content="error string\n",
                never_finish=True,
            )

            with pytest.raises(ProviderTimeoutError) as exc_info:
                claude_provider.invoke("Test", timeout=5)

            partial = exc_info.value.partial_result
            assert partial is not None
            assert "partial string output" in partial.stdout
            assert "error string" in partial.stderr


class TestBackwardCompatibility:
    """Test backward compatibility with existing provider tests."""

    def test_provider_result_fields_unchanged(
        self, claude_provider: ClaudeSubprocessProvider
    ) -> None:
        """Test backward compatibility: ProviderResult fields unchanged."""
        with patch("bmad_assist.providers.claude.Popen") as mock_popen:
            mock_popen.return_value = create_mock_process(
                response_text="Response",
                stderr_content="Warning\n",
            )

            result = claude_provider.invoke("Test", timeout=5)

            # Verify all expected fields exist and have correct types
            assert hasattr(result, "stdout")
            assert hasattr(result, "stderr")
            assert hasattr(result, "exit_code")
            assert hasattr(result, "duration_ms")
            assert hasattr(result, "model")
            assert hasattr(result, "command")

            assert isinstance(result.stdout, str)
            assert isinstance(result.stderr, str)
            assert isinstance(result.exit_code, int)
            assert isinstance(result.duration_ms, int)
            assert isinstance(result.model, str)
            assert isinstance(result.command, tuple)

    def test_provider_result_immutable(self) -> None:
        """Test backward compatibility: ProviderResult is frozen dataclass."""
        result = ProviderResult(
            stdout="test",
            stderr="",
            exit_code=0,
            duration_ms=100,
            model="sonnet",
            command=("claude", "-p", "test"),
        )

        # Frozen dataclass should raise on modification
        with pytest.raises(AttributeError):
            result.stdout = "modified"  # type: ignore[misc]
