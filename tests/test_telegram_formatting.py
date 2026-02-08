"""
Test: Telegram Markdown formatting safety.

Verifies that escape_markdown and escape_markdown_v2 correctly handle
dynamic content containing special characters that would break Telegram
message parsing (URLs, SHAs, file paths, braces, underscores).
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from telegram_bot_v2.bot import escape_markdown, escape_markdown_v2


# ---------------------------------------------------------------------------
# escape_markdown (Markdown v1)
# ---------------------------------------------------------------------------

class TestEscapeMarkdown:
    """escape_markdown escapes all Markdown v1 special characters."""

    def test_escapes_underscores(self):
        """Underscores in failure types like api_unreachable are escaped."""
        assert "\\_" in escape_markdown("api_unreachable")

    def test_escapes_asterisks(self):
        assert "\\*" in escape_markdown("bold*text")

    def test_escapes_backticks(self):
        assert "\\`" in escape_markdown("code`block")

    def test_escapes_brackets(self):
        result = escape_markdown("[link](url)")
        assert "\\[" in result
        assert "\\]" in result
        assert "\\(" in result
        assert "\\)" in result

    def test_url_with_underscores(self):
        """URL like https://health_api.example.com is safely escaped."""
        url = "https://health_api.example.com/v2"
        result = escape_markdown(url)
        assert "\\_" in result
        assert "\\(" not in result or result.count("\\(") == 0  # No parens in this URL

    def test_empty_string(self):
        assert escape_markdown("") == ""

    def test_none_returns_none(self):
        assert escape_markdown(None) is None

    def test_git_sha(self):
        """Git SHAs are safe (hex only)."""
        sha = "d33aca4f1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e"
        assert escape_markdown(sha) == sha

    def test_file_path(self):
        """File paths with underscores and dots are escaped."""
        path = "/home/health_api/public_html/app/main.py"
        result = escape_markdown(path)
        assert "\\_" in result

    def test_complex_failure_message(self):
        """A realistic failure message with mixed special chars is fully escaped."""
        msg = "HTTP 503: api_health check failed at https://api.example.com/health (timeout)"
        result = escape_markdown(msg)
        # Should not raise when parsed — all special chars escaped
        assert "\\_" in result
        assert "\\(" in result
        assert "\\)" in result

    def test_numeric_input_converted(self):
        """Non-string input is converted to string."""
        assert escape_markdown(503) == "503"


# ---------------------------------------------------------------------------
# escape_markdown_v2 (MarkdownV2)
# ---------------------------------------------------------------------------

class TestEscapeMarkdownV2:
    """escape_markdown_v2 escapes all MarkdownV2 reserved characters."""

    def test_escapes_dots(self):
        assert "\\." in escape_markdown_v2("file.txt")

    def test_escapes_dashes(self):
        assert "\\-" in escape_markdown_v2("micro-remediation")

    def test_escapes_exclamation(self):
        assert "\\!" in escape_markdown_v2("Failed!")

    def test_escapes_braces(self):
        result = escape_markdown_v2("{key: value}")
        assert "\\{" in result
        assert "\\}" in result

    def test_escapes_pipe(self):
        assert "\\|" in result if (result := escape_markdown_v2("a|b")) else False

    def test_escapes_hash(self):
        assert "\\#" in escape_markdown_v2("#channel")

    def test_escapes_plus(self):
        assert "\\+" in escape_markdown_v2("c++")

    def test_escapes_equals(self):
        assert "\\=" in escape_markdown_v2("x=1")

    def test_escapes_tilde(self):
        assert "\\~" in escape_markdown_v2("~approx")

    def test_escapes_gt(self):
        assert "\\>" in escape_markdown_v2("> quote")

    def test_full_url(self):
        """URL with all sorts of special chars is fully escaped."""
        url = "https://healthapi.gahfaudio.in/health"
        result = escape_markdown_v2(url)
        assert "\\." in result
        assert "\\/" in result or "/" in result  # / is not reserved

    def test_empty_string(self):
        assert escape_markdown_v2("") == ""

    def test_none_returns_none(self):
        assert escape_markdown_v2(None) is None


# ---------------------------------------------------------------------------
# safe_reply_text and safe_edit_text (fallback behavior)
# ---------------------------------------------------------------------------

class TestSafeReplyText:
    """safe_reply_text falls back to plain text on parse error."""

    @pytest.mark.asyncio
    async def test_fallback_on_parse_error(self):
        from telegram_bot_v2.bot import safe_reply_text

        mock_message = AsyncMock()
        # First call raises parse error, second succeeds
        mock_message.reply_text = AsyncMock(
            side_effect=[
                Exception("Can't parse entities: can't find end of the entity"),
                MagicMock(),  # plain text succeeds
            ]
        )

        await safe_reply_text(mock_message, "*broken_text*", parse_mode="Markdown")

        # Should have been called twice: once with Markdown, once without
        assert mock_message.reply_text.call_count == 2
        second_call_kwargs = mock_message.reply_text.call_args_list[1]
        # Second call should NOT have parse_mode
        assert "parse_mode" not in second_call_kwargs.kwargs or second_call_kwargs.kwargs.get("parse_mode") is None

    @pytest.mark.asyncio
    async def test_success_no_fallback(self):
        from telegram_bot_v2.bot import safe_reply_text

        mock_message = AsyncMock()
        mock_message.reply_text = AsyncMock(return_value=MagicMock())

        await safe_reply_text(mock_message, "*valid bold*", parse_mode="Markdown")

        assert mock_message.reply_text.call_count == 1


class TestSafeEditText:
    """safe_edit_text falls back to plain text on parse error."""

    @pytest.mark.asyncio
    async def test_fallback_on_parse_error(self):
        from telegram_bot_v2.bot import safe_edit_text

        mock_query = AsyncMock()
        mock_query.edit_message_text = AsyncMock(
            side_effect=[
                Exception("Can't parse entities: can't find end of the entity"),
                MagicMock(),
            ]
        )

        await safe_edit_text(mock_query, "*broken_text*", parse_mode="Markdown")

        assert mock_query.edit_message_text.call_count == 2
