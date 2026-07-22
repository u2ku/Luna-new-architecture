"""Gmail API client (stdlib urllib, no extra deps).

Tiny wrapper over the Gmail REST API for the email channel. Only
the endpoints the runtime needs:

* ``users.messages.list``   — find new messages (by query or
  history id)
* ``users.messages.get``    — fetch the raw RFC 822 of a message
* ``users.messages.send``   — POST a base64url-encoded raw message
  (used by the outbox flush; included here for symmetry)

Auth: Bearer token from :mod:`luna.channels.email.credentials`.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .credentials import GmailCredentials


GMAIL_API_BASE = "https://gmail.googleapis.com/gmail/v1/users/me"
DEFAULT_TIMEOUT = 30.0


class GmailAPIError(RuntimeError):
    """Raised on any non-2xx Gmail API response or transport error."""


def _b64url_decode(data: str) -> bytes:
    """Gmail uses base64url-without-padding; add it back."""
    pad = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + pad)


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


class GmailClient:
    """A thin Bearer-authenticated Gmail API client."""

    def __init__(
        self,
        credentials: GmailCredentials,
        *,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.credentials = credentials
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = GMAIL_API_BASE + path
        if params:
            url += "?" + urllib.parse.urlencode(
                {k: v for k, v in params.items() if v is not None}
            )
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.credentials.access_token()}",
                "Accept": "application/json",
                **({"Content-Type": "application/json"} if data is not None else {}),
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace")
            raise GmailAPIError(
                f"Gmail {method} {path} returned {e.code}: {body_text}"
            ) from e
        except (urllib.error.URLError, OSError) as e:
            raise GmailAPIError(f"Gmail {method} {path} failed: {e}") from e
        if not raw.strip():
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise GmailAPIError(f"Gmail {method} {path} returned non-JSON: {e}") from e

    def list_messages(
        self,
        *,
        query: str | None = None,
        label_ids: tuple[str, ...] | None = None,
        max_results: int = 20,
        page_token: str | None = None,
    ) -> tuple[list[str], str | None]:
        """List message IDs matching ``query`` (default: ``is:unread``).

        Returns ``(ids, next_page_token)``.
        """
        params: dict[str, Any] = {
            "q": query,
            "labelIds": ",".join(label_ids) if label_ids else None,
            "maxResults": max_results,
            "pageToken": page_token,
        }
        body = self._request("GET", "/messages", params=params)
        ids = [m["id"] for m in body.get("messages", []) if "id" in m]
        return ids, body.get("nextPageToken")

    def get_message_raw(self, message_id: str) -> bytes:
        """Fetch a single message as raw RFC 822 bytes.

        Uses ``format=raw`` so the bytes are ready for stdlib
        ``email.message_from_bytes``. The Gmail API returns the
        raw body as base64url in ``payload.body``.
        """
        body = self._request("GET", f"/messages/{message_id}", params={"format": "raw"})
        if "raw" not in body:
            raise GmailAPIError(
                f"Gmail message {message_id} response has no 'raw' field"
            )
        return _b64url_decode(body["raw"])

    def send_raw(self, raw_message: bytes) -> str:
        """Send a raw RFC 822 message. Returns the Gmail message id.

        Used by the outbox flush. Lives here so the API surface
        is in one place.
        """
        encoded = _b64url_encode(raw_message)
        body = self._request(
            "POST",
            "/messages/send",
            body={"raw": encoded},
        )
        return body["id"]
