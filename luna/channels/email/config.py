"""SMTP config for the email channel.

Credentials are NEVER in source. Read them from env at runtime via
:func:`SmtpConfig.from_env`. The placeholders in ``.env.example``
list the variable names; the actual values go in ``.env`` (which
is gitignored).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


SUPPORTED_SECURITY = ("starttls", "tls", "none")


@dataclass(frozen=True)
class SmtpConfig:
    """SMTP connection config for outbound email.

    Attributes
    ----------
    host:
        SMTP server hostname, e.g. ``smtp.gmail.com``.
    port:
        SMTP submission port. ``587`` for STARTTLS, ``465`` for
        implicit TLS, ``25`` for plaintext (LAN / dev only).
    security:
        ``"starttls"`` upgrades a plaintext connection with
        ``STARTTLS`` after ``EHLO``. ``"tls"`` uses implicit TLS
        (the connection is TLS from the first byte). ``"none"``
        skips encryption (do not use on the public internet).
    username, password:
        SMTP auth credentials. May be ``None`` for open relays /
        LAN test servers.
    """

    host: str
    port: int
    security: str
    username: str | None = None
    password: str | None = None

    def __post_init__(self) -> None:
        if self.security not in SUPPORTED_SECURITY:
            raise ValueError(
                f"smtp.security must be one of {SUPPORTED_SECURITY}, "
                f"got {self.security!r}"
            )
        if self.port <= 0 or self.port > 65535:
            raise ValueError(f"smtp.port out of range: {self.port}")

    @classmethod
    def from_env(cls) -> "SmtpConfig":
        """Build from LUNA_EMAIL_SMTP_* environment variables.

        Returns a config with empty host if no env vars are set; the
        sender will treat that as "no SMTP configured" and queue
        outbound mail instead of sending.
        """
        return cls(
            host=os.environ.get("LUNA_EMAIL_SMTP_HOST", "").strip(),
            port=int(os.environ.get("LUNA_EMAIL_SMTP_PORT", "587")),
            security=os.environ.get(
                "LUNA_EMAIL_SMTP_SECURITY", "starttls"
            ).strip().lower(),
            username=os.environ.get("LUNA_EMAIL_SMTP_USERNAME") or None,
            password=os.environ.get("LUNA_EMAIL_SMTP_PASSWORD") or None,
        )

    def is_configured(self) -> bool:
        """True if this config has enough info to attempt delivery."""
        return bool(self.host) and self.port > 0


def default_from_address() -> str:
    """Return the configured From address, or a sane default."""
    return os.environ.get(
        "LUNA_EMAIL_FROM_ADDRESS", "luna@resonantconstructs.ai"
    ).strip()
