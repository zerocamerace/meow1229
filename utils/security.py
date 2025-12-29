from __future__ import annotations

import ipaddress
import re
import socket
from typing import Any
from urllib.parse import urlparse

_DISALLOWED_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _is_public_host(hostname: str) -> bool:
    def _ip_allowed(ip_obj: ipaddress._BaseAddress) -> bool:
        return not (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_multicast
            or ip_obj.is_link_local
            or ip_obj.is_reserved
        )

    try:
        ip_obj = ipaddress.ip_address(hostname)
        return _ip_allowed(ip_obj)
    except ValueError:
        try:
            results = socket.getaddrinfo(hostname, None)
        except socket.gaierror:
            return False
        for result in results:
            ip_str = result[4][0]
            try:
                ip_obj = ipaddress.ip_address(ip_str)
            except ValueError:
                return False
            if not _ip_allowed(ip_obj):
                return False
        return True


def _safe_url(url: str | None) -> str | None:
    if not url:
        return None
    try:
        parsed = urlparse(url)
    except ValueError:
        return None

    path = parsed.path or ""
    if parsed.scheme not in {"http", "https"}:
        if parsed.scheme == "" and parsed.netloc == "" and path.startswith("/static/"):
            return path
        return None

    host = parsed.hostname
    if not host:
        return None

    if host.lower() in _DISALLOWED_HOSTS:
        if path.startswith("/static/"):
            return path
        return None

    if not _is_public_host(host):
        return None

    return parsed.geturl()


def _tokenize_text(text: str) -> list[str]:
    if not text:
        return []
    lowered = str(text).lower()
    return [token for token in re.split(r"[\s,、/，。!?！？」「]+", lowered) if token]


def _mask_email(email: str | None) -> str:
    if not email:
        return ""
    email = str(email)
    if "@" not in email:
        return (email[:1] or "") + "***"
    local_part, domain_part = email.split("@", 1)
    if len(local_part) <= 1:
        masked_local = (local_part[:1] or "") + "***"
    elif len(local_part) == 2:
        masked_local = local_part[0] + "***"
    else:
        masked_local = f"{local_part[0]}***{local_part[-1]}"
    return f"{masked_local}@{domain_part}"


def _mask_uid(uid: str | None) -> str:
    if not uid:
        return ""
    uid = str(uid)
    if len(uid) <= 3:
        return uid[0] + "***" if uid else ""
    return f"{uid[:2]}***{uid[-2:]}"


def _form_keys(form_data: Any) -> list[str]:
    try:
        return list(form_data.keys())
    except AttributeError:
        return []


def _is_text_within_limit(text: str | None, limit: int) -> bool:
    if text is None:
        return True
    return len(text) <= limit


def _user_messages_within_limit(conversation: Any, limit: int) -> bool:
    if not isinstance(conversation, list):
        return False
    for entry in conversation:
        if not isinstance(entry, dict):
            continue
        if entry.get("role") != "user":
            continue
        parts = entry.get("parts") or []
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if text and len(text) > limit:
                return False
    return True
