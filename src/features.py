"""Text and metadata feature builders for tweet records."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import numpy as np

EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)
SOURCE_RE = re.compile(r">([^<]+)<")
URL_RE = re.compile(r"https?://\S+|t\.co/\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
TWITTER_TIME_FMT = "%a %b %d %H:%M:%S +0000 %Y"

METADATA_NAMES = [
    "retweet_count",
    "favorite_count",
    "reply_count",
    "quote_count",
    "is_quote_status",
    "truncated",
    "num_hashtags",
    "num_mentions",
    "num_urls",
    "num_media",
    "text_length",
    "emoji_count",
    "url_token_count",
    "mention_token_count",
    "hashtag_token_count",
    "user_statuses_count",
    "user_favourites_count",
    "user_listed_count",
    "user_default_profile",
    "user_geo_enabled",
    "user_has_url",
    "user_has_location",
    "user_has_banner",
    "is_reply",
    "retweeted",
    "favorited",
    "user_protected",
    "user_default_profile_image",
    "user_account_age_days",
]


def _safe_get(obj: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key, default)
        if cur is None:
            return default
    return cur


def _extract_source(source_html: str | None) -> str:
    if not source_html:
        return "unknown"
    match = SOURCE_RE.search(source_html)
    return match.group(1).strip() if match else source_html


def _tweet_text(record: dict[str, Any], prefix: str = "") -> str:
    extended = _safe_get(record, "extended_tweet", "full_text")
    if extended:
        return str(extended)
    text = record.get(f"{prefix}text") if prefix else record.get("text")
    return str(text or "")


def build_text(record: dict[str, Any]) -> str:
    """Build the transformer input text from a tweet record."""
    parts: list[str] = []

    main_text = _tweet_text(record)
    if main_text:
        parts.append(main_text)

    description = _safe_get(record, "user", "description")
    if description:
        parts.append(str(description))

    quoted = record.get("quoted_status")
    if isinstance(quoted, dict):
        quoted_text = _tweet_text(quoted)
        if quoted_text:
            parts.append(f"[QUOTED] {quoted_text}")

    entities = record.get("entities") or {}
    hashtags = entities.get("hashtags") or []
    if hashtags:
        tags = " ".join(f"#{h.get('text', '')}" for h in hashtags if h.get("text"))
        if tags:
            parts.append(f"[HASHTAGS] {tags}")

    mentions = entities.get("user_mentions") or []
    if mentions:
        handles = " ".join(f"@{m.get('screen_name', '')}" for m in mentions if m.get("screen_name"))
        if handles:
            parts.append(f"[MENTIONS] {handles}")

    lang = record.get("lang")
    if lang:
        parts.append(f"[LANG] {lang}")

    source = _extract_source(record.get("source"))
    parts.append(f"[SOURCE] {source}")

    return " ".join(parts).strip()


def _count_entities(record: dict[str, Any], key: str) -> float:
    entities = record.get("entities") or {}
    items = entities.get(key) or []
    return float(len(items))


def _count_media(record: dict[str, Any]) -> float:
    extended = record.get("extended_tweet") or {}
    media = (extended.get("entities") or {}).get("media") or []
    if media:
        return float(len(media))
    media = (record.get("entities") or {}).get("media") or []
    return float(len(media))


def _bool_to_float(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def _parse_twitter_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, TWITTER_TIME_FMT)
    except ValueError:
        return None


def _is_reply(record: dict[str, Any]) -> bool:
    return any(
        record.get(key) is not None
        for key in (
            "in_reply_to_status_id",
            "in_reply_to_status_id_str",
            "in_reply_to_user_id",
            "in_reply_to_user_id_str",
            "in_reply_to_screen_name",
        )
    )


def _account_age_days(record: dict[str, Any], user: dict[str, Any]) -> float:
    tweet_time = _parse_twitter_time(record.get("created_at"))
    user_time = _parse_twitter_time(user.get("created_at"))
    if tweet_time is None or user_time is None:
        return 0.0
    return max((tweet_time - user_time).total_seconds() / 86400.0, 0.0)


def extract_metadata(record: dict[str, Any]) -> np.ndarray:
    """Extract numeric metadata vector from a tweet record."""
    text = build_text(record)
    user = record.get("user") or {}

    values = [
        float(record.get("retweet_count") or 0),
        float(record.get("favorite_count") or 0),
        float(record.get("reply_count") or 0),
        float(record.get("quote_count") or 0),
        _bool_to_float(record.get("is_quote_status")),
        _bool_to_float(record.get("truncated")),
        _count_entities(record, "hashtags"),
        _count_entities(record, "user_mentions"),
        _count_entities(record, "urls"),
        _count_media(record),
        float(len(text)),
        float(len(EMOJI_RE.findall(text))),
        float(len(URL_RE.findall(text))),
        float(len(MENTION_RE.findall(text))),
        float(len(HASHTAG_RE.findall(text))),
        float(user.get("statuses_count") or 0),
        float(user.get("favourites_count") or 0),
        float(user.get("listed_count") or 0),
        _bool_to_float(user.get("default_profile")),
        _bool_to_float(user.get("geo_enabled")),
        _bool_to_float(user.get("url")),
        _bool_to_float(user.get("location")),
        _bool_to_float(user.get("profile_banner_url")),
        _bool_to_float(_is_reply(record)),
        _bool_to_float(record.get("retweeted")),
        _bool_to_float(record.get("favorited")),
        _bool_to_float(user.get("protected")),
        _bool_to_float(user.get("default_profile_image")),
        _account_age_days(record, user),
    ]
    return np.asarray(values, dtype=np.float32)


def compute_metadata_stats(metadata: np.ndarray) -> dict[str, np.ndarray]:
    mean = metadata.mean(axis=0)
    std = metadata.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def normalize_metadata(metadata: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    return ((metadata - stats["mean"]) / stats["std"]).astype(np.float32)
