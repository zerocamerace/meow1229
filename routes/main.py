from __future__ import annotations

import imghdr
import logging

import requests
from flask import Blueprint, Response, abort, render_template, request, session

from config.settings import CARD_LIMIT_MODAL_TEXT, REWARD_BADGES, DEFAULT_AVATAR
from services.firebase import db
from utils.security import _mask_uid, _safe_url

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def home():
    is_logged_in = "user_id" in session
    show_card_limit_modal = session.pop("show_card_limit_modal", False)
    card_limit_modal_text = None
    if show_card_limit_modal:
        card_limit_modal_text = session.pop(
            "card_limit_modal_text", CARD_LIMIT_MODAL_TEXT
        )
    else:
        session.pop("card_limit_modal_text", None)
    return render_template(
        "home.html",
        is_logged_in=is_logged_in,
        show_card_limit_modal=show_card_limit_modal,
        card_limit_modal_text=card_limit_modal_text,
    )


@main_bp.route("/featured_cats")
def featured_cats():
    is_logged_in = "user_id" in session
    return render_template("featured_cats.html", is_logged_in=is_logged_in)


@main_bp.route("/proxy_image")
def proxy_image():
    image_url = request.args.get("url", "")
    safe_url = _safe_url(image_url)
    if not safe_url:
        abort(400, description="Invalid image URL")

    try:
        upstream = requests.get(safe_url, timeout=6)
        upstream.raise_for_status()
    except requests.RequestException as exc:
        logging.warning("Image proxy fetch failed: %s", exc)
        abort(502, description="Image fetch failed")

    content_type = upstream.headers.get("Content-Type", "").lower()
    if not content_type.startswith("image"):
        detected = imghdr.what(None, upstream.content)
        if detected:
            content_type = f"image/{detected}"
        else:
            abort(415, description="Unsupported media type")

    response = Response(
        upstream.content, content_type=content_type or "image/png"
    )
    response.headers["Cache-Control"] = "public, max-age=86400"
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


def register_context_processors(app):
    @app.context_processor
    def inject_badge_context():
        user_id = session.get("user_id")
        user_points = session.get("points")
        user_avatar = session.get("avatar")

        if user_id and (user_points is None or user_avatar is None):
            try:
                snapshot = db.collection("users").document(user_id).get()
                data = snapshot.to_dict() or {}
                if user_points is None:
                    user_points = int(data.get("points") or 0)
                    session["points"] = user_points
                if user_avatar is None:
                    user_avatar = data.get("avatar") or DEFAULT_AVATAR
                    session["avatar"] = user_avatar
            except Exception as exc:
                logging.debug(
                    "Failed to refresh user info for %s: %s",
                    _mask_uid(user_id),
                    exc,
                )
                if user_points is None:
                    user_points = 0
                if user_avatar is None:
                    user_avatar = DEFAULT_AVATAR

        user_points = int(user_points or 0)
        user_avatar = user_avatar or DEFAULT_AVATAR
        unlocked_points = [
            badge["points"] for badge in REWARD_BADGES if user_points >= badge["points"]
        ]
        session["badge_thresholds"] = unlocked_points

        reward_unlocks = []
        if "reward_unlocks" in session:
            reward_unlocks = session.pop("reward_unlocks", [])

        enriched_badges = []
        for badge in REWARD_BADGES:
            enriched = dict(badge)
            enriched["unlocked"] = badge["points"] in unlocked_points
            enriched_badges.append(enriched)

        return {
            "is_logged_in": bool(user_id),
            "user_points": user_points,
            "reward_badges": enriched_badges,
            "reward_unlocks": reward_unlocks,
            "user_avatar": user_avatar,
            "debug_enabled": bool(app.debug),
        }
