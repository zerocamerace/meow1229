from __future__ import annotations

import logging

from flask import Blueprint, flash, redirect, render_template, request, session, url_for

from config.settings import AVATAR_CHOICES, DEFAULT_AVATAR, REWARD_BADGES
from services.firebase import db

profile_bp = Blueprint("profile", __name__)


@profile_bp.route("/profile", methods=["GET", "POST"])
@profile_bp.route("/badges", methods=["GET", "POST"])
def profile():
    user_id = session.get("user_id")
    if not user_id:
        flash("請先登入後再查看我的檔案頁面。", "error")
        return redirect(url_for("auth.login"))

    current_avatar = session.get("avatar") or DEFAULT_AVATAR
    user_email = session.get("user_email")

    if request.method == "POST":
        selected_avatar = request.form.get("avatar")
        if selected_avatar not in AVATAR_CHOICES:
            flash("選擇的頭像無效，請重新選擇。", "error")
        else:
            try:
                db.collection("users").document(user_id).set(
                    {"avatar": selected_avatar}, merge=True
                )
                session["avatar"] = selected_avatar
                current_avatar = selected_avatar
                flash("頭像已更新！", "success")
            except Exception as exc:
                logging.error("Failed to update avatar for %s: %s", user_id, exc)
                flash("頭像更新失敗，請稍後再試。", "error")
        return redirect(url_for("profile.profile"))

    user_points = session.get("points")
    need_user_doc = user_points is None or not user_email
    user_data = {}
    if need_user_doc:
        try:
            snapshot = db.collection("users").document(user_id).get()
            user_data = snapshot.to_dict() or {}
        except Exception as exc:
            logging.debug("Failed to refresh user doc in profile page: %s", exc)

    if user_points is None:
        user_points = int(user_data.get("points") or 0)
        session["points"] = user_points
        current_avatar = user_data.get("avatar") or current_avatar
        session["avatar"] = current_avatar
    if not user_email:
        user_email = user_data.get("email") or ""
        if user_email:
            session["user_email"] = user_email

    user_points = int(user_points or 0)
    max_points = REWARD_BADGES[-1]["points"] if REWARD_BADGES else 0
    progress_percent = 0
    if max_points > 0:
        progress_percent = min(user_points, max_points) / max_points * 100

    next_badge = next(
        (badge for badge in REWARD_BADGES if badge["points"] > user_points), None
    )
    points_to_next = next_badge["points"] - user_points if next_badge else 0

    previous_badge = None
    for badge in REWARD_BADGES:
        if badge["points"] <= user_points:
            previous_badge = badge
        else:
            break

    segment_start_points = previous_badge["points"] if previous_badge else 0
    segment_start_label = (
        previous_badge["name"]
        if previous_badge
        else (REWARD_BADGES[0]["name"] if REWARD_BADGES else "起點")
    )
    if next_badge:
        segment_end_points = next_badge["points"]
        segment_end_label = next_badge["name"]
    else:
        segment_end_points = max_points or segment_start_points or user_points
        segment_end_label = REWARD_BADGES[-1]["name"] if REWARD_BADGES else "目標"
    segment_range = max(segment_end_points - segment_start_points, 1)
    segment_percent = (
        100.0
        if user_points >= segment_end_points
        else max(
            0.0,
            min(
                100.0,
                ((user_points - segment_start_points) / segment_range) * 100.0,
            ),
        )
    )

    enriched_badges = []
    for badge in REWARD_BADGES:
        enriched = dict(badge)
        enriched["unlocked"] = user_points >= badge["points"]
        enriched["position_percent"] = (
            (badge["points"] / max_points) * 100 if max_points else 0
        )
        enriched_badges.append(enriched)

    if user_email and "@" in user_email:
        user_name = user_email.split("@", 1)[0] or "朋友"
    else:
        user_name = user_email or "朋友"

    return render_template(
        "badges.html",
        reward_badges=enriched_badges,
        user_points=user_points,
        progress_percent=progress_percent,
        progress_segment_percent=segment_percent,
        progress_segment_start_points=segment_start_points,
        progress_segment_end_points=segment_end_points,
        progress_segment_start_label=segment_start_label,
        progress_segment_end_label=segment_end_label,
        next_badge=next_badge,
        points_to_next=points_to_next,
        max_badge_points=max_points,
        avatar_options=AVATAR_CHOICES,
        current_avatar=current_avatar,
        user_name=user_name,
        is_logged_in=True,
    )
