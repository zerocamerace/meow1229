from __future__ import annotations

import logging
from datetime import datetime

import requests
from flask import session

from config.settings import (
    FIREBASE_AUTH_ENDPOINT,
    FIREBASE_WEB_API_KEY,
    REWARD_BADGES,
)
from services.firebase import bucket, db, auth


def authenticate_with_password(email: str, password: str) -> str:
    if not FIREBASE_WEB_API_KEY:
        logging.error("FIREBASE_WEB_API_KEY is not configured")
        raise RuntimeError("AUTH_SERVICE_NOT_CONFIGURED")

    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True,
    }
    try:
        resp = requests.post(
            f"{FIREBASE_AUTH_ENDPOINT}?key={FIREBASE_WEB_API_KEY}",
            json=payload,
            timeout=10,
        )
        data = resp.json()
    except requests.RequestException as exc:
        logging.error("Firebase password auth network error: %s", exc)
        raise RuntimeError("AUTH_NETWORK_ERROR") from exc

    if resp.status_code != 200:
        error_code = ((data.get("error") or {}).get("message") or "UNKNOWN_ERROR").upper()
        logging.warning("Firebase password auth failed with error code")
        raise ValueError("AUTH_FAILED")

    firebase_uid = data.get("localId")
    if not firebase_uid:
        logging.error(
            "Firebase auth response missing localId for email %s", email
        )
        raise RuntimeError("AUTH_RESPONSE_INVALID")
    logging.debug("Firebase auth succeeded for email=%s", email)
    return firebase_uid


def delete_user_health_reports(user_id: str):
    try:
        reports = (
            db.collection("health_reports").where("user_uid", "==", user_id).stream()
        )
        for doc in reports:
            doc.reference.delete()
    except Exception as exc:
        logging.warning("Failed to delete health reports for %s: %s", user_id, exc)


def delete_user_psychology_tests(user_id: str):
    try:
        tests = (
            db.collection("users").document(user_id).collection("psychology_tests").stream()
        )
        for doc in tests:
            doc.reference.delete()
    except Exception as exc:
        logging.warning(
            "Failed to delete psychology tests for %s: %s", user_id, exc
        )


def delete_user_storage_files(user_id: str):
    try:
        prefix = f"health_reports/{user_id}/"
        for blob in bucket.list_blobs(prefix=prefix):
            blob.delete()
    except Exception as exc:
        logging.warning("Failed to delete storage objects for %s: %s", user_id, exc)


def award_login_points(user_ref, daily_points: int = 5):
    try:
        snapshot = user_ref.get()
        data = snapshot.to_dict() or {}
    except Exception as exc:
        logging.warning("Failed to fetch user for points: %s", exc)
        data = {}

    today = datetime.now().strftime("%Y-%m-%d")
    last_date = data.get("last_point_login_date")
    current_points = int(data.get("points") or 0)

    if last_date == today:
        return current_points, []

    updated_points = current_points + daily_points
    unlocked_before = {
        badge["points"] for badge in REWARD_BADGES if current_points >= badge["points"]
    }
    unlocked_after = {
        badge["points"] for badge in REWARD_BADGES if updated_points >= badge["points"]
    }
    new_badges = sorted(unlocked_after - unlocked_before)

    try:
        user_ref.set(
            {
                "points": updated_points,
                "last_point_login_date": today,
            },
            merge=True,
        )
    except Exception as exc:
        logging.error("Failed to update points for %s: %s", user_ref.id, exc)
        return current_points, []

    return updated_points, new_badges


def maybe_award_daily_points():
    user_id = session.get("user_id")
    if not user_id:
        return

    today = datetime.now().strftime("%Y-%m-%d")
    if session.get("daily_points_check") == today:
        return

    try:
        user_ref = db.collection("users").document(user_id)
        points, new_badges = award_login_points(user_ref)
        session["points"] = points
        session["daily_points_check"] = today
        if new_badges:
            session["reward_unlocks"] = new_badges
    except Exception as exc:
        logging.debug("Daily point refresh skipped for %s: %s", user_id, exc)


def refresh_daily_points():
    maybe_award_daily_points()
