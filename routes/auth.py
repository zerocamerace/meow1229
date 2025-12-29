from __future__ import annotations

import logging

from flask import Blueprint, flash, redirect, render_template, request, session, url_for

from config.settings import DEFAULT_AVATAR, REWARD_BADGES
from google.cloud.firestore import SERVER_TIMESTAMP
from services.auth_points import (
    authenticate_with_password,
    award_login_points,
    delete_user_health_reports,
    delete_user_psychology_tests,
    delete_user_storage_files,
)
from services.firebase import auth, db
from utils.security import _form_keys, _mask_email

auth_bp = Blueprint("auth", __name__)


def _localized_login_error(message: str, email: str) -> str:
    code = (message or "").upper()
    msg = (message or "").lower()

    if (
        "EMAIL_NOT_FOUND" in code
        or "INVALID_LOGIN_CREDENTIALS" in code
        or "no user record" in msg
        or "invalid_login_credentials" in msg
    ):
        return "登入失敗：帳號或密碼有誤，請重新輸入。"
    if "INVALID_PASSWORD" in code or "password is invalid" in msg:
        return "登入失敗：帳號或密碼有誤，請重新輸入。"
    if "TOO_MANY_ATTEMPTS" in code or "too many attempts" in msg:
        return "登入失敗：嘗試次數過多，請稍後再試。"
    if "USER_DISABLED" in code or "user disabled" in msg:
        return "登入失敗：此帳號已被停用，請聯絡管理員。"
    if "AUTH_SERVICE_NOT_CONFIGURED" in code:
        return "登入失敗：尚未設定身份驗證服務，請通知系統管理員。"
    if "AUTH_NETWORK_ERROR" in code:
        return "登入失敗：驗證服務暫時無法使用，請稍後再試。"
    if "AUTH_RESPONSE_INVALID" in code:
        return "登入失敗：驗證結果格式異常，請稍後再試。"
    return "登入失敗：系統目前忙碌，請稍後再試。"


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    is_logged_in = "user_id" in session

    if request.method == "GET":
        session.pop("_flashes", None)

    if request.method == "POST":
        logging.debug("Register submission fields: %s", _form_keys(request.form))
        email = request.form.get("email")
        password = request.form.get("password")
        gender = request.form.get("gender")
        logging.debug("Parsed register data for email=%s", _mask_email(email))

        if not email or not password or not gender:
            flash("請輸入電子郵件、密碼和生理性別！", "error")
            logging.warning("Missing email, password, or gender in form submission")
            return render_template(
                "register.html",
                error="請輸入電子郵件、密碼和生理性別",
                is_logged_in=is_logged_in,
            )

        try:
            user = auth.create_user(email=email, password=password)
            logging.debug(
                "User created: uid=%s, email=%s", user.uid, _mask_email(email)
            )
            db.collection("users").document(user.uid).set(
                {
                    "email": email,
                    "gender": gender,
                    "created_at": SERVER_TIMESTAMP,
                    "last_login": None,
                    "avatar": DEFAULT_AVATAR,
                    "points": 0,
                }
            )
            logging.debug("User document created in Firestore for uid: %s", user.uid)
            session["user_id"] = user.uid
            session["user_email"] = email
            flash("註冊成功！請上傳健康報告。", "success")
            return redirect(url_for("cards.upload_health"))
        except Exception as exc:
            logging.error("Registration failed: %s", exc)
            flash("註冊失敗：請稍後再試。", "error")
            return render_template(
                "register.html", error=str(exc), is_logged_in=is_logged_in
            )

    return render_template("register.html", is_logged_in=is_logged_in)


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    is_logged_in = "user_id" in session

    if request.method == "GET":
        session.pop("_flashes", None)

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        logging.debug("Login attempt: email=%s", _mask_email(email))

        if not email or not password:
            flash("請輸入電子郵件和密碼！", "error")
            logging.warning("Missing email or password in login submission")
            return render_template("login.html", is_logged_in=is_logged_in)

        try:
            firebase_uid = authenticate_with_password(email, password)
            user_ref = db.collection("users").document(firebase_uid)
            snapshot = user_ref.get()
            user_data = {}
            if snapshot.exists:
                user_data = snapshot.to_dict() or {}
                user_ref.update({"last_login": SERVER_TIMESTAMP})
            else:
                user_data = {
                    "email": email,
                    "created_at": SERVER_TIMESTAMP,
                    "last_login": SERVER_TIMESTAMP,
                    "points": 0,
                    "avatar": DEFAULT_AVATAR,
                }
                user_ref.set(user_data, merge=True)
            if not user_data.get("avatar"):
                user_data["avatar"] = DEFAULT_AVATAR
                user_ref.set({"avatar": DEFAULT_AVATAR}, merge=True)
            points, new_badges = award_login_points(user_ref)
            session["user_id"] = firebase_uid
            session["user_email"] = email
            session["points"] = points
            session["avatar"] = user_data.get("avatar") or DEFAULT_AVATAR
            session["badge_thresholds"] = [
                badge["points"] for badge in REWARD_BADGES if points >= badge["points"]
            ]
            if new_badges:
                session["reward_unlocks"] = new_badges
            flash("登入成功！", "success")
            return redirect(url_for("main.home"))
        except ValueError as auth_error:
            error_code = str(auth_error)
            logging.warning("Login failed: %s", error_code)
            flash(_localized_login_error(error_code, email), "error")
            return render_template("login.html", is_logged_in=is_logged_in)
        except RuntimeError as auth_runtime:
            err = str(auth_runtime)
            logging.error("Password auth runtime error: %s", err)
            flash(_localized_login_error(err, email), "error")
            return render_template("login.html", is_logged_in=is_logged_in)
        except Exception as exc:
            logging.error("Unexpected login error: %s", exc)
            flash("登入失敗：系統目前忙碌，請稍後再試。", "error")
            return render_template("login.html", is_logged_in=is_logged_in)

    return render_template("login.html", is_logged_in=is_logged_in)


@auth_bp.route("/delete_account", methods=["GET", "POST"])
def delete_account():
    if "user_id" not in session:
        flash("請先登入。", "error")
        return redirect(url_for("auth.login"))

    user_id = session["user_id"]
    user_ref = db.collection("users").document(user_id)
    try:
        user_doc = user_ref.get()
        user_data = user_doc.to_dict() if user_doc.exists else {}
    except Exception as exc:
        logging.error("Failed to load user document for deletion: %s", exc)
        user_data = {}

    user_email = user_data.get("email") or ""

    if request.method == "POST":
        password = (request.form.get("password") or "").strip()
        confirm_delete = request.form.get("confirm_delete") == "yes"

        if not password:
            flash("請輸入密碼以確認刪除帳號。", "error")
            return render_template(
                "delete_account.html", user_email=user_email, is_logged_in=True
            )
        if not confirm_delete:
            flash("請勾選確認刪除，刪除後無法復原。", "error")
            return render_template(
                "delete_account.html", user_email=user_email, is_logged_in=True
            )
        if not user_email:
            flash("無法確認此帳號的電子郵件，請聯絡客服。", "error")
            return render_template(
                "delete_account.html", user_email=user_email, is_logged_in=True
            )

        try:
            authenticate_with_password(user_email, password)
        except ValueError as auth_error:
            flash(_localized_login_error(str(auth_error), user_email), "error")
            return render_template(
                "delete_account.html", user_email=user_email, is_logged_in=True
            )
        except RuntimeError as runtime_error:
            flash(_localized_login_error(str(runtime_error), user_email), "error")
            return render_template(
                "delete_account.html", user_email=user_email, is_logged_in=True
            )

        delete_user_psychology_tests(user_id)
        delete_user_health_reports(user_id)
        delete_user_storage_files(user_id)

        try:
            user_ref.delete()
        except Exception as exc:
            logging.warning("Failed to delete user document %s: %s", user_id, exc)

        try:
            auth.delete_user(user_id)
        except Exception as exc:
            logging.warning("Failed to delete auth user %s: %s", user_id, exc)

        session.clear()
        flash("帳戶已刪除。", "success")
        return redirect(url_for("main.home"))

    return render_template(
        "delete_account.html", user_email=user_email, is_logged_in=True
    )


@auth_bp.route("/logout")
def logout():
    session.pop("user_id", None)
    session.pop("_flashes", None)
    flash("已成功登出！", "success")
    return redirect(url_for("main.home"))
