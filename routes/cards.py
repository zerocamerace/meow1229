from __future__ import annotations

import logging
import time
from datetime import datetime

from flask import Blueprint, flash, redirect, render_template, request, session, url_for

from config.settings import (
    ALLOWED_UPLOAD_EXTENSIONS,
    ALLOWED_UPLOAD_MIMES,
    CARD_LIMIT_MESSAGE,
    CARD_LIMIT_MODAL_TEXT,
    CAT_CARD_DIR,
    MAX_DAILY_CARD_GENERATIONS,
    MAX_DAILY_DEMO_GENERATIONS,
    MAX_DAILY_HEALTH_REPORT_UPLOADS,
    MAX_UPLOAD_BYTES,
    MAX_USER_TEXT_CHARS,
    OVER_LIMIT_MESSAGE,
)
from google.cloud.firestore import SERVER_TIMESTAMP
from services.health_report_module import analyze_health_report, calculate_health_score
from services.cards import build_cat_card, render_cat_card_image
from services.health import build_health_tips, normalize_health_data
from services.firebase import db
from utils.security import _form_keys, _mask_uid, _safe_url

cards_bp = Blueprint("cards", __name__)

DEMO_NOTICE = (
    "已套用35歲上班族的示範資料，生活型態：久坐、壓力大、偶爾運動"
    "（示範資料僅供體驗系統功能，非醫療建議）"
)
DEMO_LIMIT_MESSAGE = "已使用過範例資料！上傳真實報告以儲存數據＋解鎖完整版圖卡。"
DEMO_VITAL_STATS = {
    "glucose": 87,
    "hemoglobin_a1c": None,
    "total_cholesterol": 205,
    "triglycerides": 148,
    "ldl_cholesterol": 129,
    "hdl_cholesterol": 40,
    "bmi": 21.5,
    "alt": 8,
    "ast": 14,
    "creatinine": 0.7,
    "egfr": 120,
    "uric_acid": 4.5,
    "wbc": None,
    "rbc": None,
    "hemoglobin": None,
    "platelet": None,
    "urine_glucose": None,
    "urine_protein": 11.9,
    "blood_pressure_systolic": 100,
    "blood_pressure_diastolic": 65,
    "HBsAg": None,
    "urine_ob": None,
}


def _evaluate_daily_limit(user_doc: dict | None, count_field: str, date_field: str, limit: int):
    today_str = datetime.now().strftime("%Y-%m-%d")
    data = user_doc or {}
    last_date = data.get(date_field)
    try:
        daily_count = int(data.get(count_field) or 0)
    except (TypeError, ValueError):
        daily_count = 0
    if last_date != today_str:
        daily_count = 0
    allowed = daily_count < limit
    return allowed, daily_count, today_str


def _safe_next_path(value: str | None) -> str | None:
    if not value:
        return None
    value = str(value)
    if value.startswith("/") and not value.startswith("//"):
        return value
    return None


def _to_datetime(value):
    if value is None:
        return datetime.min
    if hasattr(value, "to_datetime"):
        try:
            return value.to_datetime()
        except TypeError:
            pass
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%Y%m%d", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
    return datetime.min


def _build_demo_report(user_id: str, gender: str | None) -> dict:
    health_score, health_warnings = calculate_health_score(
        DEMO_VITAL_STATS, gender=gender
    )
    report_date = datetime.now().strftime("%Y/%m/%d")
    return {
        "user_uid": user_id,
        "report_date": report_date,
        "file_type": "demo",
        "is_demo": True,
        "vital_stats": DEMO_VITAL_STATS,
        "health_score": health_score,
        "health_warnings": health_warnings,
        "created_at": datetime.now().replace(microsecond=0).isoformat(),
        "id": f"demo_{report_date}",
    }


@cards_bp.route("/upload_health", methods=["GET", "POST"])
def upload_health():
    if "user_id" not in session:
        flash("請先登錄！", "error")
        return redirect(url_for("auth.login"))

    user_id = session["user_id"]
    logging.debug("Current user_id from session: %s", _mask_uid(user_id))
    next_url = _safe_next_path(request.args.get("next") or request.form.get("next"))
    upload_redirect = (
        url_for("cards.upload_health", next=next_url)
        if next_url
        else url_for("cards.upload_health")
    )

    user_gender = None
    try:
        user_doc = db.collection("users").document(user_id).get()
        if not user_doc.exists:
            flash("找不到使用者資料！", "error")
            logging.warning(
                "User document not found for uid: %s", _mask_uid(user_id)
            )
            return redirect(url_for("auth.register"))
        user_data = user_doc.to_dict()
        user_gender = user_data.get("gender")
        if not user_gender:
            flash("請先完成註冊並提供生理性別資料！", "error")
            logging.warning(
                "User gender missing for uid: %s", _mask_uid(user_id)
            )
            return redirect(url_for("auth.register"))
        logging.debug("Retrieved user gender for uid %s", _mask_uid(user_id))
    except Exception as exc:
        logging.error(
            "Failed to retrieve user gender for %s: %s",
            _mask_uid(session.get("user_id")),
            exc,
        )
        flash("取得使用者資料失敗：請稍後再試。", "error")
        return redirect(url_for("auth.login"))

    demo_allowed, demo_used_count, demo_today = _evaluate_daily_limit(
        user_data,
        "demo_card_generations_count",
        "demo_last_card_generation_date",
        MAX_DAILY_DEMO_GENERATIONS,
    )

    reupload_requested = request.args.get("reupload") == "1"
    try:
        existing_reports = list(
            db.collection("health_reports")
            .where("user_uid", "==", user_id)
            .limit(1)
            .stream()
        )
    except Exception as exc:
        logging.error(
            "Failed to check existing health reports for %s: %s",
            _mask_uid(session.get("user_id")),
            exc,
        )
        existing_reports = []

    has_existing_report = bool(existing_reports)
    auto_redirect = False
    invalid_report_prompt = session.pop("invalid_report_prompt", False)
    if invalid_report_prompt:
        auto_redirect = False
    elif request.method == "GET" and has_existing_report and not reupload_requested:
        auto_redirect = True

    if request.method == "POST":
        if request.form.get("use_demo") == "1":
            if not demo_allowed:
                flash(DEMO_LIMIT_MESSAGE, "error")
                return redirect(upload_redirect)
            demo_report = _build_demo_report(user_id, user_gender)
            session["demo_report"] = demo_report
            flash(DEMO_NOTICE, "success")
            return redirect(next_url or url_for("cards.psychology_test"))

        upload_allowed, upload_count, upload_today = _evaluate_daily_limit(
            user_data,
            "daily_report_upload_count",
            "last_report_upload_date",
            MAX_DAILY_HEALTH_REPORT_UPLOADS,
        )
        if not upload_allowed:
            flash("今日上傳次數已達上限，請明天再試。", "error")
            return redirect(upload_redirect)

        if "health_report" not in request.files:
            flash("未選擇檔案！", "error")
            return redirect(upload_redirect)

        file = request.files["health_report"]
        if file.filename == "":
            flash("未選擇檔案！", "error")
            return redirect(upload_redirect)

        logging.debug(
            "Upload request fields=%s, files=%s",
            _form_keys(request.form),
            list(request.files.keys()),
        )

        filename_lower = file.filename.lower()
        if not filename_lower.endswith(ALLOWED_UPLOAD_EXTENSIONS):
            flash("請上傳 JPEG、PNG 或 PDF 檔！", "error")
            return redirect(upload_redirect)

        mimetype = (file.mimetype or "").lower()
        if mimetype not in ALLOWED_UPLOAD_MIMES:
            flash("請上傳 JPEG、PNG 或 PDF 檔！", "error")
            return redirect(upload_redirect)

        content_length = request.content_length
        if content_length and content_length > MAX_UPLOAD_BYTES:
            flash("檔案超過 10MB，請重新上傳！", "error")
            return redirect(upload_redirect)

        is_image = mimetype in {"image/jpeg", "image/png"}

        logging.debug("Starting health report analysis...")
        recognized_metric_count = 0
        try:
            file.seek(0)
            file_data = file.read()
            if len(file_data) > MAX_UPLOAD_BYTES:
                flash("檔案超過 10MB，請重新上傳！", "error")
                return redirect(upload_redirect)
            file_type = "image" if is_image else "pdf"
            (
                analysis_data,
                health_score,
                health_warnings,
                recognized_metric_count,
            ) = analyze_health_report(
                file_data, user_id, file_type, gender=user_gender
            )
            logging.debug(
                "Analysis result - data: %s, score: %s, warnings: %s, metrics: %s",
                analysis_data is not None,
                health_score,
                health_warnings,
                recognized_metric_count,
            )
            if recognized_metric_count <= 0 or not analysis_data:
                logging.warning(
                    "No recognizable health metrics were extracted; prompting re-upload"
                )
                session["invalid_report_prompt"] = True
                flash("未辨識為健檢報告", "invalid_report")
                return redirect(upload_redirect)
        except Exception as exc:
            logging.error("Health report analysis failed: %s", exc)
            flash("健康報告分析失敗：請稍後再試。", "warning")
            analysis_data, health_score, health_warnings, recognized_metric_count = (
                None,
                0,
                [],
                0,
            )

        health_report_doc = {
            "user_uid": user_id,
            "report_date": datetime.now().strftime("%Y/%m/%d"),
            "file_type": file_type,
            "created_at": SERVER_TIMESTAMP,
        }
        if analysis_data:
            health_report_doc.update(
                {
                    "vital_stats": analysis_data.get("vital_stats", {}),
                    "health_score": health_score,
                    "health_warnings": health_warnings,
                }
            )
            logging.debug(
                "Adding analysis data to doc: score=%s, warnings=%s",
                health_score,
                health_warnings,
            )

        doc_ref = db.collection("health_reports").document()
        doc_ref.set(health_report_doc)
        report_id = doc_ref.id
        logging.debug(
            "Health report saved for user %s with report_id %s", user_id, report_id
        )
        try:
            db.collection("users").document(user_id).set(
                {
                    "daily_report_upload_count": upload_count + 1,
                    "last_report_upload_date": upload_today,
                },
                merge=True,
            )
        except Exception as exc:
            logging.warning("Failed to update report upload count for %s: %s", user_id, exc)

        session.pop("demo_report", None)

        saved_doc = db.collection("health_reports").document(report_id).get()
        if saved_doc.exists:
            logging.debug(
                "Firestore write verified - document exists: %s",
                saved_doc.to_dict(),
            )
        else:
            logging.error("Firestore write failed - document does not exist")

        flash(
            f"上傳成功！健康分數：{health_score}，警告：{'; '.join(health_warnings) if health_warnings else '無'}",
            "success",
        )
        return redirect(next_url or url_for("cards.psychology_test"))

    return render_template(
        "upload_health.html",
        force_reupload=reupload_requested,
        has_existing_report=has_existing_report,
        auto_redirect=auto_redirect,
        invalid_report_prompt=invalid_report_prompt,
        psychology_url=next_url or url_for("cards.psychology_test"),
        demo_notice=DEMO_NOTICE,
        demo_available=demo_allowed,
        demo_limit_message=DEMO_LIMIT_MESSAGE,
        next_url=next_url,
    )


@cards_bp.route("/psychology_test", methods=["GET", "POST"])
def psychology_test():
    if "user_id" not in session:
        flash("請先登入！", "error")
        return redirect(url_for("auth.login"))

    user_id = session["user_id"]
    demo_report = session.get("demo_report")
    try:
        health_reports = list(
            db.collection("health_reports").where("user_uid", "==", user_id).stream()
        )
        logging.debug(
            "Psychology test check - existing reports: %d", len(health_reports)
        )
        if not health_reports and not demo_report:
            flash("請先上傳健康報告！", "error")
            return redirect(url_for("cards.upload_health"))
    except Exception as exc:
        logging.error("Error checking health reports: %s", exc)
        if not demo_report:
            flash("檢查健康報告失敗：請稍後再試。", "error")
            return redirect(url_for("cards.upload_health"))
        health_reports = []

    if request.method == "GET":
        session.pop("_flashes", None)

        latest_report_data = None
        try:
            if demo_report:
                latest_report_data = demo_report
            else:
                def _report_sort_key(doc_snapshot):
                    data = doc_snapshot.to_dict() or {}
                    created = data.get("created_at")
                    if hasattr(created, "timestamp"):
                        return created.timestamp()
                    return 0.0

                if health_reports:
                    latest_snapshot = max(health_reports, key=_report_sort_key)
                    latest_report_data = latest_snapshot.to_dict() or {}
                    created_at = latest_report_data.get("created_at")
                    if hasattr(created_at, "isoformat"):
                        latest_report_data["created_at"] = created_at.isoformat()
        except Exception as exc:
            logging.warning(
                "Failed to prepare latest health report for template: %s", exc
            )
            latest_report_data = None

        return render_template(
            "psychology_test.html",
            is_logged_in=True,
            latest_health_report=latest_report_data,
        )

    question1 = (request.form.get("question1") or "").strip()
    question2 = (request.form.get("question2") or "").strip()
    if not question1 or not question2:
        flash("請回答所有問題！", "error")
        return render_template(
            "psychology_test.html", error="請回答所有問題", is_logged_in=True
        )
    if len(question1) > MAX_USER_TEXT_CHARS or len(question2) > MAX_USER_TEXT_CHARS:
        flash(OVER_LIMIT_MESSAGE, "error")
        return render_template(
            "psychology_test.html",
            error=OVER_LIMIT_MESSAGE,
            is_logged_in=True,
        )

    try:
        db.collection("users").document(user_id).collection(
            "psychology_tests"
        ).add(
            {
                "question1": question1,
                "question2": question2,
                "submit_time": SERVER_TIMESTAMP,
            }
        )
        logging.debug("Psychology test saved to Firestore for uid: %s", user_id)
        flash("表單已送出！歡迎前往生成貓咪圖卡。", "success")
        return redirect(url_for("cards.generate_card"))
    except Exception as exc:
        logging.error("Psychology test error: %s", exc)
        flash("提交失敗：請稍後再試。", "error")
        return render_template(
            "psychology_test.html",
            error=f"提交失敗：{str(exc)}",
            is_logged_in=True,
        )


@cards_bp.route("/generate_card")
def generate_card():
    if "user_id" not in session:
        flash("請先登入！", "error")
        return redirect(url_for("auth.login"))

    session.pop("_flashes", None)

    try:
        user_id = session["user_id"]
        user_ref = db.collection("users").document(user_id)
        try:
            user_doc_data = user_ref.get().to_dict() or {}
        except Exception as exc:
            logging.warning("Failed to load user doc for card limit: %s", exc)
            user_doc_data = {}
        demo_report = session.get("demo_report")
        demo_mode = isinstance(demo_report, dict) and demo_report.get("is_demo")
        if demo_mode:
            demo_allowed, demo_count, demo_today = _evaluate_daily_limit(
                user_doc_data,
                "demo_card_generations_count",
                "demo_last_card_generation_date",
                MAX_DAILY_DEMO_GENERATIONS,
            )
            if not demo_allowed:
                flash(DEMO_LIMIT_MESSAGE, "error")
                return redirect(url_for("cards.upload_health"))
        else:
            allowed, card_count, today_str = _evaluate_daily_limit(
                user_doc_data,
                "daily_card_generations_count",
                "last_card_generation_date",
                MAX_DAILY_CARD_GENERATIONS,
            )
            if not allowed:
                flash(CARD_LIMIT_MESSAGE, "error")
                session["show_card_limit_modal"] = True
                session["card_limit_modal_text"] = CARD_LIMIT_MODAL_TEXT
                return redirect(url_for("main.home"))

        reports = []
        if demo_mode:
            demo_report.setdefault(
                "id", f"demo_{demo_report.get('report_date') or 'session'}"
            )
            reports = [demo_report]
        else:
            health_report_docs = (
                db.collection("health_reports").where("user_uid", "==", user_id).stream()
            )
            for doc in health_report_docs:
                data = doc.to_dict() or {}
                data["id"] = doc.id
                reports.append(data)
            logging.debug("Generate card - reports found: %d", len(reports))
            if not reports:
                flash("請先上傳健康報告！", "error")
                return redirect(url_for("cards.upload_health"))

        psych_docs = (
            db.collection("users").document(user_id).collection("psychology_tests").stream()
        )
        tests = []
        for doc in psych_docs:
            data = doc.to_dict() or {}
            data["id"] = doc.id
            tests.append(data)
        if not tests:
            flash("請先完成心理測驗！", "error")
            return redirect(url_for("cards.psychology_test"))

        sorted_reports = sorted(
            reports,
            key=lambda r: _to_datetime(r.get("created_at") or r.get("report_date")),
        )
        latest_report = sorted_reports[-1]
        warnings, vitals_display = normalize_health_data(latest_report)
        latest_report["_display_warnings"] = warnings
        latest_report["_display_vitals"] = vitals_display
        health_tips = build_health_tips(latest_report, warnings)
        history_labels = []
        history_scores = []
        for entry in sorted_reports[-6:]:
            score_value = entry.get("health_score")
            if score_value is None:
                continue
            created_at = _to_datetime(entry.get("created_at") or entry.get("report_date"))
            if created_at and created_at != datetime.min:
                label = created_at.strftime("%m/%d")
            else:
                label = entry.get("report_date") or "N/A"
            history_labels.append(label)
            history_scores.append(score_value)

        latest_test = max(
            tests,
            key=lambda t: _to_datetime(t.get("submit_time") or t.get("created_at")),
        )

        use_cache = request.args.get("nocache") != "1"
        cache_key_current = f"{latest_report.get('id')}_{latest_test.get('id')}"
        cache_entry = session.get("cat_card_cache") if use_cache else None
        card_payload = None
        image_filename = None
        cat_source = None

        if cache_entry:
            cache_path = CAT_CARD_DIR / cache_entry.get("filename", "")
            cache_age = time.time() - cache_entry.get("timestamp", 0)
            cache_key_match = cache_entry.get("cache_key") == cache_key_current
            if cache_path.exists() and cache_age < 3600 and cache_key_match:
                logging.debug("Using cached cat card for user %s", user_id)
                card_payload = cache_entry.get("card", {})
                image_filename = cache_entry.get("filename")
                cat_source = _safe_url(cache_entry.get("cat_source"))

        if not card_payload or not image_filename:
            card_payload = build_cat_card(latest_report, latest_test)
            if warnings:
                card_payload["warnings"] = warnings
            cache_key = cache_key_current
            image_filename, cat_source = render_cat_card_image(
                card_payload, user_id, cache_key=cache_key
            )
            session["cat_card_cache"] = {
                "timestamp": time.time(),
                "filename": image_filename,
                "cat_source": cat_source,
                "card": card_payload,
                "cache_key": cache_key,
            }

        card_image_url = url_for("static", filename=f"cat_cards/{image_filename}")
        card_payload["image_url"] = card_image_url
        card_payload["cat_image_source"] = cat_source
        card_payload.setdefault("warnings", warnings)
        logging.debug("cat image source selected: %s", cat_source)

        try:
            if demo_mode:
                user_ref.set(
                    {
                        "is_demo_user": True,
                        "demo_card_generations_count": demo_count + 1,
                        "demo_last_card_generation_date": demo_today,
                    },
                    merge=True,
                )
            else:
                user_ref.set(
                    {
                        "daily_card_generations_count": card_count + 1,
                        "last_card_generation_date": today_str,
                    },
                    merge=True,
                )
        except Exception as exc:
            logging.warning("Failed to update card count for %s: %s", user_id, exc)

        return render_template(
            "generate_card.html",
            card=card_payload,
            card_image_url=card_image_url,
            report=latest_report,
            psychology=latest_test,
            health_tips=health_tips,
            health_history_labels=history_labels,
            health_history_scores=history_scores,
            is_logged_in=True,
            demo_mode=demo_mode,
        )
    except Exception as exc:
        logging.error(
            "Generate card error for user %s: %s",
            _mask_uid(session.get("user_id")),
            exc,
        )
        flash("生成貓咪圖卡失敗，請稍後再試。", "error")
        return render_template(
            "generate_card.html",
            error="生成圖卡失敗，請稍後再試。",
            is_logged_in=True,
        )
