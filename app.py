# -*- coding: utf-8 -*-
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    jsonify,
    Response,
    abort,
)
import firebase_admin
from firebase_admin import credentials, firestore, storage, auth
from firebase_admin.exceptions import FirebaseError
import os
import socket
import ipaddress
from datetime import datetime, timezone
import logging
import time
import requests  # ?? 0929修改：呼叫外部貓圖來源
import random  # ?? 0929修改：貓咪圖卡風格隨機與備援使用
import textwrap  # ?? 0929修改：圖卡文字換行處理
import hashlib  # ?? 0929修改：圖卡輸出避免檔名衝突
import imghdr  # ?? 0929修改：驗證下載圖片格式
from pathlib import Path  # ?? 0929修改：設定圖卡輸出路徑
from io import BytesIO  # ?? 0929修改：處理圖片位元組資料
from urllib.parse import urlparse  # ?? 0929修改：驗證圖片網址安全性

from PIL import (
    Image,
    ImageDraw,
    ImageFont,
    ImageOps,
    ImageFilter,
)  # ?? 0929修改：繪製圖卡
from health_report_module import analyze_health_report
from google.cloud.firestore import SERVER_TIMESTAMP
from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv
import json
import re


MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10MB
ALLOWED_UPLOAD_EXTENSIONS = (".jpg", ".jpeg", ".png", ".pdf")
ALLOWED_UPLOAD_MIMES = {
    "image/jpeg",
    "image/png",
    "application/pdf",
}
MAX_USER_TEXT_CHARS = 1000
OVER_LIMIT_MESSAGE = "Oops...字數超過1000字無法傳送唷"
MAX_DAILY_CARD_GENERATIONS = 10
CARD_LIMIT_MESSAGE = "今日生成次數已達上限，請明天再試。"


def extract_json_from_response(text: str) -> dict:
    """抽取第一個 JSON 物件並解析。"""  # 0929修改03：強化解析容錯
    if text is None:
        raise ValueError("LLM returned None")

    raw = str(text).strip()

    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)

    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        raise ValueError(f"No JSON object found in: {raw[:200]}")

    candidate = match.group(0)
    candidate = candidate.replace("”", '"').replace("’", "'").replace("\ufeff", "")

    return json.loads(candidate)


def _build_genai_contents(system_instruction, conversation_history):
    contents = []

    if system_instruction:
        contents.append(
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=str(system_instruction))],
            )
        )

    for msg in conversation_history:
        role = msg.get("role", "user")
        parts = msg.get("parts", [])
        if not parts:
            logging.warning(f"Empty parts in message: {msg}")
            continue

        part_obj = parts[0]
        if isinstance(part_obj, dict):
            text = part_obj.get("text", "")
        else:
            text = str(part_obj)

        if not text:
            logging.warning(f"Empty text in message: {msg}")
            continue

        genai_role = "model" if role == "model" else "user"
        contents.append(
            genai_types.Content(
                role=genai_role,
                parts=[genai_types.Part(text=text)],
            )
        )

    return contents


def _generate_with_retry(contents, generation_config=None):
    model_candidates = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ]  # 0929修改04：移除 1.5 系列，改用 2.5 模型

    last_error = None

    for model_name in model_candidates:
        for attempt in range(3):
            try:
                kwargs = {
                    "model": model_name,
                    "contents": contents,
                }
                if generation_config is not None:
                    kwargs["config"] = generation_config

                response = genai_chat_client.models.generate_content(**kwargs)
                if getattr(response, "candidates", None):
                    if attempt > 0 or model_name != model_candidates[0]:
                        logging.warning(
                            f"Gemini model '{model_name}' succeeded on attempt {attempt + 1}"
                        )
                    return response
                logging.warning(
                    f"Gemini model '{model_name}' returned empty candidates on attempt {attempt + 1}"
                )
            except Exception as e:
                logging.warning(
                    f"Gemini model '{model_name}' failed on attempt {attempt + 1}: {e}"
                )
                last_error = e
            time.sleep(1)

    if last_error:
        raise last_error
    raise RuntimeError("Gemini API returned empty response for all candidate models")


FIREBASE_AUTH_ENDPOINT = (
    "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"
)


def _authenticate_with_password(email: str, password: str) -> str:
    """Use Firebase REST API to verify email/password login."""
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
        logging.warning("Firebase password auth failed: %s", error_code)
        raise ValueError(error_code)

    firebase_uid = data.get("localId")
    if not firebase_uid:
        logging.error("Firebase auth response missing localId: %s", data)
        raise RuntimeError("AUTH_RESPONSE_INVALID")
    return firebase_uid


# ?????????
def _delete_user_health_reports(user_id: str):
    try:
        reports = (
            db.collection("health_reports").where("user_uid", "==", user_id).stream()
        )
        for doc in reports:
            doc.reference.delete()
    except Exception as exc:
        logging.warning("Failed to delete health reports for %s: %s", user_id, exc)


def _delete_user_psychology_tests(user_id: str):
    try:
        tests = (
            db.collection("users").document(user_id).collection("psychology_tests").stream()
        )
        for doc in tests:
            doc.reference.delete()
    except Exception as exc:
        logging.warning("Failed to delete psychology tests for %s: %s", user_id, exc)


def _delete_user_storage_files(user_id: str):
    try:
        prefix = f"health_reports/{user_id}/"
        for blob in bucket.list_blobs(prefix=prefix):
            blob.delete()
    except Exception as exc:
        logging.warning("Failed to delete storage objects for %s: %s", user_id, exc)


def _award_login_points(user_ref, daily_points: int = 5):
    """Add login points if user hasn't received points today. Returns (new_points, new_badges)."""
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


def _maybe_award_daily_points():
    """Ensure daily login points are granted even if user stays logged in past midnight."""
    user_id = session.get("user_id")
    if not user_id:
        return

    today = datetime.now().strftime("%Y-%m-%d")
    if session.get("daily_points_check") == today:
        return

    try:
        user_ref = db.collection("users").document(user_id)
        points, new_badges = _award_login_points(user_ref)
        session["points"] = points
        session["daily_points_check"] = today
        if new_badges:
            session["reward_unlocks"] = new_badges
    except Exception as exc:
        logging.debug("Daily point refresh skipped for %s: %s", user_id, exc)


def _refresh_daily_points():
    _maybe_award_daily_points()


# ?? 0929修改：共用工具
_DISALLOWED_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _is_public_host(hostname: str) -> bool:
    """Return True if hostname resolves only to public IPs."""
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

    if parsed.scheme not in {"http", "https"}:
        return None

    host = parsed.hostname
    if not host:
        return None

    if host.lower() in _DISALLOWED_HOSTS:
        return None

    if not _is_public_host(host):
        return None

    return parsed.geturl()


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


def _form_keys(form_data) -> list[str]:
    try:
        return list(form_data.keys())
    except AttributeError:
        return []


def _is_text_within_limit(text: str | None, limit: int = MAX_USER_TEXT_CHARS) -> bool:
    if text is None:
        return True
    return len(text) <= limit


def _user_messages_within_limit(conversation, limit: int = MAX_USER_TEXT_CHARS) -> bool:
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


def _evaluate_daily_limit(
    user_doc: dict | None,
    count_field: str,
    date_field: str,
    limit: int,
) -> tuple[bool, int, str]:
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


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        font_path = Path(path)
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), size)
            except Exception:
                continue
    logging.debug("Font fallback engaged for size %s", size)
    return ImageFont.load_default()


def _wrap_text(text: str | None, max_chars: int = 18) -> str:
    if not text:
        return ""
    collapsed = str(text).replace("\n", " ")
    return "\n".join(textwrap.wrap(collapsed, width=max_chars, break_long_words=True))


def _hex_to_rgb(hex_value: str) -> tuple[int, int, int]:
    hex_value = hex_value.lstrip("#")
    return tuple(int(hex_value[i : i + 2], 16) for i in (0, 2, 4))


def _hash_for_filename(*parts: str) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(part.encode("utf-8", errors="ignore"))
    return hasher.hexdigest()[:8]


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
        for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%Y%m%d"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
    return datetime.min


def _cleanup_old_cards(max_files: int = 40):  # ?? 0929修改：限制圖卡輸出數量
    try:
        files = sorted(
            CAT_CARD_DIR.glob("catcard_*.png"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for stale in files[max_files:]:
            stale.unlink(missing_ok=True)
    except Exception as exc:
        logging.warning("Failed to cleanup old cat cards: %s", exc)


def _normalize_health_data(report: dict):
    """Collect warnings與重要指標，確保與舊版呈現一致。"""  # ?? 0929修改：整理健檢資料給前端顯示
    warnings = []
    for key in ("health_warnings", "warnings", "alert_list", "warning_details"):
        value = report.get(key)
        if not value:
            continue
        if isinstance(value, list):
            warnings.extend(str(item) for item in value if item)
        elif isinstance(value, dict):
            warnings.extend(str(item) for item in value.values() if item)
        else:
            warnings.append(str(value))
    warnings = [w.strip() for w in warnings if w and isinstance(w, str)]

    vitals_display = []
    vitals = report.get("vital_stats") or report.get("vitals") or {}
    if isinstance(vitals, dict):
        vitals_iter = vitals.items()
    elif isinstance(vitals, list):
        vitals_iter = []
        for item in vitals:
            if isinstance(item, dict):
                vitals_iter.extend(item.items())
            else:
                vitals_display.append((str(item), ""))
    else:
        vitals_iter = []

    for key, value in vitals_iter:
        if value is None or value == "":
            continue
        vitals_display.append((str(key), str(value)))

    return warnings, vitals_display


HEALTH_RECOMMENDATIONS = [
    ("膽固醇", "減少油炸與加工食品，增加可溶性纖維與omega-3脂肪酸攝取。"),
    ("血糖", "控制精緻糖攝取，注意三餐定時並搭配適量運動。"),
    ("血壓", "減少鈉攝取，保持作息與壓力管理，維持充足睡眠。"),
    ("體重", "規律運動並調整飲食份量，朝向健康體重範圍。"),
    ("肝", "減少酒精與高脂飲食，必要時尋求醫師評估。"),
]


def _build_health_tips(report: dict, warnings: list[str], limit: int = 3):
    """Create health tips based on warnings or vitals for the '了解更多' section."""
    tips = []
    vitals = report.get("vital_stats") or {}

    def _to_number(value):
        try:
            if isinstance(value, str):
                value = value.replace(",", "").strip()
            return float(value)
        except (TypeError, ValueError):
            return None

    def _get_recommendation(text: str) -> str | None:
        lower = text.lower()
        for keyword, advice in HEALTH_RECOMMENDATIONS:
            if keyword.lower() in lower:
                return advice
        return None

    for warning in warnings:
        recommendation = _get_recommendation(warning) or "持續調整飲食與作息，並諮詢專業醫師。"
        tips.append(
            {
                "title": "健康提醒",
                "description": warning,
                "recommendation": recommendation,
                "value": None,
                "threshold": None,
                "percent": None,
            }
        )
        if len(tips) >= limit:
            break

    if len(tips) < limit:
        metrics = (
            ("total_cholesterol", "總膽固醇", 200),
            ("ldl_cholesterol", "LDL 壞膽固醇", 130),
            ("glucose", "空腹血糖", 100),
            ("triglycerides", "三酸甘油脂", 150),
            ("bmi", "BMI", 24),
        )
        for key, label, threshold in metrics:
            if len(tips) >= limit:
                break
            value = vitals.get(key)
            numeric_value = _to_number(value)
            if numeric_value is not None:
                percent = None
                if threshold:
                    percent = max(0, min((numeric_value / threshold) * 100.0, 130))
                tips.append(
                    {
                        "title": label,
                        "description": f"{label} 目前為 {numeric_value}",
                        "recommendation": _get_recommendation(label) or "保持規律運動、均衡飲食與充足睡眠。",
                        "value": numeric_value,
                        "threshold": threshold,
                        "percent": percent,
                    }
                )

    if not tips:
        tips.append(
            {
                "title": "保持良好習慣",
                "description": "目前沒有紅字，但仍建議規律作息、適量運動並持續追蹤健康。",
                "recommendation": "保持良好生活型態，並定期回診追蹤各項指標。",
                "value": None,
                "threshold": None,
                "percent": None,
            }
        )
    return tips


# ?? 0929修改：九宮格貓咪分區
def _score_to_interval(score) -> int | None:
    """將數值分數換成 1~3 區間。"""
    if score is None:
        return None
    try:
        value = float(score)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    if value <= 33:
        return 1
    if value <= 66:
        return 2
    return 3


def _resolve_persona_key(health_score, mind_score) -> str | None:
    """根據身心分數挑選對應的既有貓咪圖。"""  # ?? 0929修改：依分數選擇貓咪類型
    physical_zone = _score_to_interval(health_score)
    mental_zone = _score_to_interval(mind_score)
    if not physical_zone or not mental_zone:
        return None
    prefix = {1: "C", 2: "B", 3: "A"}.get(mental_zone)
    if not prefix:
        return None
    return f"{prefix}{physical_zone}"


def _validate_report_schema(payload: dict) -> dict:
    """驗證 Gemini 報告 JSON 結構，避免後續操作失敗。"""  # ?? 0929修改05：補上遺失的 schema 檢查 helper
    if not isinstance(payload, dict):
        raise TypeError("payload must be dict")

    for key in ("summary", "keywords", "emotionVector"):
        if key not in payload:
            raise ValueError(f"Missing key: {key}")

    if not isinstance(payload["summary"], str):
        raise TypeError("summary must be string")

    keywords = payload.get("keywords")
    if not isinstance(keywords, list) or not all(
        isinstance(item, str) for item in keywords
    ):
        raise TypeError("keywords must be list[str]")

    emotion_vector = payload.get("emotionVector")
    if not isinstance(emotion_vector, dict):
        raise TypeError("emotionVector must be object")

    for key in ("valence", "arousal", "dominance"):
        if key not in emotion_vector:
            raise ValueError(f"emotionVector missing key: {key}")
        if not isinstance(emotion_vector[key], (int, float)):
            raise TypeError(f"emotionVector.{key} must be number")

    return payload


def fetch_cat_image(
    max_retries: int = 3, timeout: int = 12, max_bytes: int = 8_000_000
):
    """從 TheCatAPI 取得貓圖，失敗時改用備援圖庫。"""  # ?? 0929修改：新增貓圖來源
    api_url = "https://api.thecatapi.com/v1/images/search?size=med&mime_types=jpg,png"
    headers = {}
    api_key = os.getenv("CAT_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    backoff = 1
    for attempt in range(max_retries):
        try:
            resp = requests.get(api_url, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                payload = resp.json() or []
                if not payload:
                    raise ValueError("Cat API returned empty list")
                img_url = payload[0].get("url")
                if not img_url:
                    raise ValueError("Cat API payload missing url")
                image_bytes, final_url = _download_image(img_url, timeout, max_bytes)
                if image_bytes:
                    return image_bytes, final_url
            elif resp.status_code in {429, 500, 502, 503, 504}:
                logging.warning(
                    "Cat API temporary failure %s, backoff %ss",
                    resp.status_code,
                    backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 8)
                continue
            else:
                raise ValueError(
                    f"Cat API status {resp.status_code}: {resp.text[:200]}"
                )
        except Exception as exc:
            logging.warning(
                "Cat API request failed (attempt %s/%s): %s",
                attempt + 1,
                max_retries,
                exc,
            )
            time.sleep(backoff)
            backoff = min(backoff * 2, 8)

    logging.warning("Cat API all retries exhausted, switching to fallback image pool")
    fallback_url = random.choice(CAT_FALLBACK_IMAGES)
    image_bytes, final_url = _download_image(
        fallback_url, timeout, max_bytes, allow_fallback_errors=False
    )
    if image_bytes:
        return image_bytes, final_url
    logging.error("Fallback gallery also failed, using placeholder")
    placeholder = Image.new("RGB", (512, 512), "#fddde6")
    return placeholder, None


def _download_image(
    url: str, timeout: int, max_bytes: int, allow_fallback_errors: bool = True
):
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        if resp.status_code != 200:
            raise ValueError(f"Image status {resp.status_code}")
        content_type = resp.headers.get("Content-Type", "")
        if "image" not in content_type:
            raise ValueError(f"Unexpected content-type {content_type}")
        content_length = int(resp.headers.get("Content-Length", "0"))
        if content_length and content_length > max_bytes:
            raise ValueError(f"Image too large: {content_length}")
        data = resp.content
        if len(data) > max_bytes:
            raise ValueError("Image exceeds max_bytes")
        kind = imghdr.what(None, data)
        if kind not in {"jpeg", "png", "webp"}:
            raise ValueError(f"Unsupported image type: {kind}")
        return data, url
    except Exception as exc:
        if allow_fallback_errors:
            logging.warning("Download image failed for %s: %s", url, exc)
        else:
            logging.error("Download fallback image failed for %s: %s", url, exc)
        return None, None


def generate_cat_card_text(report: dict, psychology: dict, preferred_style: str):
    """呼叫 Gemini 產生貓卡敘述與生活推薦。"""  # ?? 1007 修改『圖卡生成推薦』：擴充文案欄位
    prompt = (
        "你是一位數位貓咪圖卡策展師，會依據使用者的健康與心理測驗資料提供陪伴資訊。\n"
        "請回傳 JSON，必須包含下列欄位：\n"
        "- styleKey: bright/steady/healer 之一。\n"
        "- persona: 角色定位。\n"
        "- name: 貓咪名稱。\n"
        "- speech: 15 字內、溫暖有記憶點的開場話。\n"
        "- summary: 60 字內，以故事化語氣描述當前狀態，不可提到任何精確數值或檢驗指標。\n"
        "- insight: 50 字內，給出情緒洞察，避免出現具體數字或醫療指標名稱。\n"
        "- action: 40 字內，提出實際可行的小提醒，同樣勿出現具體檢驗數值。\n"
        "- keywords: 陣列，可空。\n"
        "- recommendations: 物件，內含 movie/music/activity 三個子欄位，每個子欄位需提供 title 與 reason。\n"
        "  * movie: 推薦一部符合當前情緒需求的電影或影集，reason 需點出氛圍或療癒重點。\n"
        "  * music: 推薦一首歌曲或播放清單，說明為何適合此刻的節奏。\n"
        "  * activity: 推薦一個放鬆或充電的小活動，要具體且富有品味。\n"
        "所有文字務必使用繁體中文，保持溫柔且有品味，不可引用精確的血壓、血脂等數值。\n"
        f"建議風格：{preferred_style}\n"
        f"健康資料：{json.dumps(report, ensure_ascii=False, default=str)}\n"
        f"心理測驗：{json.dumps(psychology, ensure_ascii=False, default=str)}"
    )

    contents = _build_genai_contents(prompt, [])
    try:
        response = _generate_with_retry(
            contents, generation_config=JSON_RESPONSE_CONFIG
        )
        if not response or not getattr(response, "candidates", None):
            return None
        candidate = response.candidates[0]
        text = ""
        parts = (
            getattr(candidate.content, "parts", None) or []
        )  # 0929修改03：parts 可能為 None，改採空清單避免迴圈錯誤
        for part in parts:
            if getattr(part, "text", None):
                text += part.text
        try:
            parsed = extract_json_from_response(text)
        except Exception:
            logging.exception(
                "0929修改03：Cat card JSON parse failed; raw snippet=%r", text[:500]
            )
            parsed = None
        if isinstance(parsed, dict):
            return parsed
        logging.warning("Cat card text fallback due to unparsable response")
    except Exception as exc:
        logging.error(f"generate_cat_card_text failed: {exc}")
    return None


CAT_STYLES = {
    "bright": {
        "title": "陽光守護者",
        "names": ["小橘光", "暖暖", "Sunny 喵"],
        "speech": ["今天也要補充水分喵！", "保持笑容，活力滿分！"],
        "description": "我感受到你{mood}的能量，讓我們把這份耀眼溫度延續下去。",
        "actions": [
            "午休時間散步 10 分鐘，讓身體熱起來",
            "今天晚餐試試多彩蔬菜盤，補充維生素",
        ],
        "palette": ("#FFEAA7", "#FD79A8", "#FFAFCC", "#2d3436"),
        "curations": {
            "movie": ("《翻滾吧！阿信》", "熱血卻富含體貼的勵志故事，帶來向上的動力"),
            "music": ("City Pop 暖陽歌單", "輕快律動喚醒身體的節奏感"),
            "activity": ("戶外晨間伸展", "在陽光下活動筋骨，吸收自然能量"),
        },     # ?? 1007 修改『圖卡生成推薦』
    },
    "steady": {
        "title": "溫柔照護隊長",
        "names": ["小霧", "Cotton", "霜霜"],
        "speech": ["放慢腳步，我陪著你喵。", "今天也記得深呼吸三次。"],
        "description": "你的關鍵字是 {mood}，我會在日常提醒你保持節奏，讓步調更穩定。",
        "actions": [
            "睡前做 5 分鐘伸展，放鬆肌肉",
            "把今天的情緒寫在手帳，整理一下心緒",
        ],
        "palette": ("#E0FBFC", "#98C1D9", "#3D5A80", "#2d3436"),
        "curations": {
            "movie": ("《小森林》", "四季料理與田園步調，撫慰敏感心緒"),
            "music": ("Lo-fi 書寫清單", "柔和節拍陪你整理思緒"),
            "activity": ("手寫一封慢信", "用文字梳理情緒，讓心安定下來"),
        },
    },
    "healer": {
        "title": "療癒訓練師",
        "names": ["小湯圓", "Mochi", "露露"],
        "speech": ["我們慢慢來，沒關係的喵。", "先照顧好自己，我在旁邊。"],
        "description": "看見你需要休息的訊號，我會當你的提醒小鬧鐘，陪你一起慢慢修復。",
        "actions": [
            "安排 15 分鐘的呼吸練習，舒緩壓力",
            "今天對自己說聲辛苦了，給自己一個擁抱",
        ],
        "palette": ("#E8EAF6", "#C5CAE9", "#9FA8DA", "#2d3436"),
        "curations": {
            "movie": ("《海邊的曼徹斯特》", "細膩描寫失落後的修復，讓情緒被看見"),
            "music": ("Neo Classical 冥想曲", "舒緩鋼琴聲穩定呼吸節奏"),
            "activity": ("居家香氛冥想", "點上喜歡的味道，跟著引導冥想放鬆"),
        },
    },
}

# ?? 1007 修改『圖卡生成推薦』：預設影音與活動建議
def _fallback_recommendations(style_key: str) -> list[dict[str, str]]:
    style = CAT_STYLES.get(style_key, {})
    curations = style.get("curations", {})
    defaults = {
        "movie": ("《向左走向右走》", "浪漫淺嘗的節奏，陪你梳理心情"),
        "music": ("Bossa Nova 咖啡廳", "溫柔節拍讓心慢慢沉靜"),
        "activity": ("傍晚散步", "換個場景，讓腦袋短暫放空"),
    }
    mapping = [
        ("movie", "推薦電影"),
        ("music", "推薦音樂"),
        ("activity", "推薦活動"),
    ]
    recommendations = []
    for key, label in mapping:
        title, reason = curations.get(key, defaults[key])
        recommendations.append({"label": label, "title": title, "reason": reason})
    return recommendations

# ?? 1007 修改『圖卡生成推薦』：整合 AI 與預設推薦
def _normalize_recommendations(ai_payload: dict | None, style_key: str) -> list[dict[str, str]]:
    payload = ai_payload or {}
    raw_recs = payload.get("recommendations") or {}
    mapping = [
        ("movie", "推薦電影"),
        ("music", "推薦音樂"),
        ("activity", "推薦活動"),
    ]
    fallback_list = _fallback_recommendations(style_key)
    normalized = []
    for key, label in mapping:
        source = raw_recs.get(key) if isinstance(raw_recs, dict) else None
        title = ""
        reason = ""
        if isinstance(source, dict):
            title = str(source.get("title") or "").strip()
            reason = str(source.get("reason") or "").strip()
        if not title or not reason:
            fallback = next((item for item in fallback_list if item["label"] == label), None)
            if fallback:
                title = title or fallback["title"]
                reason = reason or fallback["reason"]
        normalized.append({"label": label, "title": title, "reason": reason})
    if len(normalized) > 2:
        random.shuffle(normalized)
        normalized = normalized[:2]  # ?? 1007 修改圖卡：僅呈現兩則建議
    return normalized


def build_cat_card(report: dict, psychology: dict):
    """根據健康與心理測驗資料建立貓卡內容。"""  # ?? 0929修改：組裝貓卡資料
    health_score = report.get("health_score")
    mood_score = (
        psychology.get("combined_score")
        or psychology.get("combinedScore")
        or psychology.get("mind_score")
    )
    keywords = psychology.get("keywords") or []

    health_value = float(health_score) if health_score is not None else 72.0
    mood_value = float(mood_score) if mood_score is not None else 68.0

    if health_value >= 80 or mood_value >= 80:
        suggested_style = "bright"
    elif health_value < 60:
        suggested_style = "healer"
    else:
        suggested_style = "steady"

    ai_payload = generate_cat_card_text(report, psychology, suggested_style)
    style_key = (
        ai_payload.get("styleKey")
        if ai_payload and ai_payload.get("styleKey") in CAT_STYLES
        else suggested_style
    )
    style = CAT_STYLES[style_key]

    # Finalize fields with AI payload or defaults
    # ?? 0929修改：先試圖抓對應圖檔，失敗再退回 TheCatAPI
    # ?? 1007 修改『圖卡生成推薦』
    name = (ai_payload or {}).get("name") or random.choice(style["names"])
    persona_key = _resolve_persona_key(health_value, mood_value)
    persona_label = CAT_PERSONA_METADATA.get(persona_key)
    persona = persona_label or (ai_payload or {}).get("persona") or style["title"]
    speech = (ai_payload or {}).get("speech") or random.choice(style["speech"])

    model_keywords = (ai_payload or {}).get("keywords") or keywords
    if isinstance(model_keywords, str):
        model_keywords = [k.strip() for k in model_keywords.split(",") if k.strip()]
    mood_label = "、".join(model_keywords[:3]) if model_keywords else "平衡"

    description_template = style.get("description", "{mood} 的氣息值得被珍惜。")
    try:
        description = (ai_payload or {}).get("summary") or description_template.format(
            mood=mood_label
        )
    except Exception:
        description = (ai_payload or {}).get("summary") or description_template
    insight = (
        (ai_payload or {}).get("insight")
        or psychology.get("summary")
        or f"當下情緒偏向 {mood_label}，記得照顧自己。"
    )
    action = (ai_payload or {}).get("action") or random.choice(style["actions"])

    vitality = max(0, min(100, int(round(health_value))))
    companionship = max(0, min(100, int(round(mood_value))))
    stability = max(
        0, min(100, int((vitality + companionship) / 2 + random.randint(-4, 4)))
    )

    return {
        "persona": persona,
        "name": name,
        "speech": speech,
        "description": description,
        "insight": insight,
        "action": action,
        "stats": [
            {"label": "活力指數", "value": f"{vitality}%"},
            {"label": "陪伴力", "value": f"{companionship}%"},
            {"label": "穩定度", "value": f"{stability}%"},
        ],  # 前端不再顯示分數，但保留結構以利後續調整
        "recommendations": _normalize_recommendations(ai_payload, style_key),  # ?? 1007 修改『圖卡生成推薦』：加入影音與活動建議
        "style_key": style_key,
        "palette": style.get("palette"),
        "keywords_list": model_keywords,
        "persona_key": persona_key,
        "persona_label": persona_label,
    }


def circle_crop_image(image_bytes, diameter: int = 260) -> Image.Image:
    if isinstance(image_bytes, Image.Image):
        img = image_bytes
    else:
        img = Image.open(BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert("RGBA")
    min_side = min(img.size)
    left = (img.width - min_side) // 2
    top = (img.height - min_side) // 2
    img = img.crop((left, top, left + min_side, top + min_side))
    img = img.resize((diameter, diameter), Image.LANCZOS)

    mask = Image.new("L", (diameter, diameter), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, diameter, diameter), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(0.6))

    output = Image.new("RGBA", (diameter, diameter))
    output.paste(img, (0, 0), mask)
    return output

    # ?? 0929修改：繪製圖卡(先試圖抓對應圖檔，失敗再退回 TheCatAPI)


def render_cat_card_image(card: dict, user_id: str, cache_key: str | None = None):
    """生成圖卡 PNG，並回傳檔名與來源 URL。"""  # ?? 0929修改：圖卡繪製
    timeout = 12
    max_bytes = 8_000_000
    image_bytes = None
    source_url = None

    persona_key = card.get("persona_key")
    if persona_key:
        persona_entry = CAT_PERSONA_IMAGES.get(persona_key)
        if persona_entry:
            local_path = persona_entry.get("local_path")
            if local_path and Path(local_path).exists():
                try:
                    image_bytes = Path(local_path).read_bytes()
                    static_path = persona_entry.get("static_path")
                    if static_path:
                        try:
                            source_url = url_for(
                                "static", filename=static_path, _external=True
                            )
                        except RuntimeError:
                            source_url = f"/static/{static_path}"
                except Exception as exc:
                    logging.warning(
                        "Failed to load local persona image %s: %s", local_path, exc
                    )
                    image_bytes = None
            else:
                candidate_url = (
                    persona_entry.get("url")
                    if isinstance(persona_entry, dict)
                    else persona_entry
                )
                if candidate_url:
                    image_bytes, source_url = _download_image(
                        candidate_url, timeout, max_bytes
                    )
                    if not image_bytes:
                        logging.warning(
                            "Persona image download failed for %s", persona_key
                        )

    if not image_bytes:
        image_bytes, source_url = fetch_cat_image(timeout=timeout, max_bytes=max_bytes)

    cat_image = circle_crop_image(image_bytes)

    width, height = 900, 600
    palette = card.get("palette", ("#FFEAA7", "#FD79A8", "#FFAFCC", "#2d3436"))
    bg_start, bg_end, accent, text_color = palette

    base = Image.new("RGB", (width, height), bg_start)
    draw = ImageDraw.Draw(base)

    start_rgb = _hex_to_rgb(bg_start)
    end_rgb = _hex_to_rgb(bg_end)
    for y in range(height):
        ratio = y / max(height - 1, 1)
        r = int(start_rgb[0] * (1 - ratio) + end_rgb[0] * ratio)
        g = int(start_rgb[1] * (1 - ratio) + end_rgb[1] * ratio)
        b = int(start_rgb[2] * (1 - ratio) + end_rgb[2] * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    draw.rounded_rectangle((40, 40, width - 40, height - 40), radius=35, fill="white")

    title_font = _load_font(44)
    name_font = _load_font(56)
    body_font = _load_font(28)
    small_font = _load_font(24)
    caption_font = _load_font(22)

    x_margin = 80
    y = 90
    draw.text(
        (x_margin, y), card.get("persona", "療癒系貓咪"), font=title_font, fill=accent
    )
    y += 70
    draw.text(
        (x_margin, y), card.get("name", "專屬貓咪"), font=name_font, fill=text_color
    )
    y += 80

    speech_text = _wrap_text(card.get("speech"), 14)
    draw.text((x_margin, y), speech_text, font=body_font, fill=text_color)
    y += 110

    summary_text = _wrap_text(card.get("description"), 18)
    draw.text((x_margin, y), summary_text, font=body_font, fill=text_color)
    y += 120

    insight_text = _wrap_text(card.get("insight"), 20)
    if insight_text:
        draw.text(
            (x_margin, y),
            f"心情結論：\n{insight_text}",
            font=small_font,
            fill=text_color,
        )
        y += 120

    for stat in card.get("stats", []):
        draw.text(
            (x_margin, y),
            f"{stat.get('label')}: {stat.get('value')}",
            font=small_font,
            fill=text_color,
        )
        y += 40

    action_text = _wrap_text(card.get("action"), 18)
    if action_text:
        draw.text(
            (x_margin, y), f"建議行動：{action_text}", font=small_font, fill=text_color
        )

    circle_x = width - 320
    circle_y = 120
    highlight_box = (
        circle_x - 20,
        circle_y - 20,
        circle_x + cat_image.width + 20,
        circle_y + cat_image.height + 20,
    )
    draw.ellipse(highlight_box, fill="#fdf6ff")
    base.paste(cat_image, (circle_x, circle_y), cat_image)

    filename = f"catcard_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_hash_for_filename(user_id, str(time.time()), cache_key or '')}.png"
    output_path = CAT_CARD_DIR / filename
    base.save(output_path, format="PNG")
    _cleanup_old_cards()

    return filename, _safe_url(source_url)


# 載入 .env 檔案
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv(
    "FLASK_SECRET_KEY", "your-secret-key"
)  # 從 .env 載入或使用預設值
app.config.update(
    SESSION_COOKIE_SECURE=True, # 只允許 HTTPS
    SESSION_COOKIE_HTTPONLY=True, # JS 不能讀 cookie
    SESSION_COOKIE_SAMESITE="Strict", # 防止跨站請求帶 cookie
)
logging.basicConfig(level=logging.DEBUG)
FIREBASE_WEB_API_KEY = os.getenv('FIREBASE_WEB_API_KEY')
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.before_request(_refresh_daily_points)

# ?? 0929修改：設定圖卡輸出位置與備援資料
BASE_DIR = Path(__file__).resolve().parent
CAT_CARD_DIR = BASE_DIR / "static" / "cat_cards"
CAT_CARD_DIR.mkdir(parents=True, exist_ok=True)

CAT_FALLBACK_IMAGES = [
    "https://images.unsplash.com/photo-1518791841217-8f162f1e1131?auto=format&fit=crop&w=1000&q=80",
    "https://images.unsplash.com/photo-1533738363-b7f9aef128ce?auto=format&fit=crop&w=1000&q=80",
    "https://images.unsplash.com/photo-1504208434309-cb69f4fe52b0?auto=format&fit=crop&w=1000&q=80",
    "https://images.unsplash.com/photo-1583083527882-4bee9aba2eea?auto=format&fit=crop&w=1000&q=80",
]

# 0929修改04：統一設定模型回傳純 JSON
JSON_RESPONSE_CONFIG = genai_types.GenerateContentConfig(
    response_mime_type="application/json",
    candidate_count=1,
    temperature=0.6,
)

REWARD_BADGES = [
    {
        "points": 5,
        "name": "新晉貓奴",
        "description": "獲得貓砂盆與貓砂鏟！",
        "image": "5p.png",
    },
    {
        "points": 20,
        "name": "美食家",
        "description": "獲得元氣罐罐！",
        "image": "20p.png",
    },
    {
        "points": 30,
        "name": "快樂舞者",
        "description": "獲得貓草盆栽！",
        "image": "30p.png",
    },
    {
        "points": 50,
        "name": "活力指揮家",
        "description": "獲得逗貓棒！",
        "image": "50p.png",
    },
    {
        "points": 80,
        "name": "探險家",
        "description": "獲得貓跳台！",
        "image": "80p.png",
    },
]

AVATAR_CHOICES = [
    "profile01.png",
    "profile02.png",
    "profile03.png",
    "profile04.png",
]
DEFAULT_AVATAR = AVATAR_CHOICES[0]

# ?? 0929修改：貓咪九宮格對應既有圖庫
_CAT_LOCAL_IMAGE_DIR = CAT_CARD_DIR / "images" / "cats"
_CAT_LOCAL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

CAT_PERSONA_IMAGES = {
    key: {
        "local_path": _CAT_LOCAL_IMAGE_DIR / f"{key}.png",
        "static_path": f"cat_cards/images/cats/{key}.png",
    }
    for key in ("A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3")
}

CAT_PERSONA_METADATA = {
    "A1": "布偶貓｜心理樂觀?身體指標待加油",
    "A2": "橘貓｜情緒穩定?生活節奏良好",
    "A3": "俄羅斯藍貓｜活力均衡?能量充沛",
    "B1": "波斯貓｜身心提醒?適度調養",
    "B2": "三花貓｜日常波動?持續照顧",
    "B3": "銀漸層貓｜外強內柔?記得舒壓",
    "C1": "折耳貓｜雙重負擔?先好好休息",
    "C2": "黑貓｜心理調適中?需要陪伴",
    "C3": "暹羅貓｜內在壓力大?身體仍有力",
}

FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB W3.ttc",
    "/Library/Fonts/NotoSansCJK-Regular.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
]

# 初始化 Firebase
firebase_credentials_env = os.getenv("FIREBASE_CREDENTIALS")
firebase_storage_bucket = os.getenv(
    "FIREBASE_STORAGE_BUCKET", "gold-chassis-473807-j1.firebasestorage.app"
)

try:
    if firebase_credentials_env:
        try:
            credential_payload = json.loads(firebase_credentials_env)
        except json.JSONDecodeError as exc:
            logging.error(f"FIREBASE_CREDENTIALS contains invalid JSON: {exc}")
            raise ValueError("Invalid FIREBASE_CREDENTIALS JSON payload") from exc

        cred = credentials.Certificate(credential_payload)
        logging.debug("Firebase credentials loaded from environment variable")
    else:
        credential_path = BASE_DIR / "firebase_credentials" / "service_account.json"
        cred = credentials.Certificate(str(credential_path))
        logging.debug(f"Firebase credentials loaded from file: {credential_path}")

    firebase_admin.initialize_app(cred, {"storageBucket": firebase_storage_bucket})
    logging.debug(
        "Firebase initialized successfully with bucket: %s", firebase_storage_bucket
    )
except FileNotFoundError as e:
    logging.error(f"Firebase credential file not found: {e}")
    raise
except ValueError as e:
    logging.error(f"Firebase initialization failed: {e}")
    raise

db = firestore.client()
try:
    bucket = storage.bucket()
    logging.debug(f"Storage bucket initialized: {bucket.name}")
except Exception as e:
    logging.error(f"Storage bucket initialization failed: {str(e)}")
    raise

# Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY not found in .env file")
    raise ValueError("GEMINI_API_KEY is required")

try:
    genai_chat_client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    logging.error(f"Failed to initialise google-genai client: {e}")
    raise

# ?? 修改：啟動時列印路由表（Flask 3 不支援 before_first_request，故保留註解）
# @app.before_first_request
# def _print_url_map():
#    logging.debug("URL Map:\n" + "\n".join([str(r) for r in app.url_map.iter_rules()]))


# 圖片代理：避免跨域限制影響下載圖卡
@app.route("/proxy_image")
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

    response = Response(upstream.content, content_type=content_type or "image/png")
    response.headers["Cache-Control"] = "public, max-age=86400"
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


# 首頁
@app.route("/")
def home():
    is_logged_in = "user_id" in session
    return render_template("home.html", is_logged_in=is_logged_in)


@app.route("/profile", methods=["GET", "POST"])
@app.route("/badges", methods=["GET", "POST"])
def profile():
    user_id = session.get("user_id")
    if not user_id:
        flash("請先登入後再查看我的檔案頁面。", "error")
        return redirect(url_for("login"))

    current_avatar = session.get("avatar") or DEFAULT_AVATAR
    user_email = session.get("user_email")

    if request.method == "POST":
        selected_avatar = request.form.get("avatar")
        if selected_avatar not in AVATAR_CHOICES:
            flash("選擇的頭像無效，請重新選擇。", "error")
        else:
            try:
                db.collection("users").document(user_id).set({"avatar": selected_avatar}, merge=True)
                session["avatar"] = selected_avatar
                current_avatar = selected_avatar
                flash("頭像已更新！", "success")
            except Exception as exc:
                logging.error("Failed to update avatar for %s: %s", user_id, exc)
                flash("頭像更新失敗，請稍後再試。", "error")
        return redirect(url_for("profile"))

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

    next_badge = next((badge for badge in REWARD_BADGES if badge["points"] > user_points), None)
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
        segment_end_label = (
            REWARD_BADGES[-1]["name"] if REWARD_BADGES else "目標"
        )
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


# 註冊
@app.route("/register", methods=["GET", "POST"])
def register():
    is_logged_in = "user_id" in session

    if request.method == "GET":
        session.pop("_flashes", None)

    if request.method == "POST":
        logging.debug("Register submission fields: %s", _form_keys(request.form))
        email = request.form.get("email")
        password = request.form.get("password")
        # ?? 修改開始：新增生理性別欄位
        gender = request.form.get("gender")
        logging.debug(
            "Parsed register data: email=%s, gender=%s",
            _mask_email(email),
            gender,
        )

        if not email or not password or not gender:
            flash("請輸入電子郵件、密碼和生理性別！", "error")
            logging.warning("Missing email, password, or gender in form submission")
            return render_template(
                "register.html",
                error="請輸入電子郵件、密碼和生理性別",
                is_logged_in=is_logged_in,
            )
        # ?? 修改結束
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
            logging.debug(f"User document created in Firestore for uid: {user.uid}")
            session["user_id"] = user.uid
            session["user_email"] = email
            flash("註冊成功！請上傳健康報告。", "success")
            return redirect(url_for("upload_health"))
        except FirebaseError as e:
            error_message = str(e)
            logging.error(f"Firebase error during registration: {error_message}")
            flash(f"註冊失敗：{error_message}", "error")
            return render_template(
                "register.html",
                error=f"註冊失敗：{error_message}",
                is_logged_in=is_logged_in,
            )
        except Exception as e:
            logging.error(f"Unexpected error during registration: {str(e)}")
            flash(f"註冊失敗：{str(e)}", "error")
            return render_template(
                "register.html",
                error=f"註冊失敗：{str(e)}",
                is_logged_in=is_logged_in,
            )

    return render_template("register.html", is_logged_in=is_logged_in)


# 登入
@app.route("/login", methods=["GET", "POST"])
def login():
    is_logged_in = "user_id" in session

    def _localized_login_error(message: str, email: str) -> str:

        """Convert Firebase login error messages or codes to user-friendly Traditional Chinese."""

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

    if request.method == "GET":
        session.pop("_flashes", None)

    if request.method == "POST":
        logging.debug("Login submission fields: %s", _form_keys(request.form))
        email = request.form.get("email")
        password = request.form.get("password")
        logging.debug("Login attempt: email=%s", _mask_email(email))

        if not email or not password:
            flash("請輸入電子郵件和密碼！", "error")
            logging.warning("Missing email or password in login submission")
            return render_template("login.html", is_logged_in=is_logged_in)

        try:
            firebase_uid = _authenticate_with_password(email, password)
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
            points, new_badges = _award_login_points(user_ref)
            logging.debug(
                "User login updated in Firestore for uid: %s, points=%s",
                firebase_uid,
                points,
            )
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
            return redirect(url_for("home"))
        except ValueError as auth_error:
            error_code = str(auth_error)
            logging.warning(f"Login failed: {error_code}")
            flash(_localized_login_error(error_code, email), "error")
            return render_template("login.html", is_logged_in=is_logged_in)
        except RuntimeError as auth_runtime:
            err = str(auth_runtime)
            logging.error(f"Password auth runtime error: {err}")
            flash(_localized_login_error(err, email), "error")
            return render_template("login.html", is_logged_in=is_logged_in)
        except Exception as e:
            logging.error(f"Unexpected login error: {str(e)}")
            flash("登入失敗：系統目前忙碌，請稍後再試。", "error")
            return render_template("login.html", is_logged_in=is_logged_in)

    return render_template("login.html", is_logged_in=is_logged_in)


@app.route("/delete_account", methods=["GET", "POST"])
def delete_account():
    if "user_id" not in session:
        flash("請先登入。", "error")
        return redirect(url_for("login"))

    user_id = session["user_id"]
    user_ref = db.collection("users").document(user_id)
    try:
        user_doc = db.collection("users").document(user_id).get()
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
                "delete_account.html",
                user_email=user_email,
                is_logged_in=True,
            )
        if not confirm_delete:
            flash("請勾選確認刪除，刪除後無法復原。", "error")
            return render_template(
                "delete_account.html",
                user_email=user_email,
                is_logged_in=True,
            )
        if not user_email:
            flash("無法確認此帳號的電子郵件，請聯絡客服。", "error")
            return render_template(
                "delete_account.html",
                user_email=user_email,
                is_logged_in=True,
            )

        try:
            _authenticate_with_password(user_email, password)
        except ValueError as auth_error:
            flash(_localized_login_error(str(auth_error), user_email), "error")
            return render_template(
                "delete_account.html",
                user_email=user_email,
                is_logged_in=True,
            )
        except RuntimeError as runtime_error:
            flash(_localized_login_error(str(runtime_error), user_email), "error")
            return render_template(
                "delete_account.html",
                user_email=user_email,
                is_logged_in=True,
            )

        _delete_user_psychology_tests(user_id)
        _delete_user_health_reports(user_id)
        _delete_user_storage_files(user_id)

        try:
            db.collection("users").document(user_id).delete()
        except Exception as exc:
            logging.warning("Failed to delete user document %s: %s", user_id, exc)

        try:
            auth.delete_user(user_id)
        except FirebaseError as exc:
            logging.warning("Failed to delete auth user %s: %s", user_id, exc)

        session.clear()
        flash("帳戶已刪除。", "success")
        return redirect(url_for("home"))

    return render_template(
        "delete_account.html", user_email=user_email, is_logged_in=True
    )


# 登出
@app.route("/logout")
def logout():
    session.pop("user_id", None)
    session.pop("_flashes", None)
    flash("已成功登出！", "success")
    return redirect(url_for("home"))


# 九宮格貓咪頁面
@app.route("/featured_cats")
def featured_cats():
    is_logged_in = "user_id" in session
    return render_template("featured_cats.html", is_logged_in=is_logged_in)


# 上傳健康報告
@app.route("/upload_health", methods=["GET", "POST"])
def upload_health():
    if "user_id" not in session:
        flash("請先登錄！", "error")
        return redirect(url_for("login"))

    user_id = session["user_id"]
    logging.debug(f"Current user_id from session: {user_id}")

    # ?? 修改開始：取得使用者生理性別
    user_gender = None
    try:
        user_doc = db.collection("users").document(user_id).get()
        if not user_doc.exists:
            flash("找不到使用者資料！", "error")
            logging.warning(f"User document not found for uid: {user_id}")
            return redirect(url_for("register"))
        user_data = user_doc.to_dict()
        user_gender = user_data.get("gender")
        if not user_gender:
            flash("請先完成註冊並提供生理性別資料！", "error")
            logging.warning(f"User gender missing for uid: {user_id}")
            return redirect(url_for("register"))
        logging.debug(f"Retrieved user gender from Firestore: {user_gender}")
    except Exception as e:
        logging.error(f"Failed to retrieve user gender: {str(e)}")
        flash(f"取得使用者資料失敗：{str(e)}", "error")
        return redirect(url_for("login"))
    # ?? 修改結束

    # ?? 修改開始：已有健檢報告時自動導向心理測驗
    reupload_requested = request.args.get("reupload") == "1"
    try:
        existing_reports = list(
            db.collection("health_reports")
            .where("user_uid", "==", user_id)
            .limit(1)
            .stream()
        )
    except Exception as e:
        logging.error(f"Failed to check existing health reports: {str(e)}")
        existing_reports = []

    has_existing_report = bool(existing_reports)

    auto_redirect = False
    invalid_report_prompt = session.pop('invalid_report_prompt', False)
    if invalid_report_prompt:
        auto_redirect = False
    elif request.method == "GET" and has_existing_report and not reupload_requested:
        auto_redirect = True
    # ?? 修改結束

    if request.method == "POST":
        if "health_report" not in request.files:
            flash("未選擇檔案！", "error")
            return redirect(url_for("upload_health"))

        file = request.files["health_report"]
        if file.filename == "":
            flash("未選擇檔案！", "error")
            return redirect(url_for("upload_health"))

        logging.debug(
            "Upload request fields=%s, files=%s",
            _form_keys(request.form),
            list(request.files.keys()),
        )

        filename_lower = file.filename.lower()
        if not filename_lower.endswith(ALLOWED_UPLOAD_EXTENSIONS):
            flash("請上傳 JPEG、PNG 或 PDF 檔！", "error")
            return redirect(url_for("upload_health"))

        mimetype = (file.mimetype or "").lower()
        if mimetype not in ALLOWED_UPLOAD_MIMES:
            flash("請上傳 JPEG、PNG 或 PDF 檔！", "error")
            return redirect(url_for("upload_health"))

        content_length = request.content_length
        if content_length and content_length > MAX_UPLOAD_BYTES:
            flash("檔案超過 10MB，請重新上傳！", "error")
            return redirect(url_for("upload_health"))

        is_image = mimetype in {"image/jpeg", "image/png"}

        # 11/12???????????????????
        logging.debug("Starting health report analysis...")
        recognized_metric_count = 0
        try:
            file.seek(0)  # ??????
            # 11/12??????????????????????
            file_data = file.read()
            if len(file_data) > MAX_UPLOAD_BYTES:
                flash("檔案超過 10MB，請重新上傳！", "error")
                return redirect(url_for("upload_health"))
            file_type = "image" if is_image else "pdf"
            analysis_data, health_score, health_warnings, recognized_metric_count = analyze_health_report(
                file_data, user_id, file_type, gender=user_gender  # ?? ?????????????
            )
            logging.debug(
                f"Analysis result - data: {analysis_data is not None}, score: {health_score}, warnings: {len(health_warnings)}, matched_metrics: {recognized_metric_count}"
            )
            # GALING ?????? 10/5
            if recognized_metric_count <= 0 or not analysis_data:
                logging.warning("No recognizable health metrics were extracted; prompting re-upload")
                session['invalid_report_prompt'] = True
                flash("Sorry??????????????", "invalid_report")
                return redirect(url_for('upload_health'))
        except Exception as analysis_e:
            logging.error(f"Health report analysis failed: {str(analysis_e)}")
            flash(f"健康報告分析失敗：{str(analysis_e)}", "warning")
            analysis_data, health_score, health_warnings, recognized_metric_count = None, 0, [], 0

        # 準備 Firestore 文檔
        # 11/12?????????Firestore ???????
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
                f"Adding analysis data to doc: score={health_score}, warnings={health_warnings}"
            )

        # 儲存到 Firestore
        doc_ref = db.collection("health_reports").document()
        doc_ref.set(health_report_doc)
        report_id = doc_ref.id
        logging.debug(
            f"Health report SAVED to Firestore for user: {user_id}, report_id: {report_id}"
        )
        logging.debug(
            "Health report saved for user %s with score=%s and %d warnings",
            user_id,
            health_score,
            len(health_warnings),
        )

        # 驗證寫入
        saved_doc = db.collection("health_reports").document(report_id).get()
        if saved_doc.exists:
            logging.debug(
                f"Firestore write verified - document exists: {saved_doc.to_dict()}"
            )
        else:
            logging.error("Firestore write failed - document does not exist")

        flash(
            f"上傳成功！健康分數：{health_score}，警告：{'; '.join(health_warnings) if health_warnings else '無'}",
            "success",
        )
        return redirect(url_for("psychology_test"))

    return render_template(
        "upload_health.html",
        force_reupload=reupload_requested,
        has_existing_report=has_existing_report,
        auto_redirect=auto_redirect,
        invalid_report_prompt=invalid_report_prompt,
        psychology_url=url_for("psychology_test"),
    )


# 心理測驗
@app.route(
    "/psychology_test", methods=["GET", "POST"]
)  # ?? 修改：允許 POST 以處理心理測驗提交
def psychology_test():
    if "user_id" not in session:
        flash("請先登入！", "error")
        return redirect(url_for("login"))

    user_id = session["user_id"]
    try:
        # ?? 修改：改為查詢頂層 health_reports 並依 user_uid 過濾，避免找不到文件
        health_reports = list(
            db.collection("health_reports").where("user_uid", "==", user_id).stream()
        )  # ?? 修改：原本是 users/{uid}/health_reports
        logging.debug(
            f"Psychology test check - existing reports: {len(health_reports)}"
        )
        if not health_reports:
            flash("請先上傳健康報告！", "error")
            return redirect(url_for("upload_health"))
    except Exception as e:
        logging.error(f"Error checking health reports: {str(e)}")
        flash(f"檢查健康報告失敗：{str(e)}", "error")
        return redirect(url_for("upload_health"))

    # ?? 修改開始：支援心理測驗表單提交流程
    if request.method == "GET":
        session.pop("_flashes", None)

        latest_report_data = None
        try:

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
        except Exception as e:
            logging.warning(f"Failed to prepare latest health report for template: {e}")
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
            "psychology_test.html", error=OVER_LIMIT_MESSAGE, is_logged_in=True
        )

    try:
        db.collection("users").document(user_id).collection("psychology_tests").add(
            {
                "question1": question1,
                "question2": question2,
                "submit_time": SERVER_TIMESTAMP,
            }
        )
        logging.debug(f"Psychology test saved to Firestore for uid: {user_id}")
        flash("表單已送出！歡迎前往生成貓咪圖卡。", "success")
        return redirect(url_for("generate_card"))
    except Exception as e:
        logging.error(f"Psychology test error: {str(e)}")
        flash(f"提交失敗：{str(e)}", "error")
        return render_template(
            "psychology_test.html", error=f"提交失敗：{str(e)}", is_logged_in=True
        )
    # ?? 修改結束


# 聊天 API 端點（代理 Gemini API）
@app.route("/chat_api", methods=["POST"])
def chat_api():
    if "user_id" not in session:
        return jsonify({"error": "未登入"}), 401

    data = request.get_json()
    if not data or "conversationHistory" not in data or "systemInstruction" not in data:
        logging.error("Invalid request payload for chat_api. Keys=%s", list(data.keys()) if isinstance(data, dict) else data)
        return jsonify({"error": "缺少必要的參數"}), 400

    try:
        conversation_history = data.get("conversationHistory")
        if not isinstance(conversation_history, list):
            return jsonify({"error": "conversationHistory 格式不正確"}), 400
        if not _user_messages_within_limit(conversation_history):
            return jsonify({"error": OVER_LIMIT_MESSAGE}), 400
        logging.debug(
            "Received conversation history entries: %d", len(conversation_history)
        )

        contents = _build_genai_contents(
            data.get("systemInstruction"), conversation_history
        )

        if not contents:
            return jsonify({"error": "conversationHistory 為空或格式無效"}), 400

        try:
            response = _generate_with_retry(
                contents, generation_config=JSON_RESPONSE_CONFIG
            )
        except Exception as e:
            logging.error(f"Gemini generation failed: {e}")
            return jsonify({"nextPrompt": "AI 助手暫時無法回應，請稍後再試。"}), 200

        if not response or not getattr(response, "candidates", None):
            logging.error("Gemini API returned no candidates")
            return jsonify({"nextPrompt": "無法取得回應，請稍後再試。"}), 200

        candidate = response.candidates[0]
        reply = ""
        parts = (
            getattr(candidate.content, "parts", None) or []
        )  # 0929修改03：parts 可能為 None，改採空清單避免迴圈錯誤
        for part in parts:
            if getattr(part, "text", None):
                reply += part.text

        if not reply:
            logging.error("Gemini candidate did not include textual content")
            return jsonify({"nextPrompt": "無法取得回應，請稍後再試。"}), 200

        logging.debug("Raw reply length: %s characters", len(reply))

        try:
            parsed_json = extract_json_from_response(reply)
            logging.debug(f"Successfully parsed JSON: {parsed_json}")
        except Exception:
            logging.exception(
                "0929修改03：chat_api JSON parse failed; raw snippet=%r", reply[:500]
            )
            parsed_json = None

        if parsed_json and isinstance(parsed_json, dict):
            if "nextPrompt" in parsed_json or "summary" in parsed_json:
                return jsonify(parsed_json)
            return jsonify({"nextPrompt": reply})

        logging.warning(
            f"Could not parse JSON from reply, returning as plain text: {reply}"
        )
        return jsonify({"nextPrompt": reply})

    except Exception as e:
        logging.error(f"Unexpected error in chat_api: {str(e)}, data: {data}")
        return jsonify({"error": f"伺服器錯誤：{str(e)}"}), 500


# 報告 API 端點（代理 Gemini API）
@app.route("/report_api", methods=["POST"])
def report_api():
    if "user_id" not in session:
        return jsonify({"error": "未登入"}), 401

    data = request.get_json()
    if not data or "conversationHistory" not in data or "systemInstruction" not in data:
        logging.error(f"Invalid request data: {data}")
        return jsonify({"error": "缺少必要的參數"}), 400

    try:
        conversation_history = data.get("conversationHistory")
        if not isinstance(conversation_history, list):
            return jsonify({"error": "conversationHistory 格式不正確"}), 400
        if not _user_messages_within_limit(conversation_history):
            return jsonify({"error": OVER_LIMIT_MESSAGE}), 400
        logging.debug(
            f"Received conversationHistory for report: {len(conversation_history)} messages"
        )

        contents = _build_genai_contents(
            data.get("systemInstruction"), conversation_history
        )

        if not contents:
            return jsonify({"error": "conversationHistory 為空或格式無效"}), 400

        try:
            response = _generate_with_retry(
                contents, generation_config=JSON_RESPONSE_CONFIG
            )
        except Exception as e:
            logging.error(f"Gemini report generation failed: {e}")
            return (
                jsonify(
                    {
                        "summary": "模型沒有產生報告內容，請稍後再試。",
                        "keywords": [],
                        "emotionVector": {
                            "valence": 50,
                            "arousal": 50,
                            "dominance": 50,
                        },
                    }
                ),
                200,
            )

        if not response or not getattr(response, "candidates", None):
            logging.warning("Gemini report: no candidates, fallback to empty summary")
            report_json = {
                "summary": "模型沒有產生報告內容，請稍後再試。",
                "keywords": [],
                "emotionVector": {"valence": 50, "arousal": 50, "dominance": 50},
            }
            return jsonify(report_json), 200

        candidate = response.candidates[0]
        summary_text = ""
        parts = (
            getattr(candidate.content, "parts", None) or []
        )  # 0929修改03：parts 可能為 None，改採空清單避免迴圈錯誤
        for part in parts:
            if getattr(part, "text", None):
                summary_text += part.text

        if not summary_text:
            logging.warning("Gemini report: candidate present but empty text")
            report_json = {
                "summary": "模型沒有提供完整內容。",
                "keywords": [],
                "emotionVector": {"valence": 50, "arousal": 50, "dominance": 50},
            }
            return jsonify(report_json), 200

        logging.debug("Raw report summary length: %s", len(summary_text))

        try:
            parsed_json = extract_json_from_response(summary_text)
            parsed_json = _validate_report_schema(parsed_json)
            logging.debug(f"Successfully parsed report JSON: {parsed_json}")
            return jsonify(parsed_json)
        except Exception as exc:
            logging.exception("0929修改03：report_api JSON/schema failed: %s", exc)
            return (
                jsonify(
                    {
                        "error": "LLM returned invalid JSON",
                        "detail": str(exc),
                        "raw": summary_text[:500],
                    }
                ),
                502,
            )

    except Exception as e:
        logging.error(f"Unexpected error in report_api: {str(e)}, data: {data}")
        return jsonify({"error": f"伺服器錯誤：{str(e)}"}), 500


# 儲存心理測驗分數
# ?? 修改：明確指定 endpoint 名稱，避免因函式名或載入順序造成的註冊差異
@app.route(
    "/save_psychology_scores", methods=["POST"], endpoint="save_psychology_scores"
)  # ?? 修改
def save_psychology_scores():
    if "user_id" not in session:
        return jsonify({"error": "未登入"}), 401

    data = request.get_json()
    if not data or not all(
        key in data for key in ["mindScore", "bodyScore", "combinedScore"]
    ):
        return jsonify({"error": "缺少必要的分數參數"}), 400

    try:
        user_id = session["user_id"]
        test_id = (
            db.collection("users")
            .document(user_id)
            .collection("psychology_tests")
            .document()
            .id
        )
        db.collection("users").document(user_id).collection(
            "psychology_tests"
        ).document(test_id).set(
            {
                "mind_score": data["mindScore"],
                "body_score": data["bodyScore"],
                "combined_score": data["combinedScore"],
                "summary": data.get("summary", ""),
                "keywords": data.get("keywords", []),
                "emotion_vector": data.get("emotionVector", {}),
                # 11/12只存數據不存對話：不在 Firestore 留下聊天內容
                "conversation_history": [],
                "submit_time": SERVER_TIMESTAMP,
            }
        )
        logging.debug(f"Psychology scores saved for user {user_id}, test {test_id}")
        return jsonify({"status": "success", "test_id": test_id})
    except Exception as e:
        logging.error(f"Error saving psychology scores: {str(e)}")
        return jsonify({"error": f"儲存分數失敗：{str(e)}"}), 500


# 生成貓咪圖卡
@app.route("/generate_card")
def generate_card():
    if "user_id" not in session:
        flash("請先登入！", "error")
        return redirect(url_for("login"))

    session.pop("_flashes", None)

    try:
        user_id = session["user_id"]
        user_ref = db.collection("users").document(user_id)
        try:
            user_doc_data = user_ref.get().to_dict() or {}
        except Exception as exc:
            logging.warning("Failed to load user doc for card limit: %s", exc)
            user_doc_data = {}
        allowed, card_count, today_str = _evaluate_daily_limit(
            user_doc_data,
            "daily_card_generations_count",
            "last_card_generation_date",
            MAX_DAILY_CARD_GENERATIONS,
        )
        if not allowed:
            flash(CARD_LIMIT_MESSAGE, "error")
            return redirect(url_for("home"))
        # ?? 修改：同樣改為查詢頂層 health_reports
        health_report_docs = (
            db.collection("health_reports").where("user_uid", "==", user_id).stream()
        )
        reports = []
        for doc in health_report_docs:
            data = doc.to_dict() or {}
            data["id"] = doc.id
            reports.append(data)
        logging.debug(f"Generate card - reports found: {len(reports)}")
        if not reports:
            flash("請先上傳健康報告！", "error")
            return redirect(url_for("upload_health"))

        psych_docs = (
            db.collection("users")
            .document(user_id)
            .collection("psychology_tests")
            .stream()
        )
        tests = []
        for doc in psych_docs:
            data = doc.to_dict() or {}
            data["id"] = doc.id
            tests.append(data)
        if not tests:
            flash("請先完成心理測驗！", "error")  # ?? 0929修改：修正提示字串
            return redirect(url_for("psychology_test"))

        sorted_reports = sorted(
            reports,
            key=lambda r: _to_datetime(r.get("created_at") or r.get("report_date")),
        )
        latest_report = sorted_reports[-1]
        warnings, vitals_display = _normalize_health_data(
            latest_report
        )  # ?? 0929修改：整理健檢提醒與指標
        latest_report["_display_warnings"] = warnings
        latest_report["_display_vitals"] = vitals_display
        health_tips = _build_health_tips(latest_report, warnings)
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
            cache_key_match = (
                cache_entry.get("cache_key") == cache_key_current
            )  # ?? 1001修改01：僅當最新報告/測驗與快取一致時才沿用
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
                "cache_key": cache_key,  # ?? 1001修改01：記錄本次使用的報告/測驗組合避免取用過期圖卡
            }

        card_image_url = url_for("static", filename=f"cat_cards/{image_filename}")
        card_payload["image_url"] = card_image_url
        card_payload["cat_image_source"] = cat_source
        card_payload.setdefault("warnings", warnings)

        try:
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
        )
    except Exception as e:
        logging.error(f"Generate card error: {str(e)}")
        flash(f"生成圖卡失敗：{str(e)}", "error")
        return render_template(
            "generate_card.html", error=f"生成圖卡失敗：{str(e)}", is_logged_in=True
        )


@app.context_processor
def inject_badge_context():
    """Provide login badge state to templates for dropdown + modal rendering."""
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
            logging.debug(f"Failed to refresh user info for %s: %s", user_id, exc)
            if user_points is None:
                user_points = 0
            if user_avatar is None:
                user_avatar = DEFAULT_AVATAR

    user_points = int(user_points or 0)
    user_avatar = user_avatar or DEFAULT_AVATAR
    unlocked_points = [badge["points"] for badge in REWARD_BADGES if user_points >= badge["points"]]
    session["badge_thresholds"] = unlocked_points

    reward_unlocks = session.pop("reward_unlocks", None) if "reward_unlocks" in session else []

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
    }


if __name__ == "__main__":
    # 若要列出路由可在此紀錄
    debug_enabled = str(os.getenv("FLASK_DEBUG", "0")).lower() in {"1", "true", "yes", "on"}
    port = int(os.getenv("FLASK_PORT", "5001"))
    app.run(debug=debug_enabled, port=port)

