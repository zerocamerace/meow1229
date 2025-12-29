from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request, session

from config.settings import MAX_USER_TEXT_CHARS, OVER_LIMIT_MESSAGE
from google.cloud.firestore import SERVER_TIMESTAMP
from services.genai import (
    JSON_RESPONSE_CONFIG,
    build_genai_contents,
    extract_json_from_response,
    generate_with_retry,
)
from services.health import validate_report_schema
from services.firebase import db
from utils.security import _mask_uid, _user_messages_within_limit

api_bp = Blueprint("api", __name__)


@api_bp.route("/chat_api", methods=["POST"])
def chat_api():
    if "user_id" not in session:
        return jsonify({"error": "未登入"}), 401

    data = request.get_json()
    if not data or "conversationHistory" not in data or "systemInstruction" not in data:
        logging.error(
            "Invalid request payload for chat_api. Keys=%s",
            list(data.keys()) if isinstance(data, dict) else data,
        )
        return jsonify({"error": "缺少必要的參數"}), 400

    try:
        conversation_history = data.get("conversationHistory")
        if not isinstance(conversation_history, list):
            return jsonify({"error": "conversationHistory 格式不正確"}), 400
        if not _user_messages_within_limit(conversation_history, MAX_USER_TEXT_CHARS):
            return jsonify({"error": OVER_LIMIT_MESSAGE}), 400

        logging.debug(
            "Received conversation history entries: %d", len(conversation_history)
        )

        contents = build_genai_contents(
            data.get("systemInstruction"), conversation_history
        )

        if not contents:
            return jsonify({"error": "conversationHistory 為空或格式無效"}), 400

        try:
            response = generate_with_retry(
                contents, generation_config=JSON_RESPONSE_CONFIG
            )
        except Exception as exc:
            logging.error("Gemini generation failed: %s", exc)
            return jsonify(
                {"nextPrompt": "AI 助手暫時無法回應，請稍後再試。"}
            ), 200

        if not response or not getattr(response, "candidates", None):
            logging.error("Gemini API returned no candidates")
            return jsonify({"nextPrompt": "無法取得回應，請稍後再試。"}), 200

        candidate = response.candidates[0]
        reply = ""
        parts = getattr(candidate.content, "parts", None) or []
        for part in parts:
            if getattr(part, "text", None):
                reply += part.text

        if not reply:
            logging.error("Gemini candidate did not include textual content")
            return jsonify({"nextPrompt": "無法取得回應，請稍後再試。"}), 200

        logging.debug("Raw reply length: %s characters", len(reply))

        try:
            parsed_json = extract_json_from_response(reply)
            logging.debug("Successfully parsed JSON: %s", parsed_json)
        except Exception:
            logging.exception("chat_api JSON parse failed; raw snippet=%r", reply[:500])
            parsed_json = None

        if parsed_json and isinstance(parsed_json, dict):
            if "nextPrompt" in parsed_json or "summary" in parsed_json:
                return jsonify(parsed_json)
            return jsonify({"nextPrompt": reply})

        logging.warning(
            "Could not parse JSON from reply, returning as plain text: %s", reply
        )
        return jsonify({"nextPrompt": reply})

    except Exception as exc:
        logging.error(
            "Unexpected error in chat_api for user %s: %s",
            _mask_uid(session.get("user_id")),
            exc,
        )
        return jsonify({"error": "喵奶奶暫時有點忙碌，請稍後再試。"}), 500


@api_bp.route("/report_api", methods=["POST"])
def report_api():
    if "user_id" not in session:
        return jsonify({"error": "未登入"}), 401

    data = request.get_json()
    if not data or "conversationHistory" not in data or "systemInstruction" not in data:
        logging.error(
            "Invalid report request payload; keys=%s",
            list(data.keys()) if isinstance(data, dict) else type(data).__name__,
        )
        return jsonify({"error": "缺少必要的參數"}), 400

    try:
        conversation_history = data.get("conversationHistory")
        if not isinstance(conversation_history, list):
            return jsonify({"error": "conversationHistory 格式不正確"}), 400
        if not _user_messages_within_limit(conversation_history, MAX_USER_TEXT_CHARS):
            return jsonify({"error": OVER_LIMIT_MESSAGE}), 400

        logging.debug(
            "Received conversationHistory for report: %d messages",
            len(conversation_history),
        )

        contents = build_genai_contents(
            data.get("systemInstruction"), conversation_history
        )

        if not contents:
            return jsonify({"error": "conversationHistory 為空或格式無效"}), 400

        try:
            response = generate_with_retry(
                contents, generation_config=JSON_RESPONSE_CONFIG
            )
        except Exception as exc:
            logging.error("Gemini report generation failed: %s", exc)
            fallback = {
                "summary": "模型沒有產生報告內容，請稍後再試。",
                "keywords": [],
                "emotionVector": {
                    "valence": 50,
                    "arousal": 50,
                    "dominance": 50,
                },
            }
            return jsonify(fallback), 200

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
        parts = getattr(candidate.content, "parts", None) or []
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
            parsed_json = validate_report_schema(parsed_json)
            logging.debug("Successfully parsed report JSON: %s", parsed_json)
            return jsonify(parsed_json)
        except Exception:
            logging.exception("report_api JSON/schema failed")
            return jsonify({"error": "LLM returned invalid JSON"}), 502

    except Exception:
        logging.exception(
            "Unexpected error in report_api for user %s",
            _mask_uid(session.get("user_id")),
        )
        return jsonify({"error": "伺服器暫時忙碌，請稍後再試"}), 500


@api_bp.route("/save_psychology_scores", methods=["POST"])
def save_psychology_scores():
    if "user_id" not in session:
        return jsonify({"error": "未登入"}), 401

    data = request.get_json()
    if not data or not all(key in data for key in ["mindScore", "bodyScore", "combinedScore"]):
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
                "conversation_history": [],
                "submit_time": SERVER_TIMESTAMP,
            }
        )
        logging.debug(
            "Psychology scores saved for user %s, test %s",
            _mask_uid(user_id),
            test_id,
        )
        return jsonify({"status": "success", "test_id": test_id})
    except Exception as exc:
        logging.error(
            "Error saving psychology scores for %s: %s",
            _mask_uid(session.get("user_id")),
            exc,
        )
        return jsonify({"error": "儲存心理測驗結果時發生錯誤，請稍後再試。"}), 500
