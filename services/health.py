from __future__ import annotations

from datetime import datetime

HEALTH_RECOMMENDATIONS = [
    ("膽固醇", "減少油炸與加工食品，增加可溶性纖維與omega-3脂肪酸攝取。"),
    ("血糖", "控制精緻糖攝取，注意三餐定時並搭配適量運動。"),
    ("血壓", "減少鈉攝取，保持作息與壓力管理，維持充足睡眠。"),
    ("體重", "規律運動並調整飲食份量，朝向健康體重範圍。"),
    ("肝", "減少酒精與高脂飲食，必要時尋求醫師評估。"),
]


def normalize_health_data(report: dict):
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


def build_health_tips(report: dict, warnings: list[str], limit: int = 3):
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
                        "recommendation": _get_recommendation(label)
                        or "保持規律運動、均衡飲食與充足睡眠。",
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


def score_to_interval(score) -> int | None:
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


def resolve_persona_key(health_score, mind_score) -> str | None:
    physical_zone = score_to_interval(health_score)
    mental_zone = score_to_interval(mind_score)
    if not physical_zone or not mental_zone:
        return None
    prefix = {1: "C", 2: "B", 3: "A"}.get(mental_zone)
    if not prefix:
        return None
    return f"{prefix}{physical_zone}"


def validate_report_schema(payload: dict) -> dict:
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
