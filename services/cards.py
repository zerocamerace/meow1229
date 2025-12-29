from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import textwrap
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import imghdr
import requests
from flask import url_for
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from config.settings import CAT_CARD_DIR, RECOMMENDATION_LABELS
from services.genai import (
    JSON_RESPONSE_CONFIG,
    build_genai_contents,
    extract_json_from_response,
    generate_with_retry,
)
from services.health import resolve_persona_key
from services.movies import (
    DEFAULT_SCENARIOS,
    KEYWORD_SYNONYMS,
    STYLE_HINT_TAGS,
    rag_movie_recommendations,
)
from utils.security import _safe_url

CAT_CARD_DIR.mkdir(parents=True, exist_ok=True)

CAT_FALLBACK_IMAGES = [
    "https://images.unsplash.com/photo-1518791841217-8f162f1e1131?auto=format&fit=crop&w=1000&q=80",
    "https://images.unsplash.com/photo-1533738363-b7f9aef128ce?auto=format&fit=crop&w=1000&q=80",
    "https://images.unsplash.com/photo-1504208434309-cb69f4fe52b0?auto=format&fit=crop&w=1000&q=80",
    "https://images.unsplash.com/photo-1583083527882-4bee9aba2eea?auto=format&fit=crop&w=1000&q=80",
]

FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB W3.ttc",
    "/Library/Fonts/NotoSansCJK-Regular.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
]

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
            "movie": (
                "《翻滾吧！阿信》",
                "熱血卻富含體貼的勵志故事，帶來向上的動力",
            ),
            "music": ("City Pop 暖陽歌單", "輕快律動喚醒身體的節奏感"),
            "activity": ("戶外晨間伸展", "在陽光下活動筋骨，吸收自然能量"),
        },
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
    "A1": "布偶貓｜心理樂觀、一起變健康",
    "A2": "橘貓｜情緒穩定、生活節奏良好",
    "A3": "俄羅斯藍貓｜活力均衡、能量充沛",
    "B1": "波斯貓｜身心提醒、適度調養",
    "B2": "三花貓｜日常波動、持續照顧",
    "B3": "銀漸層貓｜外強內柔、記得舒壓",
    "C1": "折耳貓｜雙重負擔、先好好休息",
    "C2": "黑貓｜心理調適中、需要陪伴",
    "C3": "暹羅貓｜內在壓力大、身體仍有力",
}


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
    return "\n".join(
        textwrap.wrap(collapsed, width=max_chars, break_long_words=True)
    )


def _hex_to_rgb(hex_value: str) -> tuple[int, int, int]:
    hex_value = hex_value.lstrip("#")
    return tuple(int(hex_value[i : i + 2], 16) for i in (0, 2, 4))


def _hash_for_filename(*parts: str) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(part.encode("utf-8", errors="ignore"))
    return hasher.hexdigest()[:8]


def _cleanup_old_cards(max_files: int = 40):
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


def fetch_cat_image(
    max_retries: int = 3, timeout: int = 12, max_bytes: int = 8_000_000
):
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
                image_bytes, final_url = _download_image(
                    img_url, timeout, max_bytes
                )
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

    contents = build_genai_contents(prompt, [])
    try:
        response = generate_with_retry(
            contents, generation_config=JSON_RESPONSE_CONFIG
        )
        if not response or not getattr(response, "candidates", None):
            return None
        candidate = response.candidates[0]
        text = ""
        parts = getattr(candidate.content, "parts", None) or []
        for part in parts:
            if getattr(part, "text", None):
                text += part.text
        try:
            parsed = extract_json_from_response(text)
        except Exception:
            logging.exception(
                "Cat card JSON parse failed; raw snippet=%r", text[:500]
            )
            parsed = None
        if isinstance(parsed, dict):
            return parsed
        logging.warning("Cat card text fallback due to unparsable response")
    except Exception as exc:
        logging.error("generate_cat_card_text failed: %s", exc)
    return None


def _fallback_recommendations(style_key: str) -> dict[str, dict[str, str]]:
    style = CAT_STYLES.get(style_key, {})
    curations = style.get("curations", {})
    defaults = {
        "movie": (
            "《翻滾吧！阿信》",
            "熱血卻富含體貼的勵志故事，帶來向上的動力",
        ),
        "music": ("Bossa Nova 輕爵士", "柔軟又帶點陽光氣息，讓心情慢慢舒展"),
        "activity": ("療癒手作時光", "用雙手專心創作，讓注意力回到當下"),
    }
    recommendations = {}
    for key in ("movie", "music", "activity"):
        title, reason = curations.get(key, defaults[key])
        recommendations[key] = {
            "label": RECOMMENDATION_LABELS[key],
            "title": title,
            "reason": reason,
        }
    return recommendations


def _normalize_recommendations(
    ai_payload: dict | None, style_key: str, psychology: dict | None
) -> list[dict[str, str]]:
    payload = ai_payload or {}
    raw_recs = payload.get("recommendations") or {}
    fallback_map = _fallback_recommendations(style_key)
    movie_candidates = rag_movie_recommendations(psychology, style_key, top_n=5)
    normalized = []
    for key in ("movie", "music", "activity"):
        label = RECOMMENDATION_LABELS[key]
        title = ""
        reason = ""
        if key == "movie" and movie_candidates:
            candidate = movie_candidates.pop(0)
            title = candidate.get("title", "").strip()
            reason = candidate.get("reason", "").strip()
        else:
            source = raw_recs.get(key) if isinstance(raw_recs, dict) else None
            if isinstance(source, dict):
                title = str(source.get("title") or "").strip()
                reason = str(source.get("reason") or "").strip()
        if not title or not reason:
            fallback = fallback_map.get(key)
            if fallback:
                title = title or fallback["title"]
                reason = reason or fallback["reason"]
        normalized.append({"label": label, "title": title, "reason": reason})
    movie_entry = next(
        (
            item
            for item in normalized
            if item["label"] == RECOMMENDATION_LABELS["movie"]
            and item["title"]
        ),
        None,
    )
    others = [item for item in normalized if item is not movie_entry and item["title"]]
    random.shuffle(others)
    ordered = []
    if movie_entry:
        ordered.append(movie_entry)
    ordered.extend(others)
    if not ordered:
        ordered = list(fallback_map.values())
    return ordered[:2] if len(ordered) > 2 else ordered


def build_cat_card(report: dict, psychology: dict):
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

    name = (ai_payload or {}).get("name") or random.choice(style["names"])
    persona_key = resolve_persona_key(health_value, mood_value)
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
        ],
        "recommendations": _normalize_recommendations(
            ai_payload, style_key, psychology
        ),
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


def render_cat_card_image(card: dict, user_id: str, cache_key: str | None = None):
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
                candidate_url = persona_entry.get("url")
                if candidate_url:
                    image_bytes, source_url = _download_image(
                        candidate_url, timeout, max_bytes
                    )
                    if not image_bytes:
                        logging.warning(
                            "Persona image download failed for %s", persona_key
                        )

    if not image_bytes:
        image_bytes, source_url = fetch_cat_image(
            timeout=timeout, max_bytes=max_bytes
        )

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
    y_pos = 90
    draw.text(
        (x_margin, y_pos), card.get("persona", "療癒系貓咪"), font=title_font, fill=accent
    )
    y_pos += 70
    draw.text(
        (x_margin, y_pos), card.get("name", "專屬貓咪"), font=name_font, fill=text_color
    )
    y_pos += 80

    speech_text = _wrap_text(card.get("speech"), 14)
    draw.text((x_margin, y_pos), speech_text, font=body_font, fill=text_color)
    y_pos += 110

    summary_text = _wrap_text(card.get("description"), 18)
    draw.text((x_margin, y_pos), summary_text, font=body_font, fill=text_color)
    y_pos += 120

    insight_text = _wrap_text(card.get("insight"), 20)
    if insight_text:
        draw.text(
            (x_margin, y_pos),
            f"心情結論：\n{insight_text}",
            font=small_font,
            fill=text_color,
        )
        y_pos += 120

    for stat in card.get("stats", []):
        draw.text(
            (x_margin, y_pos),
            f"{stat.get('label')}: {stat.get('value')}",
            font=small_font,
            fill=text_color,
        )
        y_pos += 40

    action_text = _wrap_text(card.get("action"), 18)
    if action_text:
        draw.text(
            (x_margin, y_pos),
            f"建議行動：{action_text}",
            font=small_font,
            fill=text_color,
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

    filename = (
        f"catcard_{user_id}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
        f"{_hash_for_filename(user_id, str(time.time()), cache_key or '')}.png"
    )
    output_path = CAT_CARD_DIR / filename
    base.save(output_path, format="PNG")
    _cleanup_old_cards()

    return filename, _safe_url(source_url)
