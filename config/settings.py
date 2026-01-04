from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
CAT_CARD_DIR = STATIC_DIR / "cat_cards"
CAT_CARD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 10
ALLOWED_UPLOAD_EXTENSIONS = (".jpg", ".jpeg", ".png", ".pdf")
ALLOWED_UPLOAD_MIMES = {
    "image/jpeg",
    "image/png",
    "application/pdf",
}
MAX_USER_TEXT_CHARS = 1000
REGISTER_RATE_LIMIT_WINDOW_SECONDS = int(
    os.getenv("REGISTER_RATE_LIMIT_WINDOW_SECONDS", "3600")
)
REGISTER_RATE_LIMIT_MAX_PER_IP = int(
    os.getenv("REGISTER_RATE_LIMIT_MAX_PER_IP", "10")
)
REGISTER_RATE_LIMIT_MAX_PER_EMAIL = int(
    os.getenv("REGISTER_RATE_LIMIT_MAX_PER_EMAIL", "5")
)
REGISTER_RATE_LIMIT_MESSAGE = "Please try again later."
REGISTER_HONEYPOT_FIELD = "website"
OVER_LIMIT_MESSAGE = "Oops...字數超過1000字無法傳送唷"
MAX_DAILY_CARD_GENERATIONS = 10
MAX_DAILY_HEALTH_REPORT_UPLOADS = 10
MAX_DAILY_DEMO_GENERATIONS = 1
CARD_LIMIT_MESSAGE = "今日生成次數已達上限，請明天再試。"
CARD_LIMIT_MODAL_TEXT = "Oops...您今日已達生成圖片次數上限，請明天再過來唷！"
RECOMMENDATION_LABELS = {
    "movie": "推薦電影",
    "music": "推薦音樂",
    "activity": "推薦活動",
}

FIREBASE_AUTH_ENDPOINT = (
    "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"
)

FIREBASE_CREDENTIALS = os.getenv("FIREBASE_CREDENTIALS")
FIREBASE_STORAGE_BUCKET = os.getenv(
    "FIREBASE_STORAGE_BUCKET", "gold-chassis-473807-j1.firebasestorage.app"
)
FIREBASE_WEB_API_KEY = os.getenv("FIREBASE_WEB_API_KEY")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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
