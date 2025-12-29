from __future__ import annotations

import logging
import os

from flask import Flask

import services.firebase  # initialize Firebase before routes touch the client
from config.settings import MAX_UPLOAD_BYTES
from routes.api import api_bp
from routes.auth import auth_bp
from routes.cards import cards_bp
from routes.main import main_bp, register_context_processors
from routes.profile import profile_bp
from services.auth_points import refresh_daily_points

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key")
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Strict",
)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.before_request(refresh_daily_points)

app.register_blueprint(main_bp)
app.register_blueprint(auth_bp)
app.register_blueprint(profile_bp)
app.register_blueprint(cards_bp)
app.register_blueprint(api_bp)

register_context_processors(app)

# 保留原本不含 blueprint 前綴的 endpoint 名稱以支援既有 templates。
app.add_url_rule("/", endpoint="home", view_func=app.view_functions["main.home"])
app.add_url_rule(
    "/featured_cats",
    endpoint="featured_cats",
    view_func=app.view_functions["main.featured_cats"],
)
app.add_url_rule(
    "/proxy_image", endpoint="proxy_image", view_func=app.view_functions["main.proxy_image"]
)

app.add_url_rule(
    "/register",
    endpoint="register",
    view_func=app.view_functions["auth.register"],
    methods=["GET", "POST"],
)
app.add_url_rule(
    "/login",
    endpoint="login",
    view_func=app.view_functions["auth.login"],
    methods=["GET", "POST"],
)
app.add_url_rule(
    "/delete_account",
    endpoint="delete_account",
    view_func=app.view_functions["auth.delete_account"],
    methods=["GET", "POST"],
)
app.add_url_rule(
    "/logout",
    endpoint="logout",
    view_func=app.view_functions["auth.logout"],
)

app.add_url_rule(
    "/profile",
    endpoint="profile",
    view_func=app.view_functions["profile.profile"],
    methods=["GET", "POST"],
)

app.add_url_rule(
    "/upload_health",
    endpoint="upload_health",
    view_func=app.view_functions["cards.upload_health"],
    methods=["GET", "POST"],
)
app.add_url_rule(
    "/psychology_test",
    endpoint="psychology_test",
    view_func=app.view_functions["cards.psychology_test"],
    methods=["GET", "POST"],
)
app.add_url_rule(
    "/generate_card",
    endpoint="generate_card",
    view_func=app.view_functions["cards.generate_card"],
)

app.add_url_rule(
    "/chat_api",
    endpoint="chat_api",
    view_func=app.view_functions["api.chat_api"],
    methods=["POST"],
)
app.add_url_rule(
    "/report_api",
    endpoint="report_api",
    view_func=app.view_functions["api.report_api"],
    methods=["POST"],
)
app.add_url_rule(
    "/save_psychology_scores",
    endpoint="save_psychology_scores",
    view_func=app.view_functions["api.save_psychology_scores"],
    methods=["POST"],
)


if __name__ == "__main__":
    debug_enabled = str(os.getenv("FLASK_DEBUG", "0")).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    port = int(os.getenv("FLASK_PORT", "5001"))
    app.run(debug=debug_enabled, port=port)
