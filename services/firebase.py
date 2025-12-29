from __future__ import annotations

import json
import logging
from pathlib import Path

import firebase_admin
from firebase_admin import auth, credentials, firestore, storage

from config.settings import BASE_DIR, FIREBASE_CREDENTIALS, FIREBASE_STORAGE_BUCKET


def _load_credentials():
    if FIREBASE_CREDENTIALS:
        try:
            payload = json.loads(FIREBASE_CREDENTIALS)
        except json.JSONDecodeError as exc:
            logging.error(f"FIREBASE_CREDENTIALS contains invalid JSON: {exc}")
            raise ValueError("Invalid FIREBASE_CREDENTIALS JSON payload") from exc
        logging.debug("Firebase credentials loaded from environment payload")
        return credentials.Certificate(payload)
    credential_path = BASE_DIR / "firebase_credentials" / "service_account.json"
    logging.debug(f"Firebase credentials loaded from file: {credential_path}")
    return credentials.Certificate(str(credential_path))


def initialize_firebase():
    cred = _load_credentials()
    firebase_admin.initialize_app(cred, {"storageBucket": FIREBASE_STORAGE_BUCKET})
    logging.debug("Firebase initialized with bucket: %s", FIREBASE_STORAGE_BUCKET)


initialize_firebase()

db = firestore.client()
try:
    bucket = storage.bucket()
    logging.debug("Storage bucket initialized: %s", bucket.name)
except Exception as exc:
    logging.error("Storage bucket initialization failed: %s", exc)
    raise
