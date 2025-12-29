# Meow Yourself

Meow Yourself 將溫暖貓咪圖卡、Google GenAI 與 Firebase 服務整合，幫助使用者把情緒轉化為專屬的視覺故事與身心建議。

## 功能介紹

1. **貓咪卡片牆（Featured Cats）**：瀏覽精選貓咪卡片並點擊選擇，會記錄該貓作為後續圖卡生成的靈感並在介面呈現動畫選取效果。  
2. **圖卡生成（Card Generation）**：收集使用者輸入文字、健康資訊與貓咪風格，透過 Gemini 與 `Pillow` 生成可下載的圖卡，同步驗證圖片格式與避免檔名衝突。  
3. **健康資料上傳（Health Uploads）**：支援 PDF、圖片或文字問卷，呼叫 `health_report_module` 分析後呈現友善的健康報告與建議。  
4. **隱私與追蹤（Privacy & Tracking）**：每個模板都載入 Google Tag Manager（`GTM-K2VQTGFM`）與 GA4，並在頁尾說明 Cookie 使用，前端同時落實中文排版與 RWD 易讀性。

## 快速啟動

### 系統需求

- Python 3.11 以上（依 `requirements.txt` 安裝）  
- Google Cloud 服務帳號 JSON：`firebase_credentials/service_account.json`（切勿推上 Git）  
- `.env` 需包含以下環境變數：  
  - `GEMINI_API_KEY`  
  - `FIREBASE_WEB_API_KEY`  
  - `FIREBASE_STORAGE_BUCKET`

### 建置

```bash
python -m venv .venv
.\\.venv\\Scripts\\activate  # Windows
pip install -r requirements.txt
```

### 設定

1. 複製 `.env` 範本並填入上述變數。  
2. 確保 `firebase_credentials/service_account.json` 連到正確的 Firebase 專案。  
3. 若需直接與 Google Cloud Storage 互動，可設定 `GOOGLE_APPLICATION_CREDENTIALS` 指向該 JSON。

### 執行

```bash
flask run
```

部署時可改用 `gunicorn app:app`。需要開啟除錯，可透過 `.env` 或 `export FLASK_ENV=development` 設定。

## 部署建議

- 可以部署到 GCP Cloud Run、App Engine 或其他支援 Flask 的平台。  
- `Dockerfile.txt` 為開發參考，可改寫為正式的 `Dockerfile` 並搭配 `.dockerignore`。  
- 將 `.env`、`service_account.json` 等機密資訊透過平台的 Secret Manager 或環境變數提供，不要寫進版本控制。

## 測試建議

- 目前未提供自動化測試；可針對 `health_report_module`、`extract_json_from_response` 等函式撰寫單元測試。  
- 可加入 E2E 測試，模擬使用者在 `featured_cats` 選貓、上傳資料與生成圖卡的流程。

## 補充說明

- 所有 HTML 模板都注入 GTM 片段與 `<noscript>` iframe。  
- 前端包含隱私模組、團隊卡片、輪播等客製 CSS，調整時需同步檢視排版。  
- 圖卡生成會驗證上傳檔案大小、格式 (`MAX_UPLOAD_BYTES`, `imghdr`)，並透過 `hashlib` 及 `random` 生成安全檔名。

