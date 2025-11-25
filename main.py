import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from dotenv import load_dotenv
from line_handler import handle_text_message
from linebot.models import MessageEvent, TextMessage

app = FastAPI()

# CORS (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# .env を読み込む
load_dotenv()
CHANNEL_ACCESS_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")

line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

@app.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature")

    body = await request.body()
    body_str = body.decode("utf-8")

    try:
        handler.handle(body_str, signature)
    except InvalidSignatureError:
        return "Invalid signature", 400

    return "OK"


# Handler for text messages
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    handle_text_message(event, line_bot_api)


# 動作確認用 root endpoint
@app.get("/")
def root():
    return {"status": "LINE Emotion Bot is running!"}
