from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage
)
from inference import predict_emotion

def handle_text_message(event: MessageEvent, line_bot_api):
    user_text = event.message.text

    label, probs = predict_emotion(user_text)

    reply = f"情緒分析結果：{label}\n\n信心分數：{max(probs):.4f}"

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply)
    )
