import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, ContextTypes, filters
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO)

model_path = "model/translation_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_response(prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    response = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=400,
        num_beams=5,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(response[0], skip_special_tokens=True)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я бот-переводчик с английского языка на русский язык. просто напиши мне текст на английском, и я переведу тебе его на русский. Пожалуйста, только пиши не очень большие сообщения, я не умею переводить большие тексты :)")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_input = update.message.text
        bot_reply = generate_response(user_input)
        await update.message.reply_text(bot_reply)
    except Exception as e:
        await update.message.reply_text("Произошла ошибка. Попробуйте позже.")
        logging.error(f"Ошибка при обработке сообщения: {e}")

if __name__ == "__main__":
    app = ApplicationBuilder().token("8009389194:AAHL628RacH0p4dze2v1yfQDuainy3cAl7Q").build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()