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
        max_length=100
    )
    response = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=100,
        num_beams=5,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(response[0], skip_special_tokens=True)

def split_text(text, max_words=100):
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я бот-переводчик с английского языка на русский язык. просто напиши мне текст на английском, и я переведу тебе его на русский. :)")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:

        user_input = update.message.text
        parts = split_text(user_input, max_words=9)

        translated_parts = []
        for part in parts:
            translated_part = generate_response(part)
            translated_parts.append(translated_part)

        full_translation = ' '.join(translated_parts)
        await update.message.reply_text(full_translation)
    except Exception as e:
        await update.message.reply_text("Произошла ошибка. Попробуйте позже.")
        logging.error(f"Ошибка при обработке сообщения: {e}")

if __name__ == "__main__":
    app = ApplicationBuilder().token("8009389194:AAHL628RacH0p4dze2v1yfQDuainy3cAl7Q").build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()