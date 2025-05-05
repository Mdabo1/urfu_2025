import os
import logging
from pathlib import Path

from PIL import Image
import pytesseract
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    Defaults,
    filters,
)

# -----------------------------------------------------------------------------
# LangChain import (handles >=0.2.0 deprecation)
# -----------------------------------------------------------------------------
try:
    from langchain_community.chat_models import ChatOpenAI  # langchain >=0.2.0
except ImportError:
    from langchain.chat_models import ChatOpenAI  # langchain <0.2.0 – fallback

from langchain.schema import SystemMessage, HumanMessage

# -----------------------------------------------------------------------------
# NEW: Historical‑document preprocessing
# -----------------------------------------------------------------------------
from preprocessing import preprocess_image  # <-- NEW IMPORT

# -----------------------------------------------------------------------------
# Environment & configuration
# -----------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_TOKEN or not OPENAI_API_KEY:
    raise RuntimeError(
        "Set TELEGRAM_TOKEN and OPENAI_API_KEY as environment variables or in a .env file!"
    )

# Optional: point to tesseract.exe on Windows
if os.name == "nt":
    _tesseract_path = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    if Path(_tesseract_path).exists():
        pytesseract.pytesseract.tesseract_cmd = _tesseract_path

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "bot.log",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(console_handler)

# -----------------------------------------------------------------------------
# OCR helper (with preprocessing)
# -----------------------------------------------------------------------------

def image_to_text(image_path: Path) -> str:
    """
    Run all preprocessing steps (for side-effects, логгинг и т.п.),
    но в Tesseract отдать всегда исходный image_path.
    """
    logger.info("Running preprocessing on %s", image_path)
    try:
        _ = preprocess_image(image_path)
    except Exception as e:
        logger.warning("Preprocessing failed, will use original: %s", e)

    logger.info("Running OCR on original file %s", image_path)
    with Image.open(image_path) as im:
        return pytesseract.image_to_string(im)


# -----------------------------------------------------------------------------
# LLM helper
# -----------------------------------------------------------------------------

def ask_llm(source_text: str, question: str) -> str:
    """Ask GPT‑4o a question answered strictly from the supplied text."""
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )

    messages = [
        SystemMessage(
            content=(
                "Ты — помощник, который отвечает на вопросы по тексту, извлечённого из документа."
                "Текст может содержать ошибки распознавания - если это случается, додумывай контекст исходя из распознанных символов."
                "Отвечай на вопрос пользователя, опираясь ТОЛЬКО на предоставленный текст. "
                "Если ответа нет, скажи, что не знаешь."
            )
        ),
        HumanMessage(content=f"Текст:\n{source_text}\n\nВопрос: {question}"),
    ]

    logger.info("Sending prompt to LLM (question length=%d)", len(question))
    return llm(messages).content.strip()


# -----------------------------------------------------------------------------
# Telegram handlers
# -----------------------------------------------------------------------------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Отправьте фото или документ для распознавания текста.\n"
        "Команды:\n"
        "  /ocr — получить распознанный текст\n"
        "  /ask <вопрос> — задать вопрос по содержимому последнего документа"
    )


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    telegram_file = (
        (update.message.photo[-1] if update.message.photo else None)
        or (update.message.document if update.message.document else None)
    )
    if not telegram_file:
        return

    file_obj = await telegram_file.get_file()

    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    temp_file = temp_dir / f"{telegram_file.file_unique_id}.png"

    await file_obj.download_to_drive(custom_path=temp_file)

    text = image_to_text(temp_file)
    context.chat_data["last_ocr_text"] = text

    logger.info(
        "OCR completed (chat_id=%s, chars=%d)",
        update.effective_chat.id,
        len(text),
    )

    await update.message.reply_text(
        "Документ распознан ✅\n"
        "• /ocr — получить текст\n"
        "• /ask <вопрос> — задать вопрос по содержимому"
    )


async def ocr_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = context.chat_data.get("last_ocr_text")
    if not text:
        await update.message.reply_text("Нет распознанного документа. Сначала отправьте фото.")
        return

    for chunk in (text[i : i + 4096] for i in range(0, len(text), 4096)):
        await update.message.reply_text(chunk)


async def ask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = context.chat_data.get("last_ocr_text")
    if not text:
        await update.message.reply_text("Нет распознанного документа. Сначала отправьте фото.")
        return

    question = " ".join(context.args)
    if not question:
        await update.message.reply_text("Используйте формат: /ask <вопрос>")
        return

    try:
        answer = ask_llm(text, question)
        await update.message.reply_text(answer)
    except Exception as exc:
        logger.exception("LLM request failed: %s", exc)
        await update.message.reply_text("Ошибка при обращении к LLM. Попробуйте позже.")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main() -> None:
    # No global parse_mode to avoid HTML‑parsing issues with angle brackets
    application = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .defaults(Defaults())
        .build()
    )

    application.add_handler(CommandHandler("start", start_cmd))
    application.add_handler(CommandHandler("ocr", ocr_cmd))
    application.add_handler(CommandHandler("ask", ask_cmd))
    application.add_handler(
        MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_image)
    )

    logger.info("Bot started!")
    application.run_polling()


if __name__ == "__main__":
    main()
