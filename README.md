# Распознавание архивных документов с использованием алгоритмов ИИ

![Python](https://img.shields.io/badge/Python-3.11%2B-blue) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **Магистерская диссертация, Уральский федеральный университет**
> Направление: *Инженерия машинного обучения*
> Автор: **Акбулатов Нурислам Марселевич**

---

## О проекте

Этот репозиторий содержит Telegram‑бота, который автоматизирует распознавание архивных (в том числе рукописных) документов.

Бот выполняет три ключевые задачи:

1. **Предобработка изображений** — улучшает контраст и устраняет шумы, не меняя ориентацию страницы.
2. **OCR** (Tesseract 5+) — извлекает текст с учётом исторической специфики.
3. **Пост‑обработка GPT‑4o** — отвечает на вопросы пользователя по распознанному тексту.

> 📜 **Почему без deskew/rotation?**
> Для хрупких исторических документов параметрическая deskew‑коррекция часто ухудшает читаемость и ломает деликатную каллиграфию. Поэтому угол наклона остаётся «как есть».

---

## Быстрый старт

```bash
# 1. Клонируйте репозиторий
git clone https://github.com/<username>/<repo>.git && cd <repo>

# 2. Создайте виртуальное окружение
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# 3. Установите зависимости Python
pip install -r requirements.txt

# 4. Установите Tesseract ≥ 5.0 вместе с русской и английской моделью
## Linux (deb‑based)
sudo apt update && \
  sudo apt install tesseract-ocr tesseract-ocr-language-data \
  tesseract-ocr-rus tesseract-ocr-eng
## macOS (Homebrew)
brew install tesseract-lang  # включает eng и rus
## Windows
# 1) Скачайте инсталлятор с https://github.com/UB-Mannheim/tesseract/wiki
# 2) Во время установки отметьте English и Russian traineddata
#    (или позже скопируйте rus.traineddata / eng.traineddata в tessdata)

# 5. Добавьте Tesseract в PATH **или** укажите путь вручную в main.py
## Способ A — добавить PATH (рекомендуемо)
#  Windows PowerShell (пример)
setx PATH "%PATH%;C:\\Program Files\\Tesseract-OCR"
## Способ B — отредактировать main.py
#  _tesseract_path = r"C:\\Custom\\Path\\tesseract.exe"

# 6. Скопируйте .env.example и задайте токены
cp .env.example .env  # затем откройте .env и впишите TELEGRAM_TOKEN + OPENAI_API_KEY

# 7. Запустите бота
python main.py
```

> ⚠️ На Windows перезапустите терминал, чтобы PATH применился. Если ПО установлено
> в нетипичную папку, корректируйте `_tesseract_path` (см. комментарий в *main.py*).

Бот начнёт «слушать» входящие сообщения. В Telegram отправьте ему 📄 изображение и используйте команды:

* `/ocr` — получить извлечённый текст;
* `/ask <вопрос>` — спросить что‑угодно по содержимому последнего документа.

---

## Переменные окружения (`.env`)

| Переменная       | Описание                        |
| ---------------- | ------------------------------- |
| `TELEGRAM_TOKEN` | Токен BotFather                 |
| `OPENAI_API_KEY` | Ключ OpenAI с доступом к GPT‑4o |

> Никогда не коммитьте `.env` в открытый репозиторий.

---

## Зависимости

* **Python** ≥ 3.11
* [`opencv-python`](https://pypi.org/project/opencv-python/) — предобработка;
* [`pytesseract`](https://pypi.org/project/pytesseract/) + Tesseract ≥ 5.0;
* [`python-telegram-bot` 21.x](https://docs.python-telegram-bot.org/) — взаимодействие с API Telegram;
* [`langchain`](https://python.langchain.com/) — унифицированная работа с LLM;
* полные версии см. `requirements.txt`.

---

## Обзор архитектуры

```text
┌──────────────┐          ┌──────────────┐        ┌──────────────┐
│  Telegram    │  photo  │  Preprocess   │  img   │   Tesseract  │
│    Client    │───────▶│  (OpenCV)     │──────▶│    OCR       │
└──────────────┘         └──────────────┘        └──────┬───────┘
                                                          │ text
                                                          ▼
                                                  ┌──────────────┐
                                                  │   GPT‑4o     │
                                                  │  (Q&A)       │
                                                  └──────────────┘
```

* **Preprocessing** (`preprocessing.py`): CLAHE, билатеральная фильтрация, адаптивный гауссов порог, морфологическая «заливка».
* **OCR** (`image_to_text` в `main.py`): Tesseract v5, русско‑английская модель (`--psm 6` по умолчанию).
* **LLM** (`ask_llm`): zero‑temperature, system‑prompt фиксирует «отвечать только из текста».

---

## Структура репозитория

```text
.
├── main.py              # Telegram‑бот
├── preprocessing.py     # Пайплайн предобработки
├── samples/             # Пример изображений
├── temp/                # Временные файлы (gitignored)
├── logs/                # Логи работы бота (gitignored)
├── requirements.txt     # Python‑зависимости
├── .env.example         # Шаблон конфигурации окружения
└── README.md            # Этот файл
```

---

## Логирование

Все события пишутся в `logs/bot.log` + дублируются в stdout, чтобы удобно смотреть при запуске Docker/PM2.

---

## Запуск в Docker (опционально)

```bash
# Сборка
docker build -t archival-ocr .

# Запуск (Tesseract внутри контейнера)
docker run -e TELEGRAM_TOKEN=xxx -e OPENAI_API_KEY=yyy archival-ocr
```

> Учтите, что внутрь контейнера не попадает ваш `tessdata`. Если нужна кастомная русская/старославянская модель, смонтируйте её томом.

---

## Ограничения и TODO

* 📚 **Dataset**: публичные примеры размещены в `samples/`, но обучающий набор не включён из‑за лицензионных ограничений.
* 📝 **Layout‑анализ** страниц пока не реализован.
* 🔒 **Безопасность**: секреты берутся из переменных окружения; убедитесь, что логи не содержат чувствительных данных.

---

## Лицензия

Этот проект распространяется под лицензией **MIT**. Подробности в `LICENSE`.

---

> *«Не всякий рукописный текст поддаётся OCR, но каждый оцифрованный документ увеличивает доступность культурного наследия».*
