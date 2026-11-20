# 🎬 Movie Recommendation Telegram Bot (ALS)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![implicit](https://img.shields.io/badge/implicit-ALS-0A66C2)](https://benfred.github.io/implicit/)
[![aiogram](https://img.shields.io/badge/aiogram-3.x-2CA5E0?logo=telegram&logoColor=white)](https://docs.aiogram.dev/)

Pet-проект рекомендательной системы фильмов: коллаборативная фильтрация на матричной факторизации (ALS) с продовым инференсом через Telegram-бота.

## 📌 Описание

Сервис выдаёт персональные рекомендации фильмов на основе истории взаимодействий пользователей. Модель обучается на бенчмарке **MovieLens 100k** и считает топ-N рекомендаций для пользователя, исключая уже просмотренные тайтлы.

## 🛠 Стек

- **Python 3.10+**
- **ML / RecSys:** `implicit` (ALS), `scipy.sparse`, `pandas`
- **Бот:** `aiogram 3.x`, `asyncio`

## 🧠 Как это работает

1. **Данные:** разреженная матрица User-Item по MovieLens 100k (943 пользователя, 1682 фильма, 100k оценок)
2. **Алгоритм:** iALS — `AlternatingLeastSquares` (32 скрытых фактора, 15 итераций, регуляризация 0.1). Матрица оценок масштабируется ×10 как уровень уверенности для неявного фидбека. Модель факторизует $R \approx U \cdot V^T$ — низкоранговые эмбеддинги пользователей и фильмов
3. **Инференс:** скалярное произведение эмбеддинга пользователя с матрицей фильмов → фильтр просмотренного → топ-5

## 🚀 Быстрый старт

```bash
git clone https://github.com/rawspec/recsys-tg.git
cd recsys-tg

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**1. Обучение модели** (скрипт сам скачает MovieLens 100k и сохранит артефакты в `model_data.pkl`):

```bash
python train.py
```

**2. Запуск бота** — задай токен от [@BotFather](https://t.me/BotFather):

```bash
export BOT_TOKEN="ваш_токен"    # Windows: set BOT_TOKEN=ваш_токен
python bot.py
```

**3. Пользование:** отправь боту ID пользователя от 1 до 943 — получишь топ-5 персональных рекомендаций.

## 📁 Структура проекта

```
recsys-tg/
├── train.py         # скачивание MovieLens, обучение ALS, сохранение артефактов
├── bot.py           # Telegram-бот (aiogram 3.x)
├── model_data.pkl   # обученная модель + маппинги (артефакты train.py)
├── requirements.txt
└── README.md
```

## ⚠️ Ограничения и roadmap

- Бот работает только с пользователями обучающей выборки (ID 1–943) — **cold start** нового пользователя не обрабатывается
- Нет офлайн-оценки качества (hold-out метрики: Precision@K, MAP@K) — стоит добавить
- Одна глобальная модель, нет онлайн-обучения по реакциям
- Идеи на будущее: content-based фолбэк для холодных пользователей, переключение на нейросетевые эмбеддинги (Two-Tower), деплой через Docker, сбор фидбека из самого бота
