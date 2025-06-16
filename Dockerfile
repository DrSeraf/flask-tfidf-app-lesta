# Используем официальный минимальный образ Python
FROM python:3.11-slim

# Переменные окружения для Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Установка системных зависимостей
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Создаём рабочую директорию
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем зависимости Python
RUN pip install --no-cache-dir -r requirements.txt

# Копируем все файлы приложения
COPY . .

# Копируем .env файл (если нужен внутри контейнера)
COPY .env .env

# Загружаем стоп-слова NLTK (требуется для приложения)
RUN python -m nltk.downloader stopwords

# Переменная окружения для шрифта (если используется)
ENV FONT_PATH=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf

# Открываем порт 5005 (как в app.run)
EXPOSE 5005

# Запуск приложения
CMD ["python", "app.py"]
