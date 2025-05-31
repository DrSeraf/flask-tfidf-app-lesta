# Используем мультиархитектурный базовый образ
FROM --platform=$BUILDPLATFORM python:3.11-slim

# Устанавливаем системные зависимости для всех архитектур
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы приложения
COPY . .

# Устанавливаем зависимости Python
RUN pip install --no-cache-dir -r requirements.txt

# Загружаем ресурсы NLTK
RUN python -m nltk.downloader stopwords

# Используем системный шрифт DejaVu для кириллицы
ENV FONT_PATH=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf

# Открываем порт 5000
EXPOSE 5000

# Запускаем приложение
CMD ["python", "app.py"]
