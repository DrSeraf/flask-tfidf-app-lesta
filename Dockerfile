# Используем официальный Python-образ
FROM python:3.11-slim

# Устанавливаем переменные окружения для Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Устанавливаем системные зависимости
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
COPY requirements.txt requirements.txt

# Устанавливаем зависимости Python
RUN pip install --no-cache-dir -r requirements.txt

# Копируем все файлы приложения
COPY . .

# Копируем .env файл (если вы хотите, чтобы он попадал в контейнер)
# Если переменные окружения будут передаваться через docker run --env-file, эту строку можно убрать
COPY .env .env

# Загружаем ресурсы NLTK (стоп-слова)
RUN python -m nltk.downloader stopwords

# Указываем переменную окружения для шрифта
ENV FONT_PATH=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf

# Открываем порт (по умолчанию 5000, но можно изменить через .env)
EXPOSE 5000

# Запускаем приложение
CMD ["python", "app.py"]
