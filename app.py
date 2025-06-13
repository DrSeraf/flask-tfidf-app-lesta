import os
import re
import time
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
from io import BytesIO
import nltk
from nltk.corpus import stopwords
from corpus import CORPUS

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import func
from dotenv import load_dotenv

from flasgger import Swagger

# Загрузка переменных окружения из .env
load_dotenv()

# --- Конфигурируемые параметры ---
FLASK_RUN_PORT = int(os.getenv('FLASK_RUN_PORT', 5000))
SQLITE_DB_PATH = os.getenv('SQLITE_DB_PATH', 'app_database.db')
FLASK_SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'supersecret')
APP_VERSION = os.getenv('APP_VERSION', 'dev')
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'Uploads')
MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 2 * 1024 * 1024))
PAGE_SIZE = int(os.getenv('PAGE_SIZE', 15))

# --- Flask и БД ---
app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{SQLITE_DB_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# --- Flasgger ---
swagger = Swagger(app)

# --- Модель для истории загрузок ---
class UploadHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(120))
    processed_at = db.Column(db.DateTime, default=datetime.utcnow)
    processing_time = db.Column(db.Float)  # в миллисекундах
    document_size = db.Column(db.Integer)

with app.app_context():
    db.create_all()

# --- NLTK ---
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))

# --- Векторизаторы ---
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

tfidf_vectorizer = TfidfVectorizer(
    preprocessor=preprocess,
    sublinear_tf=True,
    smooth_idf=False
)
tfidf_vectorizer.fit([preprocess(doc) for doc in CORPUS])

count_vectorizer = CountVectorizer(
    vocabulary=tfidf_vectorizer.vocabulary_,
    preprocessor=preprocess
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'txt'}

def calculate_tfidf(text):
    processed_text = preprocess(text)
    tfidf_matrix = tfidf_vectorizer.transform([processed_text])
    count_matrix = count_vectorizer.transform([processed_text])
    features = tfidf_vectorizer.get_feature_names_out()
    idf = tfidf_vectorizer.idf_

    data = {
        'Слово': [],
        'TF': [],
        'IDF': []
    }
    seen_words = set()
    for i, word in enumerate(features):
        tf = count_matrix[0, i]
        if tf > 0 and word not in seen_words:
            data['Слово'].append(word)
            data['TF'].append(tf)
            data['IDF'].append(idf[i])
            seen_words.add(word)
    df = pd.DataFrame(data)
    df = df.sort_values('IDF', ascending=False)
    return df

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Главная страница загрузки файла
    ---
    tags:
      - Web
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: Текстовый файл .txt для анализа
    responses:
      200:
        description: Успешная загрузка и анализ файла
      400:
        description: Ошибка загрузки файла
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            start_time = time.time()
            df = calculate_tfidf(text)
            processing_time = (time.time() - start_time) * 1000  # ms

            # Сохраняем в БД
            record = UploadHistory(
                filename=file.filename,
                processing_time=processing_time,
                document_size=len(text)
            )
            db.session.add(record)
            db.session.commit()

            session['df'] = df.to_json(orient='split', force_ascii=False)
            return redirect(url_for('results', page=1))
    return render_template('index.html')

@app.route('/results')
def results():
    """
    Результаты анализа TF-IDF
    ---
    tags:
      - Web
    parameters:
      - name: page
        in: query
        type: integer
        required: false
        description: Номер страницы
    responses:
      200:
        description: HTML страница с результатами
    """
    import io
    df = pd.read_json(io.StringIO(session.get('df')), orient='split')
    total = len(df)
    page = int(request.args.get('page', 1))
    pages = (total + PAGE_SIZE - 1) // PAGE_SIZE if total > PAGE_SIZE else 1
    if total > PAGE_SIZE:
        df_page = df.iloc[(page-1)*PAGE_SIZE:page*PAGE_SIZE]
    else:
        df_page = df
    return render_template('results.html',
                           table=df_page.to_html(classes='table table-striped', index=False),
                           page=page, pages=pages, total=total,
                           wordcloud_img=None, barchart_img=None)

@app.route('/download_csv')
def download_csv():
    """
    Скачать результаты анализа в CSV
    ---
    tags:
      - Web
    responses:
      200:
        description: CSV-файл с результатами
        schema:
          type: file
    """
    import io
    df = pd.read_json(io.StringIO(session.get('df')), orient='split')
    buffer = BytesIO()
    df.to_csv(buffer, index=False, encoding='utf-8')
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype='text/csv; charset=utf-8',
        as_attachment=True,
        download_name='tfidf_results.csv'
    )

# --- API endpoints ---

@app.route('/status')
def status():
    """
    Проверка статуса сервиса
    ---
    tags:
      - API
    responses:
      200:
        description: Сервис работает
        schema:
          type: object
          properties:
            status:
              type: string
              example: OK
      500:
        description: Ошибка соединения с БД
        schema:
          type: object
          properties:
            status:
              type: string
              example: ERROR
            detail:
              type: string
    """
    try:
        db.session.execute('SELECT 1')
        return jsonify({"status": "OK"})
    except Exception as e:
        return jsonify({"status": "ERROR", "detail": str(e)}), 500

@app.route('/metrics')
def metrics():
    """
    Метрики обработки файлов
    ---
    tags:
      - API
    responses:
      200:
        description: Метрики обработки
        schema:
          type: object
          properties:
            processed_documents:
              type: integer
              example: 10
            average_processing_time_ms:
              type: number
              example: 123.45
            average_document_size_chars:
              type: number
              example: 1000
      500:
        description: Ошибка получения метрик
        schema:
          type: object
          properties:
            status:
              type: string
              example: ERROR
            detail:
              type: string
    """
    try:
        processed_docs = db.session.query(UploadHistory).count()
        avg_time = db.session.query(func.avg(UploadHistory.processing_time)).scalar() or 0
        avg_doc_size = db.session.query(func.avg(UploadHistory.document_size)).scalar() or 0
        return jsonify({
            "processed_documents": processed_docs,
            "average_processing_time_ms": round(avg_time, 2),
            "average_document_size_chars": round(avg_doc_size, 2)
        })
    except Exception as e:
        return jsonify({"status": "ERROR", "detail": str(e)}), 500

@app.route('/version')
def version():
    """
    Получить версию приложения
    ---
    tags:
      - API
    responses:
      200:
        description: Версия приложения
        schema:
          type: object
          properties:
            version:
              type: string
              example: "1.0.0"
    """
    return jsonify({"version": APP_VERSION})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True)
