import os
import re
import time
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
from io import BytesIO
import base64
from wordcloud import WordCloud
from matplotlib import pyplot as plt
import nltk
from nltk.corpus import stopwords
from corpus import CORPUS

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import func
from dotenv import load_dotenv

# Загрузка переменных окружения из .env
load_dotenv()

# --- Конфигурируемые параметры ---
FLASK_RUN_PORT = int(os.getenv('FLASK_RUN_PORT', 5000))
SQLITE_DB_PATH = os.getenv('SQLITE_DB_PATH', 'app_database.db')
FLASK_SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'supersecret')
FONT_PATH = os.getenv('FONT_PATH', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
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

def plot_wordcloud(df):
    if df.empty:
        return ""
    font_path = FONT_PATH if os.path.exists(FONT_PATH) else None
    wc = WordCloud(width=600, height=300, background_color='white', font_path=font_path)
    freqs = {row['Слово']: row['TF'] for _, row in df.iterrows()}
    img = wc.generate_from_frequencies(freqs)
    buf = BytesIO()
    img.to_image().save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

def plot_barchart(df):
    if df.empty:
        return ""
    plt.figure(figsize=(8, 3))
    plt.bar(df['Слово'][:10], df['TF'][:10], color='#a24caf')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='PNG')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()

@app.route('/', methods=['GET', 'POST'])
def index():
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
    df = pd.read_json(session.get('df'), orient='split')
    total = len(df)
    page = int(request.args.get('page', 1))
    pages = (total + PAGE_SIZE - 1) // PAGE_SIZE if total > PAGE_SIZE else 1
    if total > PAGE_SIZE:
        df_page = df.iloc[(page-1)*PAGE_SIZE:page*PAGE_SIZE]
    else:
        df_page = df
    wordcloud_img = plot_wordcloud(df)
    barchart_img = plot_barchart(df)
    return render_template('results.html',
                           table=df_page.to_html(classes='table table-striped', index=False),
                           page=page, pages=pages, total=total,
                           wordcloud_img=wordcloud_img,
                           barchart_img=barchart_img)

@app.route('/download_csv')
def download_csv():
    df = pd.read_json(session.get('df'), orient='split')
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
    try:
        # Проверка соединения с БД
        db.session.execute('SELECT 1')
        return jsonify({"status": "OK"})
    except Exception as e:
        return jsonify({"status": "ERROR", "detail": str(e)}), 500

@app.route('/metrics')
def metrics():
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
    return jsonify({"version": APP_VERSION})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=FLASK_RUN_PORT, debug=True)
