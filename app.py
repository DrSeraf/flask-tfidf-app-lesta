import matplotlib
matplotlib.use('Agg')  
from flask import Flask, render_template, request, redirect, url_for, session, send_file
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from corpus import CORPUS
import math
from io import StringIO, BytesIO
import base64
from wordcloud import WordCloud
from matplotlib import pyplot as plt  # ← Изменённый импорт


# NLTK init
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))

app = Flask(__name__)
app.secret_key = 'supersecret'
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

PAGE_SIZE = 15

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
    # Путь к кириллическому шрифту, например, Arial
    font_path = os.path.join('static', 'arial.ttf')
    if not os.path.exists(font_path):
        font_path = None  # Оставить по умолчанию, если нет arial.ttf
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
            file.save(filepath)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            df = calculate_tfidf(text)
            session['df'] = df.to_json(orient='split', force_ascii=False)
            return redirect(url_for('results', page=1))
    return render_template('index.html')

@app.route('/results')
def results():
    df = pd.read_json(session.get('df'), orient='split')
    total = len(df)
    page = int(request.args.get('page', 1))
    pages = math.ceil(total / PAGE_SIZE) if total > PAGE_SIZE else 1
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
    
    # Создаём BytesIO вместо StringIO
    buffer = BytesIO()
    # Сохраняем CSV в бинарном режиме с указанием кодировки
    df.to_csv(buffer, index=False, encoding='utf-8')
    buffer.seek(0)  # Сбрасываем позицию в начало
    
    return send_file(
        buffer,
        mimetype='text/csv; charset=utf-8',
        as_attachment=True,
        download_name='tfidf_results.csv'
    )


if __name__ == '__main__':
    app.run(debug=True)
