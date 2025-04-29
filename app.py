from flask import Flask, render_template, request, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from corpus import CORPUS

# Инициализация NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024

# Создаём папку для загрузки
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    # Удаляем стоп-слова
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Инициализация TF-IDF
tfidf_vectorizer = TfidfVectorizer(
    preprocessor=preprocess,
    sublinear_tf=True,
    smooth_idf=False
)
tfidf_vectorizer.fit([preprocess(doc) for doc in CORPUS])

# Инициализация CountVectorizer
count_vectorizer = CountVectorizer(
    vocabulary=tfidf_vectorizer.vocabulary_,
    preprocessor=preprocess
)

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
            if df.empty:
                return render_template('results.html', table="Нет данных для отображения")
            
            return render_template('results.html', 
                                 table=df.to_html(classes='table table-striped', index=False))
    
    return render_template('index.html')

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
    
    # Уникальные слова
    seen_words = set()
    
    for i, word in enumerate(features):
        tf = count_matrix[0, i]
        if tf > 0 and word not in seen_words:
            data['Слово'].append(word)
            data['TF'].append(tf)
            data['IDF'].append(idf[i])
            seen_words.add(word)
    
    df = pd.DataFrame(data)
    df = df.sort_values('IDF', ascending=False).head(50)
    return df

if __name__ == '__main__':
    app.run(debug=True)