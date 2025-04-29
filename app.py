from flask import Flask, render_template, request, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # Ограничение 2MB

# Создаем папку для загрузок если её нет
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Проверка наличия файла в запросе
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        # Проверка на пустой файл
        if file.filename == '':
            return redirect(request.url)
        
        # Сохранение и обработка файла
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Чтение файла
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Вычисление TF-IDF
            tfidf, features, tfidf_matrix = calculate_tfidf(text)
            
            # Создание DataFrame
            df = create_dataframe(tfidf, features, tfidf_matrix)
            
            return render_template('results.html', 
                                table=df.to_html(classes='table table-striped', index=False))
    
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'txt'}

def calculate_tfidf(text):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([text])
    return tfidf, tfidf.get_feature_names_out(), tfidf_matrix

def create_dataframe(tfidf, features, matrix):
    return pd.DataFrame({
        'Слово': features,
        'TF': matrix.toarray()[0],
        'IDF': tfidf.idf_
    }).sort_values('IDF', ascending=False).head(50)

if __name__ == '__main__':
    app.run(debug=True)
