import os
import re
import uuid
from datetime import datetime
from functools import wraps
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import func
from dotenv import load_dotenv
import nltk
from nltk.corpus import stopwords
from flasgger import Swagger
from flasgger.utils import swag_from
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

# Загрузка переменных окружения
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'supersecret')
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.getenv('SQLITE_DB_PATH', 'app_database.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 2 * 1024 * 1024))

# Настройка Swagger
app.config['SWAGGER'] = {
    'title': 'TF-IDF API',
    'uiversion': 3,
    'specs_route': '/apidocs/',
    'securityDefinitions': {
        'Bearer': {
            'type': 'apiKey',
            'name': 'x-access-token',
            'in': 'header'
        }
    },
    'security': [{'Bearer': []}]
}

swagger = Swagger(app)

db = SQLAlchemy(app)

# Инициализация NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))

# Модели базы данных
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

class Document(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = db.Column(db.String(100), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class Collection(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(100), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class CollectionDocument(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    collection_id = db.Column(db.String(36), db.ForeignKey('collection.id'), nullable=False)
    document_id = db.Column(db.String(36), db.ForeignKey('document.id'), nullable=False)

# Вспомогательные функции
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('x-access-token')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        current_user = User.query.get(token)
        if not current_user:
            return jsonify({'message': 'Token is invalid!'}), 401
        return f(current_user, *args, **kwargs)
    return decorated

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def calculate_tfidf(documents):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    
    tfidf_scores = []
    for doc_idx, doc in enumerate(documents):
        feature_index = tfidf_matrix[doc_idx, :].nonzero()[1]
        tfidf_scores_doc = zip(
            [feature_names[i] for i in feature_index],
            [tfidf_matrix[doc_idx, i] for i in feature_index]
        )
        tfidf_scores.append(sorted(tfidf_scores_doc, key=lambda x: x[1], reverse=True))
    
    return tfidf_scores

def get_document_text(document_id):
    document = Document.query.get(document_id)
    if not document:
        return None
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], document.filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return None

# API Endpoints

# User endpoints
@app.route('/register', methods=['POST'])
@swag_from({
    'tags': ['Users'],
    'description': 'Register a new user',
    'parameters': [{
        'name': 'body',
        'in': 'body',
        'required': True,
        'schema': {
            'type': 'object',
            'properties': {
                'username': {'type': 'string', 'example': 'test_user'},
                'password': {'type': 'string', 'example': 'test_pass'}
            }
        }
    }],
    'responses': {
        201: {'description': 'User created successfully'},
        400: {'description': 'Username already exists or invalid input'}
    }
})
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'message': 'Username and password required'}), 400
    
    if User.query.filter_by(username=username).first():
        return jsonify({'message': 'User already exists'}), 400
        
    hashed_password = generate_password_hash(password)
    new_user = User(username=username, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    
    return jsonify({'message': 'User created successfully'}), 201

@app.route('/login', methods=['POST'])
@swag_from({
    'tags': ['Users'],
    'description': 'Login user',
    'parameters': [{
        'name': 'body',
        'in': 'body',
        'required': True,
        'schema': {
            'type': 'object',
            'properties': {
                'username': {'type': 'string', 'example': 'test_user'},
                'password': {'type': 'string', 'example': 'test_pass'}
            }
        }
    }],
    'responses': {
        200: {'description': 'Login successful', 'schema': {'properties': {'token': {'type': 'string'}}}},
        401: {'description': 'Invalid credentials'}
    }
})
def login():
    auth = request.get_json()
    username = auth.get('username')
    password = auth.get('password')
    
    if not username or not password:
        return jsonify({'message': 'Username and password required'}), 400
    
    user = User.query.filter_by(username=username).first()
    
    if not user or not check_password_hash(user.password, password):
        return jsonify({'message': 'Invalid credentials'}), 401
    
    return jsonify({'token': str(user.id)}), 200

@app.route('/logout', methods=['GET'])
@token_required
@swag_from({
    'tags': ['Users'],
    'description': 'Logout user',
    'security': [{'x-access-token': []}],
    'responses': {
        200: {'description': 'Successfully logged out'}
    }
})
def logout(current_user):
    # В нашей реализации токен хранится на клиенте, поэтому просто возвращаем успех
    return jsonify({'message': 'Successfully logged out'}), 200

@app.route('/user/<user_id>', methods=['PATCH'])
@token_required
@swag_from({
    'tags': ['Users'],
    'description': 'Change user password',
    'parameters': [
        {
            'name': 'user_id',
            'in': 'path',
            'type': 'string',
            'required': True
        },
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'new_password': {'type': 'string', 'example': 'new_password'}
                }
            }
        }
    ],
    'security': [{'x-access-token': []}],
    'responses': {
        200: {'description': 'Password changed successfully'},
        403: {'description': 'Forbidden - cannot change other user password'},
        400: {'description': 'Invalid input'}
    }
})
def change_password(current_user, user_id):
    if str(current_user.id) != user_id:
        return jsonify({'message': 'You can only change your own password'}), 403
    
    data = request.get_json()
    new_password = data.get('new_password')
    
    if not new_password:
        return jsonify({'message': 'New password required'}), 400
        
    current_user.password = generate_password_hash(new_password)
    db.session.commit()
    
    return jsonify({'message': 'Password changed successfully'}), 200

@app.route('/user/<user_id>', methods=['DELETE'])
@token_required
@swag_from({
    'tags': ['Users'],
    'description': 'Delete user account',
    'parameters': [{
        'name': 'user_id',
        'in': 'path',
        'type': 'string',
        'required': True
    }],
    'security': [{'x-access-token': []}],
    'responses': {
        200: {'description': 'User deleted successfully'},
        403: {'description': 'Forbidden - cannot delete other user account'}
    }
})
def delete_user(current_user, user_id):
    if str(current_user.id) != user_id:
        return jsonify({'message': 'You can only delete your own account'}), 403
    
    # Удаляем все документы пользователя
    documents = Document.query.filter_by(user_id=current_user.id).all()
    for doc in documents:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], doc.filename)
        if os.path.exists(filepath):
            os.remove(filepath)
    
    # Удаляем все коллекции пользователя
    Collection.query.filter_by(user_id=current_user.id).delete()
    
    # Удаляем самого пользователя
    db.session.delete(current_user)
    db.session.commit()
    
    return jsonify({'message': 'User deleted successfully'}), 200

# Document endpoints
@app.route('/documents', methods=['GET'])
@token_required
@swag_from({
    'tags': ['Documents'],
    'description': 'Get list of user documents',
    'security': [{'x-access-token': []}],
    'responses': {
        200: {'description': 'List of documents', 
              'schema': {
                  'type': 'object',
                  'properties': {
                      'documents': {
                          'type': 'array',
                          'items': {
                              'type': 'object',
                              'properties': {
                                  'id': {'type': 'string'},
                                  'filename': {'type': 'string'}
                              }
                          }
                      }
                  }
              }}
    }
})
def get_documents(current_user):
    documents = Document.query.filter_by(user_id=current_user.id).all()
    return jsonify({
        'documents': [{'id': doc.id, 'filename': doc.filename} for doc in documents]
    }), 200

@app.route('/documents/<document_id>', methods=['GET'])
@token_required
@swag_from({
    'tags': ['Documents'],
    'description': 'Get document content',
    'parameters': [{
        'name': 'document_id',
        'in': 'path',
        'type': 'string',
        'required': True
    }],
    'security': [{'x-access-token': []}],
    'responses': {
        200: {'description': 'Document content'},
        404: {'description': 'Document not found'},
        403: {'description': 'Forbidden - not your document'}
    }
})
def get_document_content(current_user, document_id):
    document = Document.query.get(document_id)
    if not document:
        return jsonify({'message': 'Document not found'}), 404
    
    if document.user_id != current_user.id:
        return jsonify({'message': 'You can only access your own documents'}), 403
    
    content = get_document_text(document_id)
    if content is None:
        return jsonify({'message': 'Could not read document file'}), 500
    
    return jsonify({'content': content}), 200

@app.route('/documents/<document_id>/statistics', methods=['GET'])
@token_required
@swag_from({
    'tags': ['Documents'],
    'description': 'Get document statistics (TF-IDF)',
    'parameters': [{
        'name': 'document_id',
        'in': 'path',
        'type': 'string',
        'required': True
    }],
    'security': [{'x-access-token': []}],
    'responses': {
        200: {'description': 'Document statistics'},
        404: {'description': 'Document not found'},
        403: {'description': 'Forbidden - not your document'}
    }
})
def get_document_statistics(current_user, document_id):
    document = Document.query.get(document_id)
    if not document:
        return jsonify({'message': 'Document not found'}), 404
    
    if document.user_id != current_user.id:
        return jsonify({'message': 'You can only access your own documents'}), 403
    
    # Получаем все коллекции, в которых есть этот документ
    collections = db.session.query(Collection).join(CollectionDocument).filter(
        CollectionDocument.document_id == document_id,
        Collection.user_id == current_user.id
    ).all()
    
    if not collections:
        return jsonify({'message': 'Document is not in any collection'}), 400
    
    # Для простоты берем первую коллекцию
    collection = collections[0]
    
    # Получаем все документы коллекции
    collection_docs = db.session.query(Document).join(CollectionDocument).filter(
        CollectionDocument.collection_id == collection.id
    ).all()
    
    # Читаем содержимое всех документов
    documents_text = []
    for doc in collection_docs:
        content = get_document_text(doc.id)
        if content:
            documents_text.append(preprocess(content))
    
    if not documents_text:
        return jsonify({'message': 'No valid documents in collection'}), 400
    
    # Рассчитываем TF-IDF
    tfidf_scores = calculate_tfidf(documents_text)
    
    # Находим индекс нашего документа в коллекции
    doc_index = next((i for i, doc in enumerate(collection_docs) if doc.id == document_id), None)
    if doc_index is None:
        return jsonify({'message': 'Document not found in collection'}), 500
    
    # Берем топ-50 слов для этого документа
    top_words = tfidf_scores[doc_index][:50]
    
    return jsonify({
        'statistics': [{'word': word, 'tfidf': float(score)} for word, score in top_words]
    }), 200

@app.route('/documents/<document_id>', methods=['DELETE'])
@token_required
@swag_from({
    'tags': ['Documents'],
    'description': 'Delete document',
    'parameters': [{
        'name': 'document_id',
        'in': 'path',
        'type': 'string',
        'required': True
    }],
    'security': [{'x-access-token': []}],
    'responses': {
        200: {'description': 'Document deleted successfully'},
        404: {'description': 'Document not found'},
        403: {'description': 'Forbidden - not your document'}
    }
})
def delete_document(current_user, document_id):
    document = Document.query.get(document_id)
    if not document:
        return jsonify({'message': 'Document not found'}), 404
    
    if document.user_id != current_user.id:
        return jsonify({'message': 'You can only delete your own documents'}), 403
    
    # Удаляем файл
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], document.filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    
    # Удаляем связи с коллекциями
    CollectionDocument.query.filter_by(document_id=document_id).delete()
    
    # Удаляем документ
    db.session.delete(document)
    db.session.commit()
    
    return jsonify({'message': 'Document deleted successfully'}), 200

# Collection endpoints
@app.route('/collections', methods=['GET'])
@token_required
@swag_from({
    'tags': ['Collections'],
    'description': 'Get list of user collections',
    'security': [{'x-access-token': []}],
    'responses': {
        200: {'description': 'List of collections', 
              'schema': {
                  'type': 'object',
                  'properties': {
                      'collections': {
                          'type': 'array',
                          'items': {
                              'type': 'object',
                              'properties': {
                                  'id': {'type': 'string'},
                                  'name': {'type': 'string'},
                                  'documents': {
                                      'type': 'array',
                                      'items': {'type': 'string'}
                                  }
                              }
                          }
                      }
                  }
              }}
    }
})
def get_collections(current_user):
    collections = Collection.query.filter_by(user_id=current_user.id).all()
    result = []
    
    for collection in collections:
        documents = db.session.query(Document.id).join(CollectionDocument).filter(
            CollectionDocument.collection_id == collection.id
        ).all()
        
        result.append({
            'id': collection.id,
            'name': collection.name,
            'documents': [doc[0] for doc in documents]
        })
    
    return jsonify({'collections': result}), 200

@app.route('/collections/<collection_id>', methods=['GET'])
@token_required
@swag_from({
    'tags': ['Collections'],
    'description': 'Get collection documents',
    'parameters': [{
        'name': 'collection_id',
        'in': 'path',
        'type': 'string',
        'required': True
    }],
    'security': [{'x-access-token': []}],
    'responses': {
        200: {'description': 'List of document IDs in collection',
              'schema': {
                  'type': 'object',
                  'properties': {
                      'documents': {
                          'type': 'array',
                          'items': {'type': 'string'}
                      }
                  }
              }},
        404: {'description': 'Collection not found'},
        403: {'description': 'Forbidden - not your collection'}
    }
})
def get_collection_documents(current_user, collection_id):
    collection = Collection.query.get(collection_id)
    if not collection:
        return jsonify({'message': 'Collection not found'}), 404
    
    if collection.user_id != current_user.id:
        return jsonify({'message': 'You can only access your own collections'}), 403
    
    documents = db.session.query(Document.id).join(CollectionDocument).filter(
        CollectionDocument.collection_id == collection.id
    ).all()
    
    return jsonify({'documents': [doc[0] for doc in documents]}), 200

@app.route('/collections/<collection_id>/statistics', methods=['GET'])
@token_required
@swag_from({
    'tags': ['Collections'],
    'description': 'Get collection statistics (TF-IDF)',
    'parameters': [{
        'name': 'collection_id',
        'in': 'path',
        'type': 'string',
        'required': True
    }],
    'security': [{'x-access-token': []}],
    'responses': {
        200: {'description': 'Collection statistics'},
        404: {'description': 'Collection not found'},
        403: {'description': 'Forbidden - not your collection'}
    }
})
def get_collection_statistics(current_user, collection_id):
    collection = Collection.query.get(collection_id)
    if not collection:
        return jsonify({'message': 'Collection not found'}), 404
    
    if collection.user_id != current_user.id:
        return jsonify({'message': 'You can only access your own collections'}), 403
    
    # Получаем все документы коллекции
    documents = db.session.query(Document).join(CollectionDocument).filter(
        CollectionDocument.collection_id == collection.id
    ).all()
    
    # Читаем содержимое всех документов и объединяем в один "документ"
    combined_text = []
    for doc in documents:
        content = get_document_text(doc.id)
        if content:
            combined_text.append(preprocess(content))
    
    if not combined_text:
        return jsonify({'message': 'No valid documents in collection'}), 400
    
    # Объединяем все документы в один для расчета TF
    combined_doc = ' '.join(combined_text)
    
    # Рассчитываем TF-IDF для коллекции (IDF уже рассчитывается по всем документам коллекции)
    tfidf_scores = calculate_tfidf([combined_doc] + combined_text)
    
    # Берем топ-50 слов для объединенного документа
    top_words = tfidf_scores[0][:50]
    
    return jsonify({
        'statistics': [{'word': word, 'tfidf': float(score)} for word, score in top_words]
    }), 200

@app.route('/collections', methods=['POST'])
@token_required
@swag_from({
    'tags': ['Collections'],
    'description': 'Create new collection',
    'parameters': [{
        'name': 'body',
        'in': 'body',
        'required': True,
        'schema': {
            'type': 'object',
            'properties': {
                'name': {'type': 'string', 'example': 'My Collection'}
            }
        }
    }],
    'security': [{'x-access-token': []}],
    'responses': {
        201: {'description': 'Collection created successfully'},
        400: {'description': 'Invalid input'}
    }
})
def create_collection(current_user):
    data = request.get_json()
    name = data.get('name')
    
    if not name:
        return jsonify({'message': 'Collection name required'}), 400
    
    new_collection = Collection(name=name, user_id=current_user.id)
    db.session.add(new_collection)
    db.session.commit()
    
    return jsonify({
        'message': 'Collection created successfully',
        'collection_id': new_collection.id
    }), 201

@app.route('/collections/<collection_id>/<document_id>', methods=['POST'])
@token_required
@swag_from({
    'tags': ['Collections'],
    'description': 'Add document to collection',
    'parameters': [
        {
            'name': 'collection_id',
            'in': 'path',
            'type': 'string',
            'required': True
        },
        {
            'name': 'document_id',
            'in': 'path',
            'type': 'string',
            'required': True
        }
    ],
    'security': [{'x-access-token': []}],
    'responses': {
        200: {'description': 'Document added to collection successfully'},
        404: {'description': 'Collection or document not found'},
        403: {'description': 'Forbidden - not your collection or document'},
        400: {'description': 'Document already in collection'}
    }
})
def add_document_to_collection(current_user, collection_id, document_id):
    collection = Collection.query.get(collection_id)
    if not collection:
        return jsonify({'message': 'Collection not found'}), 404
    
    if collection.user_id != current_user.id:
        return jsonify({'message': 'You can only modify your own collections'}), 403
    
    document = Document.query.get(document_id)
    if not document:
        return jsonify({'message': 'Document not found'}), 404
    
    if document.user_id != current_user.id:
        return jsonify({'message': 'You can only add your own documents'}), 403
    
    # Проверяем, есть ли уже документ в коллекции
    existing = CollectionDocument.query.filter_by(
        collection_id=collection_id,
        document_id=document_id
    ).first()
    
    if existing:
        return jsonify({'message': 'Document already in collection'}), 400
    
    # Добавляем документ в коллекцию
    new_link = CollectionDocument(collection_id=collection_id, document_id=document_id)
    db.session.add(new_link)
    db.session.commit()
    
    return jsonify({'message': 'Document added to collection successfully'}), 200

@app.route('/collections/<collection_id>/<document_id>', methods=['DELETE'])
@token_required
@swag_from({
    'tags': ['Collections'],
    'description': 'Remove document from collection',
    'parameters': [
        {
            'name': 'collection_id',
            'in': 'path',
            'type': 'string',
            'required': True
        },
        {
            'name': 'document_id',
            'in': 'path',
            'type': 'string',
            'required': True
        }
    ],
    'security': [{'x-access-token': []}],
    'responses': {
        200: {'description': 'Document removed from collection successfully'},
        404: {'description': 'Collection or document not found'},
        403: {'description': 'Forbidden - not your collection or document'},
        400: {'description': 'Document not in collection'}
    }
})
def remove_document_from_collection(current_user, collection_id, document_id):
    collection = Collection.query.get(collection_id)
    if not collection:
        return jsonify({'message': 'Collection not found'}), 404
    
    if collection.user_id != current_user.id:
        return jsonify({'message': 'You can only modify your own collections'}), 403
    
    document = Document.query.get(document_id)
    if not document:
        return jsonify({'message': 'Document not found'}), 404
    
    if document.user_id != current_user.id:
        return jsonify({'message': 'You can only remove your own documents'}), 403
    
    # Находим и удаляем связь
    link = CollectionDocument.query.filter_by(
        collection_id=collection_id,
        document_id=document_id
    ).first()
    
    if not link:
        return jsonify({'message': 'Document not in collection'}), 400
    
    db.session.delete(link)
    db.session.commit()
    
    return jsonify({'message': 'Document removed from collection successfully'}), 200

@app.route('/upload', methods=['POST'])
@token_required
@swag_from({
    'tags': ['Documents'],
    'description': 'Upload a text document',
    'parameters': [{
        'name': 'file',
        'in': 'formData',
        'type': 'file',
        'required': True
    }],
    'security': [{'x-access-token': []}],
    'responses': {
        201: {'description': 'File uploaded successfully'},
        400: {'description': 'No file provided'}
    }
})
def upload_file(current_user):
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400
    
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    filename = f"{uuid.uuid4()}.txt"
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    new_doc = Document(filename=filename, user_id=current_user.id)
    db.session.add(new_doc)
    db.session.commit()
    
    return jsonify({
        'message': 'File uploaded successfully',
        'document_id': new_doc.id,
        'filename': filename
    }), 201

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5005, debug=True)