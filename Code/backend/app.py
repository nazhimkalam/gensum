import csv
import json
import os
import smtplib
import ssl
from email.message import EmailMessage

import firebase_admin
import pandas as pd
import requests
import transformers
from cryptography.fernet import Fernet
from dotenv import load_dotenv
from firebase_admin import credentials, firestore
from flask import Flask, jsonify, make_response, request
from flask_cors import CORS
from flask_restful import Api
from utils.data_preprocessing import handle_data_preprocessing
from utils.model_retraining import model_customization
from utils.types import DOMAIN_TYPES

"""
    This file contains all the functions that are used to handle the API requests.
"""

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)
API = Api(app)

FIREBASE_CREDENTIAL_CERTIFICATE = os.getenv('FIREBASE_CREDENTIAL_CERTIFICATE')
EMAIL_SENDER=os.getenv('EMAIL_SENDER')
EMAIL_PASSCODE=os.getenv('EMAIL_PASSCODE')
MODEL_NAME = 'bart-base_model'
TOKENIZER_NAME = 'bart-base_tokenizer'
GENERALIZED_MODEL_PATH = 'model/base/' + MODEL_NAME
GENERALIZED_TOKENIZER_PATH = 'model/base/' + TOKENIZER_NAME
MAX_INPUT = 512

HUGGING_FACE_BEARER_TOKEN = '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'
SENTIMENTAL_ANALYSIS_HUGGING_FACE_API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"

generalized_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(GENERALIZED_MODEL_PATH)
generalized_tokenizer = transformers.AutoTokenizer.from_pretrained(GENERALIZED_TOKENIZER_PATH)

firebase_credentials = credentials.Certificate(FIREBASE_CREDENTIAL_CERTIFICATE)
firebaseApp = firebase_admin.initialize_app(firebase_credentials)
db = firestore.client()

def query(payload):
    headers = {"Authorization": "Bearer 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"}
    response = requests.post(SENTIMENTAL_ANALYSIS_HUGGING_FACE_API_URL, headers=headers, json=payload)
    return response.json()

def triggerEmailNotification(subject, body, email_reciever):
    email_ = EmailMessage()
    context = ssl.create_default_context()
    
    email_['From'] = EMAIL_SENDER
    email_['To'] = email_reciever
    email_['Subject'] = subject
    email_.set_content(body)
    
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSCODE)
        server.send_message(email_)
        
def getOverallSentimentWithScore(sentiment):
    # LABEL_0 = 'Negative'
    # LABEL_1 = 'Neutral'
    # LABEL_2 = 'Positive'
    
    negativeScore = sentiment[0][0]['score']
    neutralScore = sentiment[0][1]['score']
    positiveScore = sentiment[0][2]['score']
    
    if negativeScore > neutralScore and negativeScore > positiveScore:
        return 'Negative', negativeScore
    elif neutralScore > negativeScore and neutralScore > positiveScore:
        return 'Neutral', neutralScore
    elif positiveScore > negativeScore and positiveScore > neutralScore:
        return 'Positive', positiveScore
    else:
        return 'Neutral', neutralScore
    
                
@app.route('/', methods=['GET'])
def hello_world():
    return {'message': 'Hello World'}, 200

@app.route('/api/gensum/general', methods=['POST'])
def getGeneralizedSummary():
    try:
        data = request.get_json()
        review = data['review']
        inputs = generalized_tokenizer.encode(review, return_tensors='pt', max_length=MAX_INPUT, truncation=True)
        outputs = generalized_model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = generalized_tokenizer.decode(outputs[0], skip_special_tokens=True)

        sentimentAnalysisOutput = query({ "inputs": summary })
        sentiment, score = getOverallSentimentWithScore(sentimentAnalysisOutput)
        return {'summary': summary, 'sentiment': {
            'sentiment': sentiment,
            'score': score
        } }, 200
    except Exception as e:
        return {'message': str(e)}, 500

@app.route('/api/gensum/review-records/user/<userId>', methods=['GET'])
def getUserData(userId):
    try:
        user = db.collection('users').document(userId).get()
        with open('encryption_key.key', 'rb') as file:
            key = file.read()
        fernet = Fernet(key)
        
        if user.exists:
            reviews = db.collection('users').document(userId).collection('reviews').get()
            user = user.to_dict() 
            user['reviews'] = []
            for review in reviews:
                summary = review.get('summary')
                reviewText = review.get('review')
                sentiment = review.get('sentiment')
                score = review.get('score')
                
                decodedSummary = fernet.decrypt(summary).decode()
                decodedReview = fernet.decrypt(reviewText).decode()
                decodedSentiment = fernet.decrypt(sentiment).decode()
                decodedScore = fernet.decrypt(score).decode()
                
                user['reviews'].append({
                    'id': review.id,
                    'summary': decodedSummary,
                    'review': decodedReview,
                    'sentiment': decodedSentiment,
                    'score': float(decodedScore)*100,
                    'createdAt': review.get('createdAt')
                })
            return user, 200
        else:
            return {'message': "User not found"}, 404
    except Exception as e:
        return {'message': str(e)}, 500

@app.route('/api/gensum/domain-profile', methods=['POST'])
def createDomainUserProfile():
    try:
        data = request.get_json()
        userId = data['userId']
        
        folder_path = 'model/' + userId
        model_path =  folder_path + '/' + MODEL_NAME
        tokenizer_path = folder_path + '/' + TOKENIZER_NAME

        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        generalized_model.save_pretrained(model_path)
        generalized_tokenizer.save_pretrained(tokenizer_path)

        return {'message': "Successfully created the model"}, 200
    except Exception as e:
        return {'message': str(e)}, 500

@app.route('/api/gensum/domain-specific', methods=['POST'])
def getDomainSpecificSummary():
    try:
        data = request.get_json()
        review = data['review']
        userId = data['userId']
        
        folder_path = 'model/' + userId
        model_path =  folder_path + '/' + MODEL_NAME
        tokenizer_path = folder_path + '/' + TOKENIZER_NAME

        if not os.path.exists(folder_path):
            return {'message': "Model not found"}, 404
        
        with open('encryption_key.key', 'rb') as file:
            key = file.read()
            
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
        fernet = Fernet(key)

        inputs = tokenizer.encode(review, return_tensors='pt', max_length=MAX_INPUT, truncation=True)
        outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        sentimentAnalysisOutput = query({ "inputs": summary })
        sentiment, score = getOverallSentimentWithScore(sentimentAnalysisOutput)
        score = round(score, 4)
        score = str(score)
        
        db.collection('users').document(userId).collection('reviews').add({
            'review': fernet.encrypt(review.encode()),
            'summary': fernet.encrypt(summary.encode()),
            'sentiment': fernet.encrypt(sentiment.encode()),
            'score': fernet.encrypt(score.encode()),
            'createdAt': firestore.SERVER_TIMESTAMP,
        })

        return {'summary': summary, 'sentiment': {
            'sentiment': sentiment,
            'score': score
        } }, 200
    except Exception as e:
        return {'message': str(e)}, 500


@app.route('/encryption', methods=['GET'])
def encrypt():
    try:
        message = "hello geeks"
        
        with open('encryption_key.key', 'rb') as file:
            key = file.read()
        
        fernet = Fernet(key)
        encMessage = fernet.encrypt(message.encode())
        decMessage = fernet.decrypt(encMessage).decode()

        print("original string: ", message)
        print("encrypted string: ", encMessage)
        print("decrypted string: ", decMessage)

        # Convert the bytes object to a base64-encoded string
        encMessageStr = encMessage.decode('utf-8')

        # Return a dictionary containing the encrypted and decrypted messages
        return json.dumps({'encrypted_text': encMessageStr, 'decrypted_text': decMessage}), 200
        
    except Exception as e:
        return {'message': str(e)}, 500

# creating an api endpoint to return a csv file with the review and summary data
@app.route('/api/gensum/review-records/user/<userId>/csv', methods=['GET'])
def getUserDataCSV(userId):
    try:
        print('user id', userId)
        user = db.collection('users').document(userId).get()
        with open('encryption_key.key', 'rb') as file:
            key = file.read()
        fernet = Fernet(key)
        
        if user.exists:
            reviews = db.collection('users').document(userId).collection('reviews').get()
            user = user.to_dict() 
            user['decrypted-reviews'] = []
            index = 1
            for review in reviews:
                summary = review.get('summary')
                reviewText = review.get('review')
                sentiment = review.get('sentiment')
                score = review.get('score')
                createdAt = review.get('createdAt')
                
                decodedSummary = fernet.decrypt(summary).decode()
                decodedReview = fernet.decrypt(reviewText).decode()
                decodedSentiment = fernet.decrypt(sentiment).decode()
                decodedScore = fernet.decrypt(score).decode()
                
                # Append the decoded review data to the user_reviews list
                user['decrypted-reviews'].append([index, decodedSummary, decodedReview, decodedSentiment, decodedScore, createdAt])
                index += 1

            # Create a CSV file
            with open('user_reviews.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Id', 'Summary', 'Review', 'Sentiment', 'Score', 'Created At'])
                writer.writerows(user['decrypted-reviews'])
                
            # Return the CSV file as a response
            with open('user_reviews.csv', 'rb') as f:
                csv_data = f.read()
                
            response = make_response(csv_data)
            response.headers['Content-Type'] = 'text/csv'
            response.headers['Content-Disposition'] = 'attachment; filename=user_reviews.csv'
            
            # Delete the CSV file
            try:
                file_path = 'user_reviews.csv'
                os.remove(file_path)
                print(f"{file_path} has been successfully deleted.")
            except OSError as e:
                print(f"Error deleting {file_path}: {e.strerror}")
            
            return response
        else:
            return {'message': "User not found"}, 404
    except Exception as e:
        return {'message': str(e)}, 500

@app.route('/api/gensum/db/write', methods=['POST'])
def handleWritingDataIntoDatabase():
    try:
        print('handling writing data into datbase')
        
        data = request.get_json()
        dataset_type = data['dataset_type']
        userId = data['userId']
        
        dataset_path = 'dataset/'
        
        if dataset_type == DOMAIN_TYPES["movies"]:
            dataset_path += 'movie_dataset.csv'
        elif dataset_type == DOMAIN_TYPES["resturant"]:
            dataset_path += 'resturant_dataset.csv'
        elif dataset_type == DOMAIN_TYPES["hotel"]:
            dataset_path += 'hotel_dataset.csv'
        elif dataset_type == DOMAIN_TYPES["ecommerce"]:
            dataset_path += 'ecommerce_dataset.csv'
        else:
            return {'message': "Invalid dataset type"}, 400

        df = pd.read_csv(dataset_path)
        df = df.drop(df.columns[0], axis=1)
        print('completed reading the database')
        
        user = db.collection('users').document(userId).get()
        if user.exists:
            with open('encryption_key.key', 'rb') as file:
                key = file.read()
            fernet = Fernet(key)
            numberOfDataRecords = 250
        
            dataset_array = df.iloc[:numberOfDataRecords].values  
            for row in dataset_array:
                db.collection('users').document(userId).collection('reviews').add({
                    'review': fernet.encrypt(row[0].encode()),
                    'summary': fernet.encrypt(row[1].encode()),
                    'createdAt': firestore.SERVER_TIMESTAMP,
                })
            return {'message': "Completed writing the data into the database"}, 200
        
            
        
    except Exception as e:
        return {'message': str(e)}, 500

@app.route('/api/gensum/retrain', methods=['POST'])
def retrainDomainSpecifcModel():
    try:
        data = request.get_json()
        newReviewSummaryData = []
        userId = data['userId'] 
        
        print('Finding user from database...')
        if not db.collection('users').document(userId).get().exists:
            return {'message': "User not found"}, 404
        
        userMetadata = db.collection('users').document(userId).get().to_dict()
        print('Get user metadata...', userMetadata)
        
        print('Getting receiver email...')
        email_receiver = userMetadata['email']
        
        print('Getting domain type...')
        domainType = userMetadata['type']
        
        isUseOtherData = data['isUseOtherData']
        print('Email trigger for retraining the model...')
        triggerEmailNotification("Retraining the gensum model", "Your model is preparing for retraining, you will be notified once the model is retrained", email_receiver)
        print('Notification sent to the user...')
        
        # Steps to be considered for retraining the model and dataset recreation
        # 1. By checking the isAccessible flag, we can decide whether to use the data for model retraining, then we get all the data from the database which isAccessible = true for the given domainType
        print('Fetching data from the database...')
        if isUseOtherData == True:
            users = db.collection('users').where('type', '==', domainType).where('isAccessible', '==', True).get()
            for user in users:
                reviews = db.collection('users').document(user.id).collection('reviews').get()
                for review in reviews:
                    newReviewSummaryData.append(review.to_dict())
        else:
            user = db.collection('users').document(userId).get()
            if user.exists:
                reviews = db.collection('users').document(userId).collection('reviews').get()
                for review in reviews:
                    newReviewSummaryData.append(review.to_dict())
            else:
                return {'message': "User not found"}, 404
        print('Successfully fetched data from the database')
        
        # 2. Decrypt all the data in the newReviewSummaryData
        print('Decrypting the fetched data...')
        decryptedReviewSummaryData = []
        with open('encryption_key.key', 'rb') as file:
            key = file.read()
        fernet = Fernet(key)
        
        for review in newReviewSummaryData:
            summary = review.get('summary')
            review = review.get('review')
            
            decodedSummary = fernet.decrypt(summary).decode()
            decodedReview = fernet.decrypt(review).decode()
            
            decryptedReviewSummaryData.append({
                'summary': decodedSummary,
                'review': decodedReview
            })

        # 3. We create another dataframe with the generalized dataset we have and then we append the new data to the dataframe
        print('Creating the dataframe using the fetched decrypted dataset...')
        new_df = pd.DataFrame(decryptedReviewSummaryData)
        print('Successfully created the dataframe')

        # 4. We then perform the necessary preprocessing steps on the data and then we create the dataset
        print('Preprocessing the dataset...')
        preprocess_dataset = handle_data_preprocessing(new_df)
        print('Completed data preprocessing')

        # 5. We can start model customization and hyperparameter tuning and model retraining
        print('Performing model customization and hyperparameter tuning and model retraining...')
        folder_path = 'model/' + userId
        model_path =  folder_path + '/' + MODEL_NAME
        tokenizer_path = folder_path + '/' + TOKENIZER_NAME

        if not os.path.exists(folder_path):
            return {'message': "Model not found"}, 404

        # model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
        
        print('preprocess_dataset content', preprocess_dataset)
        model_customization(preprocess_dataset, userId, model_path, tokenizer, db)
        print('completed model retraining...')
        
        triggerEmailNotification("Model retrained", "The model has been retrained, you can now use the model for summarization", email_receiver)
        return jsonify({'message': 'Success!'}), 200

    except Exception as e:
        triggerEmailNotification("Model retraining failed", "The model retraining has failed, please try again later", email_receiver)
        return {'message': str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True)
