from flask_cors import CORS
from flask_restful import Api
from flask import Flask, request, make_response
import requests
import transformers
import pandas as pd
import os
import csv
from jsonify import convert
import json
from cryptography.fernet import Fernet
import firebase_admin
from firebase_admin import firestore, credentials
from utils.model_retraining import hyperparameter_serach

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)
API = Api(app)

FIREBASE_CREDENTIAL_CERTIFICATE = 'firebase/gensumdb-firebase-adminsdk-twx36-8ad8a7b05c.json'
MODEL_NAME = 'bart-base_model'
TOKENIZER_NAME = 'bart-base_tokenizer'
GENERALIZED_DATASET_PATH = 'dataset/xsum.csv'
GENERALIZED_MODEL_PATH = 'model/base/' + MODEL_NAME
GENERALIZED_TOKENIZER_PATH = 'model/base/' + TOKENIZER_NAME
MAX_INPUT = 512

HUGGING_FACE_BEARER_TOKEN = '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'
SENTIMENTAL_ANALYSIS_HUGGING_FACE_API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"

generalized_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(GENERALIZED_MODEL_PATH)
generalized_tokenizer = transformers.AutoTokenizer.from_pretrained(GENERALIZED_TOKENIZER_PATH)

firebase_credentials = credentials.Certificate(FIREBASE_CREDENTIAL_CERTIFICATE)
firebaseApp = firebase_admin.initialize_app(firebase_credentials)
db = firestore.client()

def query(payload):
    headers = {"Authorization": "Bearer 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"}
    response = requests.post(SENTIMENTAL_ANALYSIS_HUGGING_FACE_API_URL, headers=headers, json=payload)
    return response.json()

def getOverallSentimentWithScore(sentiment):
    positiveScore = sentiment[0][0]['score']
    negativeScore = sentiment[0][1]['score']
    if positiveScore > negativeScore:
        return 'Positive', positiveScore
    else:
        return 'Negative', negativeScore
                
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
        return {'summary': summary, 'sentment': {
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
                    'score': decodedScore,
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

        return {'summary': summary, 'sentment': {
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
            return response
        else:
            return {'message': "User not found"}, 404
    except Exception as e:
        return {'message': str(e)}, 500

@app.route('/api/gensum/retrain', methods=['POST'])
def retrainDomainSpecifcModel():
    try:
        data = request.get_json()
        newReviewSummaryData = []

        userId = data['userId'] # The user id is only needed to save the model in the respective folder
        domainType = data['type'] # Using the domainType, we can get all the data from other users which have been given access for retraining
        isUseOtherData = data['isUseOtherData'] # we can have a radio button in the frontend to select if the user wants to retrain only with their data or with the other users data as well

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


        # 2. We create another dataframe with the generalized dataset we have and then we append the new data to the dataframe
        print('Creating the dataframe using the fetched dataset...')
        # rename the 'review' into 'document' in newReviewSummaryData
        for review in newReviewSummaryData:
            review['document'] = review.pop('review')

        new_df = pd.DataFrame(newReviewSummaryData)
        # old_df = pd.read_csv(GENERALIZED_DATASET_PATH)
        # combined_df = pd.concat([new_df, old_df], axis=0) 
        print('Successfully created the dataframe')

        # 3. We then perform the necessary preprocessing steps on the data and then we create the dataset
        print('Preprocessing the dataset...')
        # Here we will include the data preprocessing steps taken
        print('Completed data preprocessing')

        # 4. Since the new data is combined with the old data, we can start hyperparameter tuning and model retraining
        # 5. Once the hyperparameter tuning is done, we can save the model and tokenizer in the respective folder for the given userId
        # 6. Evaluation results needs to be stored in the database for the given userId and domainType for the model training
        print('Performing hyperparameter tuning and model retraining...')
        folder_path = 'model/' + userId
        model_path =  folder_path + '/' + MODEL_NAME
        tokenizer_path = folder_path + '/' + TOKENIZER_NAME

        if not os.path.exists(folder_path):
            return {'message': "Model not found"}, 404

        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
        hyperparameter_serach(new_df, userId, model, tokenizer, db)

        return {'message': "Successfully triggered the model, this will take a while for completion and will automatically update to the latest model", "response": newReviewSummaryData }, 200
    except Exception as e:
        return {'message': str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True)
