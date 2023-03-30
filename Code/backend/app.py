import csv
import json
import os
import smtplib
import ssl

import firebase_admin
import pandas as pd
import requests
import transformers
from cryptography.fernet import Fernet
from firebase_admin import credentials, firestore
from flask import Flask, make_response, request
from flask_cors import CORS
from flask_restful import Api
from utils.data_preprocessing import handle_data_preprocessing
from utils.model_retraining import hyperparameter_serach
from utils.types import DOMAIN_TYPES
from email.message import EmailMessage

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)
API = Api(app)

FIREBASE_CREDENTIAL_CERTIFICATE = 'firebase/gensumdb-firebase-adminsdk-twx36-8ad8a7b05c.json'
EMAIL_SENDER='nazhimkalamfyp@gmail.com'
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

def triggerEmailNotification(subject, body, email_reciever):
    email_ = EmailMessage()
    context = ssl.create_default_context()
    
    print("EMAIL_SENDER", EMAIL_SENDER)
    print("enail_passcode", enail_passcode)
    
    with open('email_passcode.key', 'rb') as file:
        enail_passcode = file.read()
        
    email_['From'] = EMAIL_SENDER
    email_['To'] = email_reciever
    email_['Subject'] = subject
    email_.set_content(body)
    
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(EMAIL_SENDER, enail_passcode)
        server.send_message(email_)
        
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
        
        print('user id', userId)

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
            # print("The length of the dataset is", len(dataset_array))
            # print("This is the content of the array", dataset_array)
            # print("This the columns of the dataset", df.info())
            for row in dataset_array:
                db.collection('users').document(userId).collection('reviews').add({
                    'review': fernet.encrypt(row[0].encode()),
                    'summary': fernet.encrypt(row[1].encode()),
                    'createdAt': firestore.SERVER_TIMESTAMP,
                })
                # i dont want to set but i want to append or add to the existing data
            # print("Completed writing the data into the database")
            return {'message': "Completed writing the data into the database"}, 200
        
            
        
    except Exception as e:
        return {'message': str(e)}, 500

@app.route('/api/gensum/retrain', methods=['POST'])
def retrainDomainSpecifcModel():
    try:
        data = request.get_json()
        newReviewSummaryData = []

        # The user id is only needed to save the model in the respective folder
        userId = data['userId'] 
        
        if not db.collection('users').document(userId).get().exists:
            return {'message': "User not found"}, 404
        
        email_receiver = db.collection('users').document(userId).get().to_dict()['email']
        
        # Using the domainType, we can get all the data from other users which have been given access for retraining
        domainType = db.collection('users').document(userId).get().to_dict()['type']
        # we can have a radio button in the frontend to select if the user wants to retrain only with their data or with the other users data as well
        isUseOtherData = data['isUseOtherData']
        triggerEmailNotification("Retraining the model", "The model is being retrained, you will be notified once the model is retrained", email_receiver)
        
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
        # # rename the 'review' into 'document' in decryptedReviewSummaryData
        # for review in decryptedReviewSummaryData:
        #     review['document'] = review.pop('review')

        new_df = pd.DataFrame(decryptedReviewSummaryData)
        # old_df = pd.read_csv(GENERALIZED_DATASET_PATH)
        # combined_df = pd.concat([new_df, old_df], axis=0) 
        print('Successfully created the dataframe')

        # 4. We then perform the necessary preprocessing steps on the data and then we create the dataset
        print('Preprocessing the dataset...')
        # Here we will include the data preprocessing steps taken
        preprocess_dataset = handle_data_preprocessing(new_df)
        print(preprocess_dataset.head(5))
        print('Completed data preprocessing')

        # 5. Since the new data is combined with the old data, we can start hyperparameter tuning and model retraining
        # 6. Once the hyperparameter tuning is done, we can save the model and tokenizer in the respective folder for the given userId
        # 7. Evaluation results needs to be stored in the database for the given userId and domainType for the model training
        print('Performing hyperparameter tuning and model retraining...')
        folder_path = 'model/' + userId
        model_path =  folder_path + '/' + MODEL_NAME
        tokenizer_path = folder_path + '/' + TOKENIZER_NAME

        if not os.path.exists(folder_path):
            return {'message': "Model not found"}, 404

        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
        hyperparameter_serach(new_df, userId, model, tokenizer, db)
        triggerEmailNotification("Model retrained", "The model has been retrained, you can now use the model for summarization", email_receiver)

    except Exception as e:
        return {'message': str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True)
