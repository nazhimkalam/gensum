from flask_cors import CORS
from flask_restful import Api
from flask import Flask, request
import requests
import transformers
import pandas as pd
import os
import firebase_admin
from firebase_admin import firestore, credentials

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)
API = Api(app)

FIREBASE_CREDENTIAL_CERTIFICATE = 'firebase/gatot-7b39d-firebase-adminsdk-bqeyf-0366f4bf3a.json'
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
firestoreDb = firestore.client()

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

@app.route('/text-summarizer/general', methods=['POST'])
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

@app.route('/user/<userId>', methods=['GET'])
def getUserData(userId):
    try:
        user = firestoreDb.collection('domainUsers').document(userId).get()
        if user.exists:
            reviewData = firestoreDb.collection('domainUsers').document(userId).collection('reviewData').get()
            user = user.to_dict() 
            user['reviewData'] = []
            for review in reviewData:
                user['reviewData'].append(review.to_dict())
            return user, 200
        else:
            return {'message': "User not found"}, 404
    except Exception as e:
        return {'message': str(e)}, 500

@app.route('/domain-profile-creation', methods=['POST'])
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

@app.route('/text-summarizer/domain', methods=['POST'])
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

        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

        inputs = tokenizer.encode(review, return_tensors='pt', max_length=MAX_INPUT, truncation=True)
        outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        sentimentAnalysisOutput = query({ "inputs": summary })
        sentiment, score = getOverallSentimentWithScore(sentimentAnalysisOutput)

        firestoreDb.collection('domainUsers').document(userId).collection('reviewData').add({
            'review': review,
            'summary': summary,
            'sentiment': sentiment,
            'score': score
        })

        return {'summary': summary, 'sentment': {
            'sentiment': sentiment,
            'score': score
        } }, 200
    except Exception as e:
        return {'message': str(e)}, 500



@app.route('/domain-profile-retraining', methods=['POST'])
def retrainDomainSpecifcModel():
    try:
        data = request.get_json()
        newReviewSummaryData = []

        userId = data['userId'] # The user id is only needed to save the model in the respective folder
        domainType = data['domainType'] # Using the domainType, we can get all the data from other users which have been given access for retraining
        isUseOtherData = data['isUseOtherData'] # we can have a radio button in the frontend to select if the user wants to retrain only with their data or with the other users data as well

        # Steps to be considered for retraining the model and dataset recreation
        # 1. By checking the isAccessible flag, we can decide whether to use the data for model retraining, then we get all the data from the database which isAccessible = true for the given domainType
        print('Fetching data from the database...')
        if isUseOtherData == True:
            users = firestoreDb.collection('domainUsers').where('domainType', '==', domainType).where('isAccessible', '==', True).get()
            for user in users:
                reviewData = firestoreDb.collection('domainUsers').document(user.id).collection('reviewData').get()
                for review in reviewData:
                    newReviewSummaryData.append(review.to_dict())

        else:
            user = firestoreDb.collection('domainUsers').document(userId).get()
            if user.exists:
                reviewData = firestoreDb.collection('domainUsers').document(userId).collection('reviewData').get()
                for review in reviewData:
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

        return {'message': "Successfully triggered the model, this will take a while for completion and will automatically update to the latest model", "response": newReviewSummaryData }, 200
    except Exception as e:
        return {'message': str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True)
