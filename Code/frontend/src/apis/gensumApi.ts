
export const BASE_URL = 'http://localhost:5000';

export const gensumApi = {
    generalSummarization: `${BASE_URL}/api/gensum/general`,
    domainSpecificSummarization: `${BASE_URL}/api/gensum/domain-specific`,
    domainProfileCreation: `${BASE_URL}/api/gensum/domain-profile`,
    modelRetraining: `${BASE_URL}/api/gensum/retrain`,
    reviewRecords: `${BASE_URL}/api/gensum/review-records`,
    generateReviewCsv: `${BASE_URL}/api/gensum/review-records/user`,
}