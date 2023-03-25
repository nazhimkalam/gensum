import { gensumApi } from "../apis/gensumApi";
import { getRequest, postRequest } from "../utils/requests";

export const getReviewsByUserId = async (userId: string) =>
  await getRequest(`${gensumApi.reviewRecords}/user/${userId}`)
    .then((response) => {
      if (response.status === 200 || response.status === 201)
        return response.data;
      throw response;
    })
    .catch((error) => {
      throw error;
    });

export const triggerModelRetrain = async (userId: string, type: number, isUseOtherData: boolean) => 
  await postRequest(`${gensumApi.modelRetraining}`, {userId, type, isUseOtherData})
    .then((response) => {
      if (response.status === 200 || response.status === 201)
        return response.data;
      throw response;
    })
    .catch((error) => {
      throw error;
    });