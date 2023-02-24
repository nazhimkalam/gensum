import { gensumApi } from "../apis/gensumApi";
import { getRequest } from "../utils/requests";

export const getReviewsByUserId = async (userId: string) =>
  await getRequest(`${gensumApi.reviewRecords}/user/${userId}`)
    .then((response) => {
    //   console.log(response);
      if (response.status === 200 || response.status === 201)
        return response.data;
      throw response;
    })
    .catch((error) => {
      throw error;
    });
