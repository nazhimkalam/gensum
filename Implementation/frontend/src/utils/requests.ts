import axios from "axios";

export const postRequest = (api: string, body: object, token: string | undefined | null, removeHeader?: boolean) =>
  axios
    .post(api, body, { headers: { 'Content-Type': !removeHeader && 'application/json', 'Authorization': `Bearer ${token}` } })
    .then((response) => response)
    .catch((error) => error);

export const getRequest = (api: string, token: string | undefined | null) =>
  axios
    .get(api, { headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` } })
    .then((response) => response)
    .catch((error) => error);

export const updateRequest = (api: string, body: object, token: string | undefined | null) =>
  axios
    .put(api, body, { headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` } })
    .then((response) => response)
    .catch((error) => error);

export const deleteRequest = (api: string, token: string | undefined | null) =>
  axios
    .delete(api, { headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` } })
    .then((response) => response)
    .catch((error) => error);

export const patchRequest = (api: string, body: object, token: string | undefined | null) =>
  axios
    .patch(api, body, { headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` } })
    .then((response) => response)
    .catch((error) => error);