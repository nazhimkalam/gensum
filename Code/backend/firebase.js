// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyDQigpOaCm61f4NRM0mbBe23yzS8s3hGKA",
  authDomain: "gatot-7b39d.firebaseapp.com",
  projectId: "gatot-7b39d",
  storageBucket: "gatot-7b39d.appspot.com",
  messagingSenderId: "663262838201",
  appId: "1:663262838201:web:7daf761c62c64216200a68",
  measurementId: "G-J8F4ECXLBL"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);