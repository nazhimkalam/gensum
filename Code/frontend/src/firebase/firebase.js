import firebase from "firebase";

const firebaseConfig = {
  apiKey: "AIzaSyDQigpOaCm61f4NRM0mbBe23yzS8s3hGKA",
  authDomain: "gatot-7b39d.firebaseapp.com",
  projectId: "gatot-7b39d",
  storageBucket: "gatot-7b39d.appspot.com",
  messagingSenderId: "663262838201",
  appId: "1:663262838201:web:7daf761c62c64216200a68",
  measurementId: "G-J8F4ECXLBL",
};

const firebaseApp = firebase.initializeApp(firebaseConfig);
const db = firebaseApp.firestore();
const auth = firebase.auth();
const provider = new firebase.auth.GoogleAuthProvider();

export { db, provider, auth };
