import firebase from "firebase";

const firebaseConfig = {
  apiKey: "AIzaSyDlBMB_ESMwqVNPwfJYwsBch8AE2ggMIas",
  authDomain: "gensumdb.firebaseapp.com",
  projectId: "gensumdb",
  storageBucket: "gensumdb.appspot.com",
  messagingSenderId: "822063654597",
  appId: "1:822063654597:web:7c8eca0c3d3c508d6732ac",
  measurementId: "G-5E0E3MS389"
};

const firebaseApp = firebase.initializeApp(firebaseConfig);
const db = firebaseApp.firestore();
const auth = firebase.auth();
const provider = new firebase.auth.GoogleAuthProvider();

export { db, provider, auth };
