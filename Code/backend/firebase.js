import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";

const firebaseConfig = {
  apiKey: "AIzaSyDlBMB_ESMwqVNPwfJYwsBch8AE2ggMIas",
  authDomain: "gensumdb.firebaseapp.com",
  projectId: "gensumdb",
  storageBucket: "gensumdb.appspot.com",
  messagingSenderId: "822063654597",
  appId: "1:822063654597:web:7c8eca0c3d3c508d6732ac",
  measurementId: "G-5E0E3MS389"
};

const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);