import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import MatrixChatPage from './App';
import reportWebVitals from './reportWebVitals';

// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyB79qVrOyvw_z2Thw8cEyCmJ-KcLmtZyhs",
  authDomain: "reorganism-in.firebaseapp.com",
  databaseURL: "https://reorganism-in-default-rtdb.firebaseio.com",
  projectId: "reorganism-in",
  storageBucket: "reorganism-in.appspot.com",
  messagingSenderId: "319224134881",
  appId: "1:319224134881:web:2bf0384036f050da390ba0",
  measurementId: "G-HHZKPDSW6K"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <MatrixChatPage />
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
