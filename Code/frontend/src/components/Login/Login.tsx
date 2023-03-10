import styled from "styled-components";
import { useDispatch } from "react-redux";
import { auth, db, provider } from "../../firebase/firebase.js";
import { login } from "../../redux/reducers/userReducer";
import { notification } from "antd";

const Login = () => {
  const disptach = useDispatch();
  const triggerNotification = (title: string, message: string) => {
    notification.open({ message: title, description: message, placement: "bottomRight" });
  };

  const onHandleLogin = () => {
    auth
    .signInWithPopup(provider)
    .then((result: any) => {
      db.collection("users")
        .doc(result.user.uid)
        .get()
        .then((doc: any) => {
          if (doc.exists) {
            console.log("User already exists");
          } else {
            db.collection("users").doc(result.user.uid).set({
              email: result.user.email,
              name: "",
              type: 0,
              isAccessible: false,
            });

            console.log("User created");
          }
        })
        .catch((error: any) => {
          console.log("Error getting document:", error);
        });

      disptach(
        login({
          id: result.user.uid,
          displayName: result.user.displayName,
          email: result.user.email,
        })
      );
      triggerNotification("Success", "You have been successfully registered, please naviagte to the edit profile to update your profile information");
    })
    .catch((error: any) => {
      console.log(error.message);
      triggerNotification("Error", "Error occurred, please refresh the page");
    });
  };
  return (
    <StyledContainer>
      <button onClick={onHandleLogin}>Sign In using Google</button>
    </StyledContainer>
  );
};

export default Login;

const StyledContainer = styled.div`
  height: 70vh;
  border-radius: 5px;

  display: grid;
  place-items: center;

  button {
    background-color: #fff;
    border: 1px solid #000;
    border-radius: 5px;
    padding: 10px 20px;
    font-size: 16px;
    cursor: pointer;
    &:focus {
      outline: none;
    }
    &:hover {
      background-color: #000;
      color: #fff;
    }
  }
`;
