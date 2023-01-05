import styled from "styled-components";
import { useDispatch } from "react-redux";
import { auth, db, provider } from "../../firebase/firebase.js";
import { login } from "../../redux/reducers/userReducer";

const Register = () => {
  const disptach = useDispatch();

  const onHandleRegister = async () => {
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
                domainName: "",
                domainType: 0,
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
        alert(
          "You have been successfully registered, please naviagte to the edit profile to update your profile information"
        );
      })
      .catch((error: any) => {
        console.log(error.message);
        alert("Error occurred, please refresh the page");
      });
  };
  return (
    <StyledContainer>
      <button onClick={onHandleRegister}>Register using Google</button>
    </StyledContainer>
  );
};

export default Register;

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
