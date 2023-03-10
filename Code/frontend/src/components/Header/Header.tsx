import styled from "styled-components";
import { Image, Typography, notification } from "antd";
import { useNavigate } from "react-router-dom";
import { routePaths } from "../../app/routes";
import { useDispatch, useSelector } from "react-redux";
import { auth, db, provider } from "../../firebase/firebase.js";
import { login, logout, selectUser } from "../../redux/reducers/userReducer";
import type { MenuProps } from "antd";
import { Button, Dropdown, Space } from "antd";

const Header = () => {
  const { Title } = Typography;
  const navigate = useNavigate();
  const disptach = useDispatch();
  const user = useSelector(selectUser);

  const triggerNotification = (title: string, message: string) => {
    notification.open({ message: title, description: message, placement: "bottomRight" });
  };

  const handleUserLogout = () => {
    auth.signOut().then(() => {
      disptach(logout());
    });
  };

  const navigateToHome = () => {
    navigate(routePaths.home);
  };

  const handleSignIn = () => {
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
                contactNumber: "",
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

  const navigateToProfile = () => { 
    navigate(routePaths.profile);
  };

  const items: MenuProps["items"] = [
    {
      key: "1",
      label: <p onClick={navigateToProfile}>Manage Profile</p>,
    },
    {
      key: "2",
      label: <p onClick={handleUserLogout}>Logout</p>,
    },
  ];

  return (
    <Container>
      <section className="section-one">
        <ImageContainer onClick={() => navigateToHome()}>
          <Image
            src="logos/logo1.PNG"
            alt="logo"
            title="Antagonism"
            preview={false}
          />
          <Title level={3}>Gensum</Title>
        </ImageContainer>
      </section>
      <section className="section-two">
        {/* {user?.id && <Button style={{ backgroundColor: 'black', color: 'white' }} onClick={() => handleSignIn()}>Edit Profile</Button>} */}
        {!user?.id && (
          <Button
            style={{ backgroundColor: "black", color: "white" }}
            onClick={() => handleSignIn()}
          >
            Sign In
          </Button>
        )}
        {user?.id && (
          <Button
            onClick={() => navigate(routePaths.records)}
            style={{ backgroundColor: "black", color: "white" }}
          >
            Reviews
          </Button>
        )}
        {/* {user?.id && <Button onClick={() => handleUserLogout()}>Logout</Button>} */}
        {user?.id && (
          <Space direction="vertical">
            <Space wrap>
              <Dropdown menu={{ items }} placement="bottom">
                <Button style={{ border: "1px black solid" }}>Profile</Button>
              </Dropdown>
            </Space>
          </Space>
        )}
      </section>
    </Container>
  );
};

export default Header;

const Container = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0px 5vw;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12), 0 1px 2px rgba(0, 0, 0, 0.24);

  .section-two {
    display: flex;
    align-items: center;
    justify-content: space-between;

    button {
      margin-left: 10px;
      height: 40px;

      &:hover {
        background-color: black;
        color: white;
      }
    }
  }
`;

const ImageContainer = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  img {
    width: 80px !important;
    cursor: pointer;
    object-fit: contain;
  }
`;
