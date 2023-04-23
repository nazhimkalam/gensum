import styled from "styled-components";
import { useNavigate } from "react-router-dom";
import { routePaths } from "../../app/routes";
import { useDispatch, useSelector } from "react-redux";
import { auth, db, provider } from "../../firebase/firebase.js";
import { login, logout, selectUser } from "../../redux/reducers/userReducer";
import type { MenuProps } from "antd";
import { Button, Dropdown, Space, Modal, Image, Typography, notification } from "antd";
import { ArrowDownOutlined } from "@ant-design/icons";
import { useState } from "react";
import { postRequest } from "../../utils/requests";
import { gensumApi } from "../../apis/gensumApi";
import { domainModelInitilization } from "../../services/gensum.service";

const Header = () => {
  const { Title } = Typography;
  const navigate = useNavigate();
  const disptach = useDispatch();
  const user = useSelector(selectUser);
  
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [modalText] = useState('Select if you want to retrain the model other business common domain data aswell or only with your business domain data...');

  const triggerNotification = (title: string, message: string) => {
    notification.open({ message: title, description: message, placement: "bottomRight" });
  };

  const showModal = () => {
    setOpen(true);
  };

  const handleOk = () => {
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
      setOpen(false);
    }, 3000);
  };

  const handleCancel = () => {
    setOpen(false);
  };

  const handleUserLogout = () => {
    auth.signOut().then(() => {
      disptach(logout());
      triggerNotification("Success", "You have been successfully logged out");
    });
  };

  const navigateToHome = () => {
    navigate(routePaths.home);
  };

  const handleSignIn = async () => {
    auth
      .signInWithPopup(provider)
      .then((result: any) => {
        db.collection("users").doc(result.user.uid).get().then(async(doc: any) => {
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
              await domainModelInitilization(result.user.uid);
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
        triggerNotification("Success", "You have been successfully logged in, naviagte to the manage profile to update your profile information");
      })
      .catch((error: any) => {
        console.log(error.message);
        triggerNotification("Error", "Error occurred, please refresh the page");
      });
  };

  const navigateToProfile = () => { 
    navigate(routePaths.profile);
  };

  const navigateToReviewsPage = () => {
    navigate(routePaths.records);
  }

  const items: MenuProps["items"] = [
    {
      key: "1",
      label: <p onClick={navigateToReviewsPage}>Handle Reviews</p>,
    },
    {
      key: "2",
      label: <p onClick={navigateToProfile}>Manage Profile</p>,
    },
    {
      key: "3",
      label: <p onClick={handleUserLogout}>Logout</p>,
    },
  ];

  const handleModelRetrain = async (isUseOtherData: boolean) => {
    const userId = user?.id;

    if (userId) {
      const apiEndpoint = gensumApi.modelRetraining;
      const requestBody = {
        userId, isUseOtherData
      }
      await postRequest(apiEndpoint, requestBody);
      triggerNotification("Success", "Model retraining request has been sent, please wait for the email notification");
    }
  }

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
        {!user?.id && (
          <Button
            style={{ backgroundColor: "#880ED4", color: "white" }}
            onClick={() => handleSignIn()}
          >
            Sign In
          </Button>
        )}
        {user?.id && (
          <Button
            // onClick={() => navigate(routePaths.records)}
            onClick={showModal}
            style={{ backgroundColor: "#880ED4", color: "white" }}
          >
            Model Retrain
          </Button>
        )}
        {user?.id && (
          <Space direction="vertical">
            <Space wrap>
              <Dropdown menu={{ items }} placement="bottom">
                <Button style={{ border: "1px #880ED4 solid" }}>My Profile <ArrowDownOutlined /></Button>
              </Dropdown>
            </Space>
          </Space>
        )}
      </section>
      <Modal
        open={open}
        title="Model Retrain"
        onOk={handleOk}
        onCancel={handleCancel}
        footer={[
          <Button key="use-other-data" onClick={() => {
            handleModelRetrain(true);
            setOpen(false);
          }}>
            All businesses data
          </Button>,
          <Button key="use-my-data" type="primary" loading={loading} onClick={() => {
            handleModelRetrain(false);
            setOpen(false);
          }}>
            My business data
        </Button>,
        <Button key="back" type="link" onClick={handleCancel}>
          Cancel 
        </Button>,
         ]}
      >
        <p>{modalText}</p>
      </Modal>
    </Container>
  );
};

export default Header;

const Container = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0px 5vw;
  box-shadow: 0 1px 3px rgba(178, 75, 244, 0.12), 0 1px 2px rgba(178, 75, 244, 0.24);

  .section-two {
    display: flex;
    align-items: center;
    justify-content: space-between;

    button {
      margin-left: 10px;
      height: 40px;

      &:hover {
        background-color: #880ED4;
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
