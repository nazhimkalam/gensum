import styled from "styled-components";
import { Image, Typography } from "antd";
import { useNavigate } from "react-router-dom";
import { routePaths } from "../../app/routes";

const Header = () => {
  const { Title } = Typography;
  const navigate = useNavigate();

  const navigateToHome = () => {
    navigate(routePaths.home);
  };
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
          <Title level={3}>Gatot</Title>
        </ImageContainer>
      </section>
      <section className="section-two">
        <button onClick={() => navigate(routePaths.register)}>
          Create an account
        </button>
        <button onClick={() => navigate(routePaths.login)}>Login</button>
      </section>
    </Container>
  );
};

export default Header;

const Container = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 20px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12), 0 1px 2px rgba(0, 0, 0, 0.24);

  .section-two {
    display: flex;
    align-items: center;
    justify-content: space-between;

    button {
      padding: 10px 20px;
      border: 1px solid black;
      border-radius: 5px;
      cursor: pointer;
      background-color: white;
      color: black;
      font-weight: bold;
      transition: all 0.2s ease-in-out;
      &:hover {
        background-color: black;
        color: white;
      }
      &:first-child {
        margin-right: 10px;
        background-color: black;
        color: white;

        &:hover {
          background-color: white;
          color: black;
        }
      }
      &:last-child {
        margin-left: 10px;
        width: 100px;
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
