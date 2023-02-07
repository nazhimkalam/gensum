import styled from "styled-components";

const Login = () => {
  const onHandleLogin = () => {};
  return (
    <StyledContainer>
      <button onClick={onHandleLogin}>Login using Google</button>
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
