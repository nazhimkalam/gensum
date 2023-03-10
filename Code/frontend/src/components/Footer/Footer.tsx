import styled from "styled-components";

const Footer = () => {
  return (
    <FooterStyled>
      <p>@nazhimkalam</p>
      <p>
        <a href="/">Terms of Service</a> | <a href="/">Privacy Policy</a>
      </p>

      <p>
        <a href="/">Facebook</a> | <a href="/">Twitter</a>
      </p>
    </FooterStyled>
  );
};

export default Footer;

const FooterStyled = styled.div`
  background-color: #000;
  color: #fff;
  text-align: center;
  margin-top: 2rem;
  p {
    margin: 0 5vw;
  }

  a {
    color: #fff;
    text-decoration: none;
  }

  a:hover {
    text-decoration: underline;
  }

  @media (min-width: 768px) {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  @media (min-width: 1024px) {
    padding: 2rem;
  }
`;
