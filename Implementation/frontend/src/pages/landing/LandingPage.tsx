import styled from "styled-components";
import Footer from "../../components/Footer/Footer";
import Header from "../../components/Header/Header";

const LandingPage = () => {
  return (
    <StyledContainer>
      {/* header */}
      <Header />

      {/* title and about (given that it also gives the sentiment) */}

      {/* how it works for both general users and domain specific users */}

      {/* the summarizer tool to be used */}

      {/* footer */}
      <Footer />
    </StyledContainer>
  );
};

export default LandingPage;

const StyledContainer = styled.div`
  margin: 0;
`;
