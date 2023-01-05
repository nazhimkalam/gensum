import styled from "styled-components";
import About from "../../components/About/About";
import Footer from "../../components/Footer/Footer";
import Header from "../../components/Header/Header";

const LandingPage = () => {
  return (
    <StyledContainer>
      {/* header */}
      <Header />

      {/* title and about (given that it also gives the sentiment) */}
      <About />

      {/* how it works for both general users and domain specific users */}
      <h2>How it works</h2>

      {/* the summarizer tool to be used */}
      <h2>Summarizer</h2>

      <Footer />
    </StyledContainer>
  );
};

export default LandingPage;

const StyledContainer = styled.div`
  margin: 0;
`;
