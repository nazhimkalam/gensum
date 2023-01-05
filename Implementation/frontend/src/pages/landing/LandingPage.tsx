import styled from "styled-components";
import About from "../../components/About/About";
import Footer from "../../components/Footer/Footer";
import Header from "../../components/Header/Header";
import HowItWorks from "../../components/HowItWorks/HowItWorks";
import Summarizer from "../../components/Summarizer/Summarizer";

const LandingPage = () => {
  return (
    <StyledContainer>
      <Header />

      {/* title and about (given that it also gives the sentiment) */}
      <About />

      {/* how it works for both general users and domain specific users */}
      <HowItWorks />

      {/* the summarizer tool to be used */}
      <Summarizer />

      <Footer />
    </StyledContainer>
  );
};

export default LandingPage;

const StyledContainer = styled.div`
  margin: 0;
`;
