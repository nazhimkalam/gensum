import styled from "styled-components";
import About from "../../components/About/About";
import Footer from "../../components/Footer/Footer";
import Header from "../../components/Header/Header";
import HowItWorks from "../../components/HowItWorks/HowItWorks";
import Summarizer from "../../components/Summarizer/Summarizer";
import ScrollToTop from "react-scroll-to-top";

const LandingPage = () => {
  return (
    <StyledContainer>
      <Header />
      <About />
      <HowItWorks />
      <Summarizer />
      <ScrollToTop smooth />
      <Footer />
    </StyledContainer>
  );
};

export default LandingPage;

const StyledContainer = styled.div`
  margin: 0;
`;
