import styled from "styled-components";

const HowItWorks = () => {
  return (
    <StyledContainer id="section">
      <h2>How gensum works?</h2>
      <p>
        User has the option to sign-in to the application as a domain user, or to also use the application as a general user without signing in. Domain users are allowed to retrain the model with their own data, and also manage their profile metadata. 
      </p>

      <p>General users are not allowed to retrain the model with their own data, and also not allowed to manage their profile metadata. However, both of the users will be able to perform the core functionality of the application which is to summarize the review text.</p>
    </StyledContainer>
  );
};

export default HowItWorks;

const StyledContainer = styled.div`
    margin: 0 5vw;
    padding: 2rem;

    h2 {
        font-size: 2rem;
        margin-bottom: 1rem;
    }

    p {
        margin: 0;
        padding: 0;
        text-align: justify;
        margin-bottom: 1rem;
    }
`;