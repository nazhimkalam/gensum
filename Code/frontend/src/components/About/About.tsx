import styled from "styled-components";

const About = () => {
  return (
    <StyledContainer id="about-section">
      <h2>About Gensum</h2>
      <p>
        Gensum is a generalized abstractive text summarization tool that can be used to summarize any english review texts, using optmizized deep learning algorithms (Transformers) and state-of-the-art NLP techniques. The tool is built using React, Typescript and Python. The backend is built using flask and the frontend using React framework. The model is built using Pytorch and Huggingface Transformers library. 
      </p>

      <p>
        The model is initlially generalized to the users and the users may determine to use it for any domain therefore it will adapat to the domain and improve its performance as the usage increases. The users will be able to retrain the model with the new data they have exposed this tool with, automated hyperparameter tuning will be also conducted during model retraing with the new data therefore the model will be able to adapt to the new domain and improve its performance.
      </p>

      <p>Apart from the core functionality, the sentiment of the summarized review will also be displayed along with the sentiment score. The user will also be able to view and manage (delete) the review text they input (only for domain users) therefore they decide which data needs to be used when model retraining this helps to preevnt model retraining with faulty data which would cause for the loss of performance.</p>

      <p>Domain users are also allowed to generate a csv of the results, which will be feteched from the database and also manage their profile metadata. Push notifications will be triggered to inform when model retraining completes and when each steps through the model retraining progress with.</p>
    </StyledContainer>
  );
};

export default About;

const StyledContainer = styled.div`
    margin: 0 5vw;
    margin-top: 2rem;
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