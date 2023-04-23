import { Image } from "antd";
import styled from "styled-components";

const About = () => {
  return (
    <StyledContainer id="about-section">
      <h2>About Gensum</h2>
      <main>
        <section>
          <p>
          Gensum is a tool for abstractive text summarization of English review texts, utilizing advanced NLP techniques and optimized deep learning algorithms (Transformers) built with Python, Pytorch, Huggingface Transformers library, React, and Typescript. Its backend is created using Flask while its frontend is built with React.
          </p>

          <p>
          Initially, the model is designed to be adaptable to any domain and will improve its performance as it is used. Users can also retrain the model with their own data and automated hyperparameter tuning will be conducted during the retraining process. This enables the model to adapt to new domains and improve its performance.
          </p>

          <p>
          In addition to the main function, the tool also displays the sentiment of the summarized review, including the sentiment score. For domain users, they can view and delete the review text they input, allowing them to decide which data to use when retraining the model. This helps prevent retraining the model with faulty data, which could result in a loss of performance.
          </p>

          <p>
          Domain users have the additional ability to generate a CSV file of the results fetched from the database, as well as manage their profile metadata. They will receive push notifications to inform them when the model retraining completes, as well as updates throughout the retraining progress.
          </p>
        </section>
        <section>
          <Image
            src="https://www.intelligencerefinery.io/images/post/extractive-summary.png"
            alt="logo"
            title="Antagonism"
            preview={false}
          />
        </section>
      </main>
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
    color: #880ED4;
  }

  p {
    margin: 0;
    padding: 0;
    text-align: justify;
    margin-bottom: 1rem;
  }

  main {
    display: flex;
    align-items: center;
    justify-content: space-between;
    
    section {
      img {
        margin-left: 3rem;
        object-fit: contain;
      }
    }
  }
`;
