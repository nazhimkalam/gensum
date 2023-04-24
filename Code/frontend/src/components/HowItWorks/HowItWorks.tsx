import { Image } from "antd";
import styled from "styled-components";

const HowItWorks = () => {
  return (
    <StyledContainer id="section">
      <h2>How gensum works?</h2>
      <main>
        <section>
          <p>
            User has the option to sign-in to the application as a domain user,
            or to also use the application as a general user without signing in.
            Domain users are allowed to retrain the model with their own data,
            and also manage their profile metadata.
          </p>

          <p>
            General users are not allowed to retrain the model with their own
            data, and also not allowed to manage their profile metadata.
            However, both of the users will be able to perform the core
            functionality of the application which is to summarize the review
            text.
          </p>
        </section>
        <section>
          <Image
            src="https://gatherup.com/wp-content/uploads/2021/01/google-review-summaries.png"
            alt="logo"
            title="Antagonism"
            preview={false}
          />
          <Image
            src="https://www.shelfstore.co.uk/blog/wp-content/uploads/2015/11/google-reviews-shelfstore.png"
            alt="logo"
            title="Antagonism"
            preview={false}
          />
          <Image
            src="https://digimind.id/wp-content/uploads/Picture7-20.png"
            alt="logo"
            title="Antagonism"
            preview={false}
          />
        </section>
      </main>
    </StyledContainer>
  );
};

export default HowItWorks;

const StyledContainer = styled.div`
  margin: 0 5vw;
  padding: 2rem;

  transform: translateY(-100%);
  animation: slideIn 0.5s forwards;
  
  @keyframes slideIn {
    100% {
      transform: translateY(0%);
    }
  }

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
    flex-direction: column;
    justify-content: space-evenly;
    flex-wrap: wrap;

    section {
      :last-child {
        display: flex;
        justify-content: space-evenly;
        margin: 3rem 0;
        width: 100%;
      }
      img {
        padding: 2rem;
        object-fit: contain;
        height: 280px !important;
        box-shadow: 0 0 10px 0 rgba(0, 0, 0, 0.2);
      }
    }
  }
`;
