import { Button, notification, Input } from "antd";
import { useState } from "react";
import { useSelector } from "react-redux";
import styled from "styled-components";
import { gensumApi } from "../../apis/gensumApi";
import { selectUser } from "../../redux/reducers/userReducer";
import { postRequest } from "../../utils/requests";

const Summarizer = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [textareaContent, setTextareaContent] = useState("");
  const [result, setResult] = useState({});
  const user = useSelector(selectUser);
  const { TextArea } = Input;

  const triggerNotification = (title, message) => {
    notification.open({
      message: title,
      description: message,
      placement: "bottomRight",
    });
  };

  const handleScan = async () => {
    if (!textareaContent) {
      triggerNotification("Error", "Please enter text to scan");
      return;
    }
    setResult(undefined);
    setIsLoading(true);

    const apiEndpoint =
      user.id === undefined
        ? gensumApi.generalSummarization
        : gensumApi.domainSpecificSummarization;
    const summarizeRequestBody =
      user.id === undefined
        ? { review: textareaContent }
        : { review: textareaContent, userId: user.id };

    await postRequest(apiEndpoint, summarizeRequestBody)
      .then((response) => {
        console.log(response.data);
        setResult({
          review: textareaContent,
          summary: response.data.summary,
          sentiement: response.data.sentiment.sentiment,
          score: response.data.sentiment.score,
        });
      })
      .catch((error) => {
        console.log(error);
        triggerNotification(
          "Error",
          "Something went wrong when scanning the text"
        );
      })
      .finally(() => setIsLoading(false));
  };

  const handleReset = () => {
    setTextareaContent("");
    setResult(undefined);
  };

  return (
    <StyledContainer>
      {!isLoading ? (
        <TextArea
          rows={10}
          disabled={isLoading}
          value={textareaContent}
          onChange={(e) => setTextareaContent(e.target.value)}
          placeholder="Enter review here..."
        />
      ) : (
        <img src="images/loading.gif" alt="loading" />
      )}
      {!isLoading && (
        <section className="buttons">
          <Button
            className="scan-button"
            onClick={handleScan}
            disabled={isLoading}
          >
            {" "}
            Summarize{" "}
          </Button>
          <Button
            className="reset-button"
            onClick={handleReset}
            disabled={isLoading}
          >
            {" "}
            Reset{" "}
          </Button>
        </section>
      )}

      <br />
      {result?.summary && (
        <>
          <div className="detection-result summary">
            <h3>Initial review:</h3> <p>{result.review}</p>
          </div>
          <br />

          <div className="detection-result summary">
            <h3>Summarized review:</h3> <p>{result.summary}</p>
          </div>
          <br />

          <div className="detection-result sentiment">
            <h3>Review sentiment:</h3> <p>{result.sentiement}</p>
          </div>
          <br />

          <div className="detection-result sentiment">
            <h3>Review sentiment score:</h3>{" "}
            <p>{Math.round(result.score * 10000) / 100 + "%"}</p>
          </div>
          <br />
        </>
      )}
    </StyledContainer>
  );
};

export default Summarizer;

const StyledContainer = styled.div`
  display: flex;
  flex-direction: column;
  justify-content: center;
  margin: 0 5vw;
  padding: 20px;

  transform: translateY(-250%);
  animation: slideIn 0.5s forwards;
  
  @keyframes slideIn {
    100% {
      transform: translateY(0%);
    }
  }

  img {
    object-fit: contain;
    height: 150px;
    margin: 5pc auto;
  }
  p {
    text-align: justify;
  }
  textarea {
    padding: 10px;
    border: 1px solid #880ed4;
    border-radius: 5px;
    resize: none;
    box-shadow: 0 0 5px 0 rgba(0, 0, 0, 0.3);
    font-size: 16px;
    &:focus {
      outline: none;
    }
  }
  .buttons {
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 20px 0;
    button {
      box-shadow: 0 0 5px 0 rgba(0, 0, 0, 0.2);
      height: 40px;
      border: none;
      border-radius: 5px;
      font-size: 16px;
      width: 150px;
      cursor: pointer;
      &:focus {
        outline: none;
      }
      &.scan-button {
        background-color: #880ed4;
        color: #fff;
        margin-right: 10px;
      }
      &.reset-button {
        background-color: #fff;
        color: #880ed4;
        border: 1px solid #880ed4;
      }
    }
  }
  .detection-result summary {
    margin-top: 20px;
    h3 {
      margin-bottom: 10px;
    }
  }
`;
