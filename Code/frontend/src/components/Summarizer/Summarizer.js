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
    notification.open({ message: title, description: message, placement: "bottomRight" });
  };

  const handleScan = async () => {
    if (!textareaContent) {
      triggerNotification("Error", "Please enter text to scan");
      return;
    }
    setIsLoading(true);
    
    const apiEndpoint = user.id === undefined ? gensumApi.generalSummarization : gensumApi.domainSpecificSummarization;
    const summarizeRequestBody = user.id === undefined ? { review: textareaContent } : { review: textareaContent, userId: user.id };
      
    await postRequest(apiEndpoint, summarizeRequestBody).then((response) => {
      console.log(response.data)
      setResult({
        review: textareaContent,
        summary: response.data.summary,
        sentiement: response.data.sentiment.sentiment,
        score: response.data.sentiment.score
      })
    }).catch((error) => {
      console.log(error)
      triggerNotification("Error", "Something went wrong when scanning the text");
    }).finally(() => setIsLoading(false));
  };

  const handleReset = () => {
    setTextareaContent("");
    setResult(undefined);
  };

  return (
    <StyledContainer>
      {!isLoading ? (
        <TextArea rows={10} disabled={isLoading} value={textareaContent} onChange={(e) => setTextareaContent(e.target.value)} placeholder="Enter review here..."/>
      ): (
        <img src="images/loader.gif" alt="loading"/>
      )}
      {!isLoading && <section className="buttons">
        <Button className="scan-button" onClick={handleScan} disabled={isLoading}> Summarize </Button>
        <Button className="reset-button" onClick={handleReset} disabled={isLoading}> Reset </Button>
      </section>}

      {result?.summary && (
        <>
          <div className="detection-result summary">
            <h3>Initial review:</h3> {result.review}
          </div>
          <br />
          
          <div className="detection-result summary">
            <h3>Summarized review:</h3> {result.summary}
          </div>
          <br />
        
          <div className="detection-result sentiment">
            <h3>Review sentiment:</h3> {result.sentiement}
          </div>
          <br />

          <div className="detection-result sentiment">
            <h3>Review sentiment score:</h3> {result.score}
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

    img {
      object-fit: contain;
      height: 150px;
    }

    textarea {
        padding: 10px;
        border: 1px solid black;
        border-radius: 5px;
        resize: none;
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
            height: 40px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            &:focus {
                outline: none;
            }
            &.scan-button {
                background-color: black;
                color: #fff;
                margin-right: 10px;

                &:hover {
                    background-color: #fff;
                    color: black;
                    border: 1px solid black;
                }
            }
            &.reset-button {
                background-color: #fff;
                color: black;
                border: 1px solid black;

                &:hover {
                    background-color: black;
                    color: #fff;
                }
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