import { Button } from "antd";
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

  const handleScan = async () => {
    if (!textareaContent) {
      alert("Please enter text to scan");
      return;
    }
    setIsLoading(true);
    
    const apiEndpoint = user.id === undefined ? gensumApi.generalSummarization : gensumApi.domainSpecificSummarization;
    const summarizeRequestBody = user.id === undefined ? { review: textareaContent } : { review: textareaContent, userId: user.id };
      
    await postRequest(apiEndpoint, summarizeRequestBody).then((response) => {
      setResult({
        review: textareaContent,
        summary: response.summary,
        sentiement: response.sentiment.sentiment,
        score: response.sentiment.score
      })
    }).catch(() => alert("Something went wrong when scanning the text")).finally(() => setIsLoading(false));
  };

  const handleReset = () => {
    setTextareaContent("");
    setResult(undefined);
  };

  return (
    <StyledContainer>
      <textarea
        rows={10}
        disabled={isLoading}
        value={textareaContent}
        onChange={(e) => setTextareaContent(e.target.value)}
        placeholder="Enter review here..."
        style={{ cursor: isLoading ? "not-allowed" : "pointer" }}
      />
      <section className="buttons">
        <Button className="scan-button" onClick={handleScan} disabled={isLoading}
          style={{ cursor: isLoading ? "not-allowed" : "pointer" }}
        > {isLoading ? 'Summarizing...' : 'Summarize'} </Button>
        <Button className="reset-button" onClick={handleReset} disabled={isLoading}
          style={{ cursor: isLoading ? "not-allowed" : "pointer" }}
        > Reset </Button>
      </section>

      {result && (
        <div className="detection-result">
          <h3>Summarization result:</h3> {result.summary}
        </div>
      )}
    </StyledContainer>
  );
};

export default Summarizer;

const StyledContainer = styled.div`
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    margin: 0 5vw;
    padding: 20px;
    textarea {
        width:95%;
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
    .detection-result {
        margin-top: 20px;
        h3 {
            margin-bottom: 10px;
        }
    }


`;