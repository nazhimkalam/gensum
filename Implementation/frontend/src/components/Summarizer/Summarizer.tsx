import axios from "axios";
import { useState } from "react";
import { useSelector } from "react-redux";
import styled from "styled-components";
import { gatotApi } from "../../apis/gatotApi";
import { selectUser } from "../../redux/reducers/userReducer";

const Summarizer = () => {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [textareaContent, setTextareaContent] = useState<string>("");
  const [result, setResult] = useState<string | undefined>();
  const user = useSelector(selectUser);

  const handleScan = () => {
    if (!textareaContent) {
      alert("Please enter text to scan");
      return;
    }
    setIsLoading(true);
    // let summarizeRequestBody = { detectionText: textareaContent };
    // const categories = ["Not hate", "hate"]
    // axios
    //   .post(gatotApi.summarize, summarizeRequestBody)
    //   .then((response) => {
    //     let detectionResult = response.data["Prediction"];
    //     let userRequestBody = { fullName: user.displayName ?? "", email: user.email ?? "" };
    //     let detectionResultIndex = categories.indexOf(detectionResult);
        
    //     setResult(detectionResult);

    //     axios.post(userApi.create, userRequestBody).then((response) => { 
    //       let userId = parseInt(response.data["id"]);
    //       let SummarizerRequestBody = { userId: userId, result: detectionResultIndex.toString(), description: textareaContent };

    //       axios.post(SummarizerApi.create, SummarizerRequestBody).then(() => { 
    //         alert("Successfully scanned and saved!");
        
    //       }).catch(() => alert("Something weht wrong when creating detection result into the database"));
    //     }).catch(() => alert("Something went wrong while creating the user document"));
    //   })
    //   .catch(() => alert("Something went wrong when scanning the text"))
    //   .finally(() => setIsLoading(false));
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
      />
      <section className="buttons">
        <button className="scan-button" onClick={handleScan} disabled={isLoading}> Summarize </button>
        <button className="reset-button" onClick={handleReset} disabled={isLoading}> Reset </button>
      </section>

      {result && (
        <div className="detection-result">
          <h3>Detection result:</h3> Text is of type: {result}
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
            padding: 10px 40px;
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