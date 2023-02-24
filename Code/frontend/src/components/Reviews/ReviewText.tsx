import { useSelector } from "react-redux";
import styled from "styled-components";
import { db } from "../../firebase/firebase";
import { selectUser } from "../../redux/reducers/userReducer";
import { useState } from "react";

const ReviewText = (props: any) => {
  const { details, setDeletedReviewId } = props;
  const { id, review, summary, counter, createdAt, sentiment, score } = details;
  const user = useSelector(selectUser);
  const [isLoading, setIsLoading] = useState(false)

  const onHandleDeleteReview = () => {
    setIsLoading(true)
    db.collection("users").doc(user.id).collection("reviews").doc(id).delete()
      .then(() => {
        console.log("Document successfully deleted!");
        setDeletedReviewId(id)
      })
      .catch((error) => {
        console.error("Error removing document: ", error);
      })
      .finally(() => {
        setIsLoading(false)
      });
  };

  return (
    <StyledContainer>
      <section>
        <h1>Review {counter}</h1>
        <p>
          <strong>Review: </strong>
          {review}
        </p>

        <p>
          <strong>Summary: </strong>
          {summary}
        </p>

        <p>
          <strong>Sentiment: </strong>
          {sentiment}
        </p>

        <p>
          <strong>Sentiment score: </strong>
          {score}
        </p>

        <p>
          <strong>Created At: </strong>
          {createdAt}
        </p>
      </section>
      <section>
        <button onClick={onHandleDeleteReview} disabled={isLoading}>
          {isLoading ? "Deleting..." : "Delete"}
        </button>
      </section>
    </StyledContainer>
  );
};

export default ReviewText;

const StyledContainer = styled.div`
  margin: 2pc 0;
  padding: 1pc 0;

  display: flex;
  justify-content: space-between;
  align-items: end;
  box-shadow: 0 1px 3px 0 rgba(60, 64, 67, 0.3),
    0 4px 8px 3px rgba(60, 64, 67, 0.15);

  > section {
    margin: 0 1pc;

    > h1 {
      margin: 1pc 0;
    }

    > p {
      margin: 1pc 0;
      text-align: justify;
    }

    > button {
      margin: 1pc 0;
      padding: 0.5pc 1pc;
      border: 1px solid red;
      background-color: red;
      cursor: pointer;
      color: white;

      &:hover {
        background-color: white;
        color: red;
      }
    }
  }
`;
