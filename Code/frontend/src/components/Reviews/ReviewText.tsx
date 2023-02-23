import styled from "styled-components";

const ReviewText = (props: any) => {
  const { details } = props;
  const { id, review, summary, counter, createdAt } = details;

  const onHandleDeleteReview = (reviewId: string) => {

  }

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
          <strong>Created At: </strong>
          {createdAt}
        </p>
      </section>
      <section>
        <button onClick={() => onHandleDeleteReview(id)}>DELETE</button>
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
      margin: 0.5pc 0;
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
