import styled from "styled-components";
import ReviewText from "./ReviewText";

const Reviews = () => {
    const reviews = [
        {
            id: "546asd5f4asdf6521dsf",
            review: "This is a review",
            summary: "This is a summary",
            counter: 1,
            createdAt: "2021-01-01",
        },
        {
            id: '3845jksdhfgjsdfkghjksdh',
            review: "This is a review",
            summary: "This is a summary",
            counter: 2,
            createdAt: "2021-01-01",
        },
    ];
  return (
    <StyledContainer>
        <h1>Reviews</h1>

        <section>
            {reviews.map((review, index) => (
                <ReviewText key={index} details={review} />
            ))}
        </section>

    </StyledContainer>
  )
}

export default Reviews

const StyledContainer = styled.div`
    margin: 1pc;
    padding: 1pc;

    > h1 {
        margin: 1pc 0;
    }

    > section {
        margin: 1pc 0;
    }
`;