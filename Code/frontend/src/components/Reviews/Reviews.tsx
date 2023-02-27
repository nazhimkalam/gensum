import styled from "styled-components";
import ReviewText from "./ReviewText";
import { useEffect, useState } from "react";
import { getReviewsByUserId } from "../../services/gensum.service";
import { useSelector } from "react-redux";
import { selectUser } from "../../redux/reducers/userReducer";
import { Button } from "antd";
import { getRequest } from "../../utils/requests";
import { gensumApi } from "../../apis/gensumApi";

const Reviews = () => {
  const [reviews, setReviews] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const user = useSelector(selectUser);
  const [deletedReviewId, setDeletedReviewId] = useState("");
  const [generatingCsvLoader, setGeneratingCsvLoader] = useState(false);

  useEffect(() => {
    const fetchReviews = async () => {
      setIsLoading(true);
      const response = await getReviewsByUserId(user.id);
      const convertTimestamp = response.reviews.map((review: any) => {
        return { ...review, createdAt: review.createdAt };
      });
      const sortedReviews = convertTimestamp.sort((a: any, b: any) => {
        const dateA = new Date(a.createdAt);
        const dateB = new Date(b.createdAt);
        return dateB.getTime() - dateA.getTime();
      });

      const reviewsWithCounter = sortedReviews.map(
        (review: any, index: any) => {
          return { ...review, counter: index + 1 };
        }
      );

      setReviews(reviewsWithCounter);
      setIsLoading(false);
    };
    if (user.id) fetchReviews();
  }, [user]);

  useEffect(() => {
    if (deletedReviewId) {
      console.log("review deletion triggered...");
      const filteredReviews = reviews.filter((review: any) => review.id !== deletedReviewId);
      setReviews(filteredReviews);
      setDeletedReviewId("");
    }
  }, [deletedReviewId, reviews]);

  const handleDownloadRecords = async () => {
    setGeneratingCsvLoader(true)
    await getRequest(`${gensumApi.generateReviewCsv}/${user.id}/csv`).then(
      (res) => {
        const blob = new Blob([res.data], { type: "text/csv" });
        const link = document.createElement("a");
        link.href = window.URL.createObjectURL(blob);
        link.download = "reviews.csv";
        link.click();
      }
    ).catch(
      (err) => { 
        console.log(err);
      }).finally(() => { setGeneratingCsvLoader(false) });
  };

  return (
    <StyledContainer>
      <section>
        {!isLoading && <h1>{reviews.length < 1 ? 'No records available' : 'Reviews'}</h1>}
        {!isLoading && reviews.length > 0 && <Button onClick={handleDownloadRecords} disabled={generatingCsvLoader}>{generatingCsvLoader ? 'Generating...' : 'Download Records'}</Button>}
      </section>

      <section>
        {isLoading ? (
          <p>Loading...</p>
        ) : (
          reviews.map((review, index) => (
            <ReviewText key={index} details={review} setDeletedReviewId={setDeletedReviewId}/>
          ))
        )}
      </section>
    </StyledContainer>
  );
};

export default Reviews;

const StyledContainer = styled.div`
  margin: 1pc 5vw;
  padding: 1pc;

  > h1 {
    margin: 1pc 0;
  }

  > section {
    margin: 1pc 0;

    &:first-child {
      display: flex;
      justify-content: space-between;

      > button {
        margin: 0;
      }
    }
  }
`;
