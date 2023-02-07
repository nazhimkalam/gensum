import styled from "styled-components";

const About = () => {
  return (
    <StyledContainer id="about-section">
      <h2>About Gatot</h2>
      <p>
        Lorem, ipsum dolor sit amet consectetur adipisicing elit. Eius quo eaque, quam odit voluptatum omnis ex voluptate consectetur quisquam magnam porro dolor corporis id quod perspiciatis dolores animi aliquid quos fugit minima est eum iste? Quis at tenetur tempora debitis hic exercitationem, fugiat quas quos, repellat amet magnam, commodi minima.
      </p>

      <p>
        Lorem ipsum dolor sit amet consectetur adipisicing elit. Qui a architecto voluptates cupiditate. Culpa assumenda ipsum alias, repellat aperiam temporibus quos architecto error sint doloribus id enim necessitatibus tenetur ex explicabo cumque ducimus possimus labore sunt sed! Dignissimos, ullam illo?
      </p>

      <p>
        Lorem ipsum dolor sit amet consectetur adipisicing elit. Tenetur, ipsa recusandae. Atque unde facere omnis laudantium ut vel, enim, exercitationem fugiat maiores nesciunt blanditiis recusandae incidunt aut deserunt praesentium dolor expedita explicabo non aperiam magni quas placeat ipsa beatae! Laboriosam quis doloremque eveniet quisquam fugit?
      </p>
    </StyledContainer>
  );
};

export default About;

const StyledContainer = styled.div`
    margin: 0;
    padding: 2rem;

    h2 {
        font-size: 2rem;
        margin-bottom: 1rem;
    }

    p {
        margin: 0;
        padding: 0;
        text-align: justify;
        margin-bottom: 1rem;
    }
`;