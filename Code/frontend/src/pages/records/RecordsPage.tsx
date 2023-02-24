import { useEffect } from "react";
import { useSelector } from "react-redux";
import { useNavigate } from "react-router-dom";
import Footer from "../../components/Footer/Footer";
import Header from "../../components/Header/Header";
import Reviews from "../../components/Reviews/Reviews";
import { selectUser } from "../../redux/reducers/userReducer";
import { routePaths } from "../../app/routes";

const RecordsPage = () => {
  const user = useSelector(selectUser);
  const naviagte = useNavigate();

  useEffect(() => {
    if (!user?.id) {
      naviagte(routePaths.home);
    }
  }, [user, naviagte])
  
  return (
    <div>
      <Header />
      <Reviews />
      <Footer />
    </div>
  );
};

export default RecordsPage;
