import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import LoginPage from "../pages/login/LoginPage";
import LandingPage from "../pages/landing/LandingPage";
import RecordsPage from "../pages/records/RecordsPage";
import { routePaths } from "./routes";
import UserProfilePage from "../pages/profile/UserProfilePage";

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path={routePaths.home} element={<LandingPage />} />
        <Route path={routePaths.login} element={<LoginPage />} />
        <Route path={routePaths.records} element={<RecordsPage />} />
        <Route path={routePaths.profile} element={<UserProfilePage />} />
      </Routes>
    </Router>
  );
};

export default App;
