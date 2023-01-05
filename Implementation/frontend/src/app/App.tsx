import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import LoginPage from "../pages/login/LoginPage"
import LandingPage from "../pages/landing/LandingPage"
import RecordsPage from "../pages/records/RecordsPage"
import RegisterPage from "../pages/register/RegisterPage"

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage/>}/>
        <Route path="/login" element={<LoginPage/>}/>
        <Route path="/register" element={<RegisterPage/>}/>
        <Route path="/records" element={<RecordsPage/>}/>
      </Routes>
    </Router>
  )
}

export default App