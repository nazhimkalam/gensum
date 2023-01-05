import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import LoginPage from "../pages/login/LoginPage"
import LandingPage from "../pages/landing/LandingPage"
import RecordsPage from "../pages/records/RecordsPage"
import RegisterPage from "../pages/register/RegisterPage"
import { routePaths } from './routes'

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path={routePaths.home} element={<LandingPage/>}/>
        <Route path={routePaths.login} element={<LoginPage/>}/>
        <Route path={routePaths.register} element={<RegisterPage/>}/>
        <Route path={routePaths.records} element={<RecordsPage/>}/>
      </Routes>
    </Router>
  )
}

export default App