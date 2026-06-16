import { BrowserRouter, Routes, Route, Navigate, useLocation } from 'react-router-dom'
import { ConfigProvider, App as AntApp } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import Layout from './components/Layout'
import ErrorBoundary from './components/ErrorBoundary'
import NewRun from './pages/NewRun'
import Preprocess from './pages/Preprocess'
import FactorValidation from './pages/FactorValidation'
import ActiveRun from './pages/ActiveRun'
import HistoryRuns from './pages/HistoryRuns'
import RunMonitor from './pages/RunMonitor'
import FactorDashboard from './pages/FactorDashboard'
import LineageTree from './pages/LineageTree'
import FactorLibrary from './pages/FactorLibrary'
import RunReport from './pages/RunReport'

function RoutedContent() {
  const location = useLocation()
  return (
    <Layout>
      {/* resetKey on the path so navigating away clears a caught error */}
      <ErrorBoundary resetKey={location.pathname}>
        <Routes>
          <Route path="/" element={<Navigate to="/runs/new" replace />} />
          <Route path="/runs/new" element={<NewRun />} />
          <Route path="/preprocess" element={<Preprocess />} />
          <Route path="/factor-validation" element={<FactorValidation />} />
          <Route path="/running" element={<ActiveRun />} />
          <Route path="/history" element={<HistoryRuns />} />
          <Route path="/runs/:runId" element={<RunMonitor />} />
          <Route path="/runs/:runId/factors" element={<FactorDashboard />} />
          <Route path="/runs/:runId/report" element={<RunReport />} />
          <Route path="/runs/:runId/lineage" element={<LineageTree />} />
          <Route path="/factor-library" element={<FactorLibrary />} />
          {/* legacy redirect */}
          <Route path="/runs" element={<Navigate to="/history" replace />} />
        </Routes>
      </ErrorBoundary>
    </Layout>
  )
}

export default function App() {
  return (
    <ConfigProvider locale={zhCN} theme={{ token: { colorPrimary: '#1677ff' } }}>
      <AntApp>
        <BrowserRouter>
          <RoutedContent />
        </BrowserRouter>
      </AntApp>
    </ConfigProvider>
  )
}
