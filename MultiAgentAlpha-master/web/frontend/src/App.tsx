import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { ConfigProvider, App as AntApp } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import Layout from './components/Layout'
import NewRun from './pages/NewRun'
import ActiveRun from './pages/ActiveRun'
import HistoryRuns from './pages/HistoryRuns'
import RunMonitor from './pages/RunMonitor'
import FactorDashboard from './pages/FactorDashboard'
import LineageTree from './pages/LineageTree'
import FactorLibrary from './pages/FactorLibrary'
import RunReport from './pages/RunReport'

export default function App() {
  return (
    <ConfigProvider locale={zhCN} theme={{ token: { colorPrimary: '#1677ff' } }}>
      <AntApp>
        <BrowserRouter>
          <Layout>
            <Routes>
              <Route path="/" element={<Navigate to="/runs/new" replace />} />
              <Route path="/runs/new" element={<NewRun />} />
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
          </Layout>
        </BrowserRouter>
      </AntApp>
    </ConfigProvider>
  )
}
