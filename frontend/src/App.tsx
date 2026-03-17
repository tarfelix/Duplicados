import { Routes, Route, Navigate } from 'react-router-dom'
import { useAuthStore } from '@/stores/authStore'
import { useHasUsers } from '@/hooks/useAuth'
import LoginPage from '@/pages/LoginPage'
import SetupPage from '@/pages/SetupPage'
import DashboardPage from '@/pages/DashboardPage'
import UsersPage from '@/pages/UsersPage'
import Layout from '@/components/layout/Layout'

function PrivateRoute({ children }: { children: React.ReactNode }) {
  const isAuth = useAuthStore((s) => s.isAuthenticated())
  if (!isAuth) return <Navigate to="/login" replace />
  return <>{children}</>
}

function AdminRoute({ children }: { children: React.ReactNode }) {
  const isAuth = useAuthStore((s) => s.isAuthenticated())
  const isAdmin = useAuthStore((s) => s.isAdmin())
  if (!isAuth) return <Navigate to="/login" replace />
  if (!isAdmin) return <Navigate to="/" replace />
  return <>{children}</>
}

function AuthRedirect() {
  const isAuth = useAuthStore((s) => s.isAuthenticated())
  const { data: hasUsers, isLoading } = useHasUsers()

  if (isLoading) return <div className="flex items-center justify-center h-screen">Carregando...</div>
  if (isAuth) return <Navigate to="/" replace />
  if (hasUsers === false) return <Navigate to="/setup" replace />
  return <LoginPage />
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<AuthRedirect />} />
      <Route path="/setup" element={<SetupPage />} />
      <Route
        path="/"
        element={
          <PrivateRoute>
            <Layout>
              <DashboardPage />
            </Layout>
          </PrivateRoute>
        }
      />
      <Route
        path="/users"
        element={
          <AdminRoute>
            <Layout>
              <UsersPage />
            </Layout>
          </AdminRoute>
        }
      />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
