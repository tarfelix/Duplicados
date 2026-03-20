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
  const { data: hasUsers, isLoading, isError, error, refetch } = useHasUsers()

  if (isAuth) return <Navigate to="/" replace />
  if (isLoading) return <div className="flex items-center justify-center h-screen text-muted-foreground">Carregando...</div>
  if (isError) return (
    <div className="flex flex-col items-center justify-center h-screen gap-4 p-4">
      <p className="text-destructive font-medium">Erro ao conectar com o servidor</p>
      <p className="text-sm text-muted-foreground text-center max-w-md">
        {(error as any)?.message === 'Network Error'
          ? 'O backend não está acessível. Verifique se os serviços estão rodando.'
          : (error as any)?.response?.data?.detail || (error as any)?.message || 'Erro desconhecido'}
      </p>
      <button onClick={() => refetch()} className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors">
        Tentar novamente
      </button>
    </div>
  )
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
