import { useNavigate } from 'react-router-dom'
import { useAuthStore } from '@/stores/authStore'
import { LogOut, Menu, Users, Moon, Sun } from 'lucide-react'
import { useState } from 'react'

export default function Header({ onMenuClick }: { onMenuClick: () => void }) {
  const { username, role, logout } = useAuthStore()
  const navigate = useNavigate()
  const [dark, setDark] = useState(document.documentElement.classList.contains('dark'))

  const toggleDark = () => {
    document.documentElement.classList.toggle('dark')
    setDark(!dark)
  }

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  return (
    <header className="flex items-center justify-between border-b bg-card px-4 py-3 shadow-sm">
      <div className="flex items-center gap-3">
        <button onClick={onMenuClick} className="md:hidden p-1 rounded hover:bg-accent">
          <Menu className="h-5 w-5" />
        </button>
        <h1 className="text-lg font-semibold">Verificador de Duplicidade</h1>
      </div>
      <div className="flex items-center gap-2">
        <span className="text-sm text-muted-foreground hidden sm:inline">
          {username} {role === 'admin' && '(Admin)'}
        </span>
        {role === 'admin' && (
          <button
            onClick={() => navigate('/users')}
            className="p-2 rounded-md hover:bg-accent text-muted-foreground"
            title="Gerenciar usuários"
          >
            <Users className="h-4 w-4" />
          </button>
        )}
        <button onClick={toggleDark} className="p-2 rounded-md hover:bg-accent text-muted-foreground" title="Alternar tema">
          {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </button>
        <button onClick={handleLogout} className="p-2 rounded-md hover:bg-accent text-muted-foreground" title="Sair">
          <LogOut className="h-4 w-4" />
        </button>
      </div>
    </header>
  )
}
