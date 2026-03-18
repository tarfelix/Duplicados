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
    <header className="flex items-center justify-between border-b bg-card px-4 py-2.5 shadow-sm">
      <div className="flex items-center gap-3">
        <button onClick={onMenuClick} className="md:hidden p-1 rounded hover:bg-accent" aria-label="Menu">
          <Menu className="h-5 w-5" />
        </button>
        <img
          src={dark ? '/logo-dark.png' : '/logo.png'}
          alt="Soares Picon Advogados"
          className="h-9 object-contain"
        />
        <div className="hidden sm:block border-l pl-3 ml-1">
          <h1 className="text-sm font-semibold leading-tight text-foreground">Verificador de Duplicidade</h1>
          <p className="text-[10px] text-muted-foreground leading-tight">Soares, Picon Sociedade de Advogados</p>
        </div>
      </div>
      <div className="flex items-center gap-1.5">
        <span className="text-xs text-muted-foreground hidden sm:inline mr-1">
          {username} {role === 'admin' && <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary font-medium ml-1">Admin</span>}
        </span>
        {role === 'admin' && (
          <button
            onClick={() => navigate('/users')}
            className="p-2 rounded-md hover:bg-accent text-muted-foreground transition-colors"
            title="Gerenciar usuários"
          >
            <Users className="h-4 w-4" />
          </button>
        )}
        <button onClick={toggleDark} className="p-2 rounded-md hover:bg-accent text-muted-foreground transition-colors" title="Alternar tema">
          {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </button>
        <button onClick={handleLogout} className="p-2 rounded-md hover:bg-accent text-muted-foreground transition-colors" title="Sair">
          <LogOut className="h-4 w-4" />
        </button>
      </div>
    </header>
  )
}
