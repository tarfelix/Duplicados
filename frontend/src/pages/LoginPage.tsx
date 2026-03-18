import { useState } from 'react'
import { useLogin } from '@/hooks/useAuth'
import { useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import { Loader2 } from 'lucide-react'

export default function LoginPage() {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const login = useLogin()
  const navigate = useNavigate()

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!username || !password) {
      toast.error('Preencha todos os campos.')
      return
    }
    login.mutate(
      { username, password },
      {
        onSuccess: () => navigate('/'),
        onError: (err: any) => {
          toast.error(err.response?.data?.detail || 'Usuário ou senha inválidos.')
        },
      }
    )
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-[hsl(214,60%,26%)] via-[hsl(214,55%,35%)] to-[hsl(207,60%,45%)] p-4">
      <div className="w-full max-w-sm">
        <div className="bg-card rounded-xl shadow-2xl p-8 space-y-6 border border-border/50">
          {/* Logo */}
          <div className="text-center space-y-4">
            <img src="/logo.png" alt="Soares Picon" className="h-16 mx-auto" />
            <div>
              <h1 className="text-lg font-semibold text-foreground">Verificador de Duplicidade</h1>
              <p className="text-xs text-muted-foreground mt-1">Faça login para continuar</p>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label htmlFor="login-user" className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Usuário
              </label>
              <input
                id="login-user"
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="mt-1.5 w-full rounded-lg border bg-background px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary transition-colors"
                autoFocus
                autoComplete="username"
              />
            </div>
            <div>
              <label htmlFor="login-pass" className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Senha
              </label>
              <input
                id="login-pass"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="mt-1.5 w-full rounded-lg border bg-background px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary transition-colors"
                autoComplete="current-password"
              />
            </div>
            <button
              type="submit"
              disabled={login.isPending}
              className="w-full rounded-lg bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50 transition-colors flex items-center justify-center gap-2"
            >
              {login.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" /> Entrando...
                </>
              ) : (
                'Entrar'
              )}
            </button>
          </form>

          <p className="text-[10px] text-center text-muted-foreground">
            Esqueceu a senha? Contate o administrador.
          </p>
        </div>

        <p className="text-center text-[10px] text-white/50 mt-4">
          Soares, Picon Sociedade de Advogados
        </p>
      </div>
    </div>
  )
}
