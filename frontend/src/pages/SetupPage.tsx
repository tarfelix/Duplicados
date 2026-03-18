import { useState } from 'react'
import { useSetup } from '@/hooks/useAuth'
import { useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import { Loader2 } from 'lucide-react'

export default function SetupPage() {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [password2, setPassword2] = useState('')
  const setup = useSetup()
  const navigate = useNavigate()

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!username || !password) {
      toast.error('Preencha todos os campos.')
      return
    }
    if (password !== password2) {
      toast.error('As senhas não coincidem.')
      return
    }
    if (password.length < 8) {
      toast.error('A senha deve ter no mínimo 8 caracteres.')
      return
    }
    setup.mutate(
      { username, password },
      {
        onSuccess: () => {
          toast.success('Administrador criado!')
          navigate('/')
        },
        onError: (err: any) => {
          toast.error(err.response?.data?.detail || 'Erro ao criar administrador.')
        },
      }
    )
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-[hsl(214,60%,26%)] via-[hsl(214,55%,35%)] to-[hsl(207,60%,45%)] p-4">
      <div className="w-full max-w-sm">
        <div className="bg-card rounded-xl shadow-2xl p-8 space-y-6 border border-border/50">
          <div className="text-center space-y-4">
            <img src="/logo.png" alt="Soares Picon" className="h-16 mx-auto" />
            <div>
              <h1 className="text-lg font-semibold">Configuração Inicial</h1>
              <p className="text-xs text-muted-foreground mt-1">Crie o primeiro usuário administrador</p>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label htmlFor="setup-user" className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Usuário (administrador)
              </label>
              <input
                id="setup-user"
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="mt-1.5 w-full rounded-lg border bg-background px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary transition-colors"
                autoFocus
              />
            </div>
            <div>
              <label htmlFor="setup-pass" className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Senha
              </label>
              <input
                id="setup-pass"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="mt-1.5 w-full rounded-lg border bg-background px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary transition-colors"
              />
              {password.length > 0 && password.length < 8 && (
                <p className="text-[10px] text-destructive mt-1">Mínimo 8 caracteres</p>
              )}
            </div>
            <div>
              <label htmlFor="setup-pass2" className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Confirmar senha
              </label>
              <input
                id="setup-pass2"
                type="password"
                value={password2}
                onChange={(e) => setPassword2(e.target.value)}
                className="mt-1.5 w-full rounded-lg border bg-background px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary transition-colors"
              />
              {password2.length > 0 && password !== password2 && (
                <p className="text-[10px] text-destructive mt-1">Senhas não coincidem</p>
              )}
            </div>
            <button
              type="submit"
              disabled={setup.isPending}
              className="w-full rounded-lg bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50 transition-colors flex items-center justify-center gap-2"
            >
              {setup.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" /> Criando...
                </>
              ) : (
                'Criar Administrador'
              )}
            </button>
          </form>
        </div>

        <p className="text-center text-[10px] text-white/50 mt-4">
          Soares, Picon Sociedade de Advogados
        </p>
      </div>
    </div>
  )
}
