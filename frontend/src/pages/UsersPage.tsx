import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useUsers, useCreateUser, useUpdateRole, useResetPassword, useDeleteUser } from '@/hooks/useUsers'
import { useChangePassword } from '@/hooks/useAuth'
import { useAuthStore } from '@/stores/authStore'
import { toast } from 'sonner'
import { ArrowLeft, Trash2, Key, UserPlus, Loader2 } from 'lucide-react'

export default function UsersPage() {
  const navigate = useNavigate()
  const { data: users, isLoading } = useUsers()
  const createUser = useCreateUser()
  const updateRole = useUpdateRole()
  const resetPassword = useResetPassword()
  const deleteUser = useDeleteUser()
  const changePassword = useChangePassword()
  const currentUsername = useAuthStore((s) => s.username)

  const [newUser, setNewUser] = useState('')
  const [newPass, setNewPass] = useState('')
  const [newRole, setNewRole] = useState('user')
  const [editingPw, setEditingPw] = useState<string | null>(null)
  const [newPw, setNewPw] = useState('')
  const [ownPwOpen, setOwnPwOpen] = useState(false)
  const [curPw, setCurPw] = useState('')
  const [ownNewPw, setOwnNewPw] = useState('')
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)

  const handleCreate = (e: React.FormEvent) => {
    e.preventDefault()
    if (!newUser || !newPass) return toast.error('Preencha todos os campos.')
    if (newPass.length < 8) return toast.error('Senha mínima de 8 caracteres.')
    createUser.mutate(
      { username: newUser, password: newPass, role: newRole },
      {
        onSuccess: () => {
          toast.success('Usuário criado!')
          setNewUser('')
          setNewPass('')
        },
        onError: (err: any) => toast.error(err.response?.data?.detail || 'Erro ao criar.'),
      }
    )
  }

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div className="flex items-center gap-3">
        <button onClick={() => navigate('/')} className="p-2 rounded-lg hover:bg-accent transition-colors">
          <ArrowLeft className="h-4 w-4" />
        </button>
        <h2 className="text-xl font-bold">Gerenciar Usuários</h2>
      </div>

      {/* Change own password */}
      <div>
        <button
          onClick={() => setOwnPwOpen(!ownPwOpen)}
          className="text-sm text-primary hover:underline flex items-center gap-1.5"
        >
          <Key className="h-3.5 w-3.5" /> Alterar minha senha
        </button>
        {ownPwOpen && (
          <div className="mt-2 p-4 border rounded-xl space-y-3 bg-card">
            <input
              type="password"
              placeholder="Senha atual"
              value={curPw}
              onChange={(e) => setCurPw(e.target.value)}
              className="w-full rounded-lg border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
            <input
              type="password"
              placeholder="Nova senha (min 8 chars)"
              value={ownNewPw}
              onChange={(e) => setOwnNewPw(e.target.value)}
              className="w-full rounded-lg border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
            <div className="flex gap-2">
              <button
                onClick={() => {
                  if (ownNewPw.length < 8) return toast.error('Mínimo 8 caracteres.')
                  changePassword.mutate(
                    { current_password: curPw, new_password: ownNewPw },
                    {
                      onSuccess: () => {
                        toast.success('Senha alterada!')
                        setOwnPwOpen(false)
                        setCurPw('')
                        setOwnNewPw('')
                      },
                      onError: (err: any) => toast.error(err.response?.data?.detail || 'Erro.'),
                    }
                  )
                }}
                disabled={changePassword.isPending}
                className="rounded-lg bg-primary px-4 py-2 text-sm text-primary-foreground hover:bg-primary/90 disabled:opacity-50 transition-colors"
              >
                {changePassword.isPending ? 'Salvando...' : 'Salvar'}
              </button>
              <button
                onClick={() => { setOwnPwOpen(false); setCurPw(''); setOwnNewPw('') }}
                className="rounded-lg border px-4 py-2 text-sm hover:bg-accent transition-colors"
              >
                Cancelar
              </button>
            </div>
          </div>
        )}
      </div>

      {/* User list */}
      <div className="border rounded-xl divide-y bg-card overflow-hidden">
        {isLoading && (
          <div className="flex items-center justify-center p-6 gap-2">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span className="text-sm text-muted-foreground">Carregando...</span>
          </div>
        )}
        {users?.map((u) => (
          <div key={u.username} className="p-3 space-y-2">
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <span className="font-medium text-sm">{u.username}</span>
                <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${
                  u.role === 'admin' ? 'bg-primary/10 text-primary' : 'bg-muted text-muted-foreground'
                }`}>
                  {u.role}
                </span>
                {u.username === currentUsername && (
                  <span className="text-[10px] text-muted-foreground">(você)</span>
                )}
              </div>
              <div className="flex items-center gap-1">
                <select
                  value={u.role}
                  onChange={(e) =>
                    updateRole.mutate(
                      { username: u.username, role: e.target.value },
                      {
                        onSuccess: () => toast.success('Perfil atualizado.'),
                        onError: (err: any) => toast.error(err.response?.data?.detail || 'Erro.'),
                      }
                    )
                  }
                  className="rounded-lg border px-2 py-1 text-xs focus:outline-none focus:ring-2 focus:ring-primary/50"
                >
                  <option value="user">user</option>
                  <option value="admin">admin</option>
                </select>
                <button
                  onClick={() => {
                    setEditingPw(editingPw === u.username ? null : u.username)
                    setNewPw('')
                  }}
                  className="p-1.5 rounded-lg hover:bg-accent text-muted-foreground transition-colors"
                  title="Redefinir senha"
                >
                  <Key className="h-3.5 w-3.5" />
                </button>
                {u.username !== currentUsername && (
                  <button
                    onClick={() => setConfirmDelete(u.username)}
                    className="p-1.5 rounded-lg hover:bg-destructive/10 text-destructive transition-colors"
                    title="Excluir"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                )}
              </div>
            </div>
            {editingPw === u.username && (
              <div className="flex gap-2 items-center pl-2">
                <input
                  type="password"
                  placeholder="Nova senha (min 8)"
                  value={newPw}
                  onChange={(e) => setNewPw(e.target.value)}
                  className="rounded-lg border px-2 py-1.5 text-xs w-40 focus:outline-none focus:ring-2 focus:ring-primary/50"
                />
                <button
                  onClick={() => {
                    if (newPw.length < 8) return toast.error('Mínimo 8 caracteres.')
                    resetPassword.mutate(
                      { username: u.username, new_password: newPw },
                      {
                        onSuccess: () => {
                          toast.success('Senha redefinida.')
                          setEditingPw(null)
                        },
                        onError: (err: any) => toast.error(err.response?.data?.detail || 'Erro.'),
                      }
                    )
                  }}
                  className="rounded-lg bg-primary px-3 py-1.5 text-xs text-primary-foreground hover:bg-primary/90 transition-colors"
                >
                  Salvar
                </button>
                <button
                  onClick={() => setEditingPw(null)}
                  className="rounded-lg border px-3 py-1.5 text-xs hover:bg-accent transition-colors"
                >
                  Cancelar
                </button>
              </div>
            )}
            {confirmDelete === u.username && (
              <div className="flex gap-2 items-center pl-2 bg-destructive/5 rounded-lg p-2">
                <span className="text-xs text-destructive">Confirmar exclusão de <strong>{u.username}</strong>?</span>
                <button
                  onClick={() => {
                    deleteUser.mutate(u.username, {
                      onSuccess: () => { toast.success('Excluído.'); setConfirmDelete(null) },
                      onError: (err: any) => toast.error(err.response?.data?.detail || 'Erro.'),
                    })
                  }}
                  className="rounded-lg bg-destructive px-3 py-1 text-xs text-destructive-foreground hover:bg-destructive/90 transition-colors"
                >
                  Sim, excluir
                </button>
                <button
                  onClick={() => setConfirmDelete(null)}
                  className="rounded-lg border px-3 py-1 text-xs hover:bg-accent transition-colors"
                >
                  Cancelar
                </button>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* New user form */}
      <div className="border rounded-xl p-5 bg-card">
        <h3 className="font-semibold mb-4 flex items-center gap-2">
          <UserPlus className="h-4 w-4 text-primary" />
          Novo Usuário
        </h3>
        <form onSubmit={handleCreate} className="flex flex-wrap gap-3 items-end">
          <div>
            <label htmlFor="new-user" className="text-[10px] text-muted-foreground uppercase tracking-wider font-semibold">Usuário</label>
            <input
              id="new-user"
              type="text"
              value={newUser}
              onChange={(e) => setNewUser(e.target.value)}
              className="mt-1 block rounded-lg border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>
          <div>
            <label htmlFor="new-pass" className="text-[10px] text-muted-foreground uppercase tracking-wider font-semibold">Senha</label>
            <input
              id="new-pass"
              type="password"
              value={newPass}
              onChange={(e) => setNewPass(e.target.value)}
              className="mt-1 block rounded-lg border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>
          <div>
            <label htmlFor="new-role" className="text-[10px] text-muted-foreground uppercase tracking-wider font-semibold">Perfil</label>
            <select
              id="new-role"
              value={newRole}
              onChange={(e) => setNewRole(e.target.value)}
              className="mt-1 block rounded-lg border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
            >
              <option value="user">user</option>
              <option value="admin">admin</option>
            </select>
          </div>
          <button
            type="submit"
            disabled={createUser.isPending}
            className="rounded-lg bg-primary px-5 py-2 text-sm text-primary-foreground hover:bg-primary/90 disabled:opacity-50 transition-colors flex items-center gap-2"
          >
            {createUser.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
            Criar
          </button>
        </form>
      </div>
    </div>
  )
}
