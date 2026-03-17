import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useUsers, useCreateUser, useUpdateRole, useResetPassword, useDeleteUser } from '@/hooks/useUsers'
import { useChangePassword } from '@/hooks/useAuth'
import { useAuthStore } from '@/stores/authStore'
import { toast } from 'sonner'
import { ArrowLeft, Trash2, Key, Shield } from 'lucide-react'

export default function UsersPage() {
  const navigate = useNavigate()
  const { data: users, isLoading } = useUsers()
  const createUser = useCreateUser()
  const updateRole = useUpdateRole()
  const resetPassword = useResetPassword()
  const deleteUser = useDeleteUser()
  const changePassword = useChangePassword()
  const currentUsername = useAuthStore((s) => s.username)

  // New user form
  const [newUser, setNewUser] = useState('')
  const [newPass, setNewPass] = useState('')
  const [newRole, setNewRole] = useState('user')

  // Password reset
  const [editingPw, setEditingPw] = useState<string | null>(null)
  const [newPw, setNewPw] = useState('')

  // Own password
  const [ownPwOpen, setOwnPwOpen] = useState(false)
  const [curPw, setCurPw] = useState('')
  const [ownNewPw, setOwnNewPw] = useState('')

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
        <button onClick={() => navigate('/')} className="p-2 rounded hover:bg-accent">
          <ArrowLeft className="h-4 w-4" />
        </button>
        <h2 className="text-xl font-bold">Gerenciar Usuários</h2>
      </div>

      {/* Change own password */}
      <div>
        <button
          onClick={() => setOwnPwOpen(!ownPwOpen)}
          className="text-sm text-primary hover:underline flex items-center gap-1"
        >
          <Key className="h-3.5 w-3.5" /> Alterar minha senha
        </button>
        {ownPwOpen && (
          <div className="mt-2 p-4 border rounded-lg space-y-3">
            <input
              type="password"
              placeholder="Senha atual"
              value={curPw}
              onChange={(e) => setCurPw(e.target.value)}
              className="w-full rounded-md border px-3 py-1.5 text-sm"
            />
            <input
              type="password"
              placeholder="Nova senha (min 8 chars)"
              value={ownNewPw}
              onChange={(e) => setOwnNewPw(e.target.value)}
              className="w-full rounded-md border px-3 py-1.5 text-sm"
            />
            <button
              onClick={() => {
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
              className="rounded-md bg-primary px-3 py-1.5 text-sm text-primary-foreground hover:bg-primary/90"
            >
              Salvar
            </button>
          </div>
        )}
      </div>

      {/* User list */}
      <div className="border rounded-lg divide-y">
        {isLoading && <p className="p-4 text-muted-foreground">Carregando...</p>}
        {users?.map((u) => (
          <div key={u.username} className="flex items-center justify-between p-3 gap-3">
            <div className="flex-1">
              <span className="font-medium">{u.username}</span>
              <span className="ml-2 text-xs px-2 py-0.5 rounded bg-muted text-muted-foreground">{u.role}</span>
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
                className="rounded border px-2 py-1 text-xs"
              >
                <option value="user">user</option>
                <option value="admin">admin</option>
              </select>
              <button
                onClick={() => {
                  setEditingPw(editingPw === u.username ? null : u.username)
                  setNewPw('')
                }}
                className="p-1.5 rounded hover:bg-accent text-muted-foreground"
                title="Redefinir senha"
              >
                <Key className="h-3.5 w-3.5" />
              </button>
              {u.username !== currentUsername && u.role !== 'admin' && (
                <button
                  onClick={() => {
                    if (confirm(`Excluir ${u.username}?`))
                      deleteUser.mutate(u.username, {
                        onSuccess: () => toast.success('Excluído.'),
                        onError: (err: any) => toast.error(err.response?.data?.detail || 'Erro.'),
                      })
                  }}
                  className="p-1.5 rounded hover:bg-destructive/10 text-destructive"
                  title="Excluir"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              )}
            </div>
            {editingPw === u.username && (
              <div className="flex gap-2 items-center">
                <input
                  type="password"
                  placeholder="Nova senha"
                  value={newPw}
                  onChange={(e) => setNewPw(e.target.value)}
                  className="rounded border px-2 py-1 text-xs w-36"
                />
                <button
                  onClick={() =>
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
                  }
                  className="rounded bg-primary px-2 py-1 text-xs text-primary-foreground"
                >
                  Salvar
                </button>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* New user form */}
      <div className="border rounded-lg p-4">
        <h3 className="font-semibold mb-3">Novo Usuário</h3>
        <form onSubmit={handleCreate} className="flex flex-wrap gap-3 items-end">
          <div>
            <label className="text-xs text-muted-foreground">Usuário</label>
            <input
              type="text"
              value={newUser}
              onChange={(e) => setNewUser(e.target.value)}
              className="mt-1 block rounded-md border px-3 py-1.5 text-sm"
            />
          </div>
          <div>
            <label className="text-xs text-muted-foreground">Senha</label>
            <input
              type="password"
              value={newPass}
              onChange={(e) => setNewPass(e.target.value)}
              className="mt-1 block rounded-md border px-3 py-1.5 text-sm"
            />
          </div>
          <div>
            <label className="text-xs text-muted-foreground">Perfil</label>
            <select
              value={newRole}
              onChange={(e) => setNewRole(e.target.value)}
              className="mt-1 block rounded-md border px-3 py-1.5 text-sm"
            >
              <option value="user">user</option>
              <option value="admin">admin</option>
            </select>
          </div>
          <button
            type="submit"
            disabled={createUser.isPending}
            className="rounded-md bg-primary px-4 py-1.5 text-sm text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            Criar
          </button>
        </form>
      </div>
    </div>
  )
}
