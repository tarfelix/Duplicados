import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '@/api/client'
import type { User } from '@/types'

export function useUsers() {
  return useQuery({
    queryKey: ['users'],
    queryFn: async () => {
      const res = await api.get<User[]>('/api/users')
      return res.data
    },
  })
}

export function useCreateUser() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (data: { username: string; password: string; role: string }) => {
      const res = await api.post('/api/users', data)
      return res.data
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ['users'] }),
  })
}

export function useUpdateRole() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async ({ username, role }: { username: string; role: string }) => {
      const res = await api.patch(`/api/users/${username}/role`, { role })
      return res.data
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ['users'] }),
  })
}

export function useResetPassword() {
  return useMutation({
    mutationFn: async ({ username, new_password }: { username: string; new_password: string }) => {
      const res = await api.patch(`/api/users/${username}/password`, { new_password })
      return res.data
    },
  })
}

export function useDeleteUser() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (username: string) => {
      const res = await api.delete(`/api/users/${username}`)
      return res.data
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ['users'] }),
  })
}
