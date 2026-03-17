import { useMutation, useQuery } from '@tanstack/react-query'
import api from '@/api/client'
import { useAuthStore } from '@/stores/authStore'
import type { LoginRequest, TokenResponse } from '@/types'

export function useLogin() {
  const login = useAuthStore((s) => s.login)
  return useMutation({
    mutationFn: async (data: LoginRequest) => {
      const res = await api.post<TokenResponse>('/api/auth/login', data)
      return res.data
    },
    onSuccess: (data) => {
      login(data.access_token, data.username, data.role)
    },
  })
}

export function useSetup() {
  const login = useAuthStore((s) => s.login)
  return useMutation({
    mutationFn: async (data: { username: string; password: string }) => {
      const res = await api.post<TokenResponse>('/api/auth/setup', { ...data, role: 'admin' })
      return res.data
    },
    onSuccess: (data) => {
      login(data.access_token, data.username, data.role)
    },
  })
}

export function useHasUsers() {
  return useQuery({
    queryKey: ['has-users'],
    queryFn: async () => {
      const res = await api.get<{ has_users: boolean }>('/api/auth/has-users')
      return res.data.has_users
    },
    staleTime: 30_000,
  })
}

export function useChangePassword() {
  return useMutation({
    mutationFn: async (data: { current_password: string; new_password: string }) => {
      const res = await api.post('/api/auth/change-password', data)
      return res.data
    },
  })
}
