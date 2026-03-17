import { create } from 'zustand'

interface AuthState {
  token: string | null
  username: string | null
  role: 'admin' | 'user' | null
  login: (token: string, username: string, role: 'admin' | 'user') => void
  logout: () => void
  isAuthenticated: () => boolean
  isAdmin: () => boolean
}

export const useAuthStore = create<AuthState>((set, get) => ({
  token: localStorage.getItem('access_token'),
  username: localStorage.getItem('username'),
  role: localStorage.getItem('role') as 'admin' | 'user' | null,

  login: (token, username, role) => {
    localStorage.setItem('access_token', token)
    localStorage.setItem('username', username)
    localStorage.setItem('role', role)
    set({ token, username, role })
  },

  logout: () => {
    localStorage.removeItem('access_token')
    localStorage.removeItem('username')
    localStorage.removeItem('role')
    set({ token: null, username: null, role: null })
  },

  isAuthenticated: () => !!get().token,
  isAdmin: () => get().role === 'admin',
}))
