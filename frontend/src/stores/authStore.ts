import { create } from 'zustand'

function isTokenExpired(token: string): boolean {
  try {
    const payload = JSON.parse(atob(token.split('.')[1]))
    if (!payload.exp) return false
    // Add 30s buffer to avoid edge-case failures
    return Date.now() >= (payload.exp * 1000) - 30_000
  } catch {
    return true
  }
}

function getValidToken(): string | null {
  const token = localStorage.getItem('access_token')
  if (token && isTokenExpired(token)) {
    localStorage.removeItem('access_token')
    localStorage.removeItem('username')
    localStorage.removeItem('role')
    return null
  }
  return token
}

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
  token: getValidToken(),
  username: getValidToken() ? localStorage.getItem('username') : null,
  role: getValidToken() ? localStorage.getItem('role') as 'admin' | 'user' | null : null,

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

  isAuthenticated: () => {
    const token = get().token
    if (!token) return false
    if (isTokenExpired(token)) {
      // Token expired — clean up
      get().logout()
      return false
    }
    return true
  },
  isAdmin: () => get().role === 'admin',
}))
