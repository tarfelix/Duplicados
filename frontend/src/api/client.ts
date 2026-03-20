import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '',
  headers: { 'Content-Type': 'application/json' },
})

// Request interceptor: attach JWT
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Response interceptor: handle auth errors (401 expired token, 403 missing token)
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401 || error.response?.status === 403) {
      // Only redirect if this was an auth failure on a protected endpoint
      // (not the login/setup/has-users endpoints themselves)
      const url = error.config?.url || ''
      const isAuthEndpoint = url.includes('/api/auth/login') || url.includes('/api/auth/setup') || url.includes('/api/auth/has-users')
      if (!isAuthEndpoint) {
        localStorage.removeItem('access_token')
        localStorage.removeItem('username')
        localStorage.removeItem('role')
        window.location.href = '/login'
      }
    }
    return Promise.reject(error)
  }
)

export default api
