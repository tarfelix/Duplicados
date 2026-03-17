export interface LoginRequest {
  username: string
  password: string
}

export interface TokenResponse {
  access_token: string
  token_type: string
  username: string
  role: 'admin' | 'user'
}

export interface User {
  username: string
  role: 'admin' | 'user'
  created_at?: string
}

export interface ActivityItem {
  activity_id: string
  activity_folder?: string
  user_profile_name?: string
  activity_date?: string
  activity_status?: string
  texto: string
  score?: number
  score_details?: Record<string, number>
  is_principal: boolean
}

export interface Group {
  group_id: string
  items: ActivityItem[]
  folder?: string
  open_count: number
  best_principal_id: string
}

export interface GroupsResponse {
  groups: Group[]
  total_groups: number
  total_abertas: number
}

export interface Filters {
  pastas: string[]
  status: string[]
}

export interface CancelItem {
  activity_id: string
  principal_id: string
}

export interface CancelResult {
  ok: number
  err: number
  details: Record<string, any>[]
}

export interface DiffResponse {
  html_a: string
  html_b: string
}

export interface GroupFilters {
  dias: number
  pastas: string[]
  status: string[]
  min_sim: number
  min_containment: number
  use_cnj: boolean
  hide_closed: boolean
}
