import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '@/api/client'
import { useFilterStore } from '@/stores/filterStore'
import type { GroupsResponse, CancelItem, CancelResult, DiffResponse } from '@/types'

export function useGroups() {
  const filters = useFilterStore()
  const params = new URLSearchParams()
  params.set('dias', String(filters.dias))
  params.set('min_sim', String(filters.min_sim))
  params.set('min_containment', String(filters.min_containment))
  params.set('use_cnj', String(filters.use_cnj))
  params.set('hide_closed', String(filters.hide_closed))
  if (filters.pastas.length > 0) params.set('pastas', filters.pastas.join(','))
  if (filters.status.length > 0) params.set('status', filters.status.join(','))

  return useQuery({
    queryKey: ['groups', filters.dias, filters.pastas, filters.status, filters.min_sim, filters.min_containment, filters.use_cnj, filters.hide_closed],
    queryFn: async () => {
      const res = await api.get<GroupsResponse>(`/api/groups?${params.toString()}`)
      return res.data
    },
    staleTime: 60_000,
    refetchOnWindowFocus: false,
  })
}

export function useCancelActivities() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: async (data: { items: CancelItem[]; dry_run: boolean }) => {
      const res = await api.post<CancelResult>('/api/groups/cancel', data)
      return res.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['groups'] })
    },
  })
}

export function useDiff() {
  return useMutation({
    mutationFn: async (data: { text_a: string; text_b: string; limit?: number }) => {
      const res = await api.post<DiffResponse>('/api/groups/diff', data)
      return res.data
    },
  })
}

export function useExplainDiff() {
  return useMutation({
    mutationFn: async (data: { text_a: string; text_b: string }) => {
      const res = await api.post<{ explanation: string | null }>('/api/groups/explain-diff', data)
      return res.data
    },
  })
}
