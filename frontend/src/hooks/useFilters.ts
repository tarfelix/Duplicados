import { useQuery } from '@tanstack/react-query'
import api from '@/api/client'
import type { Filters } from '@/types'

export function useActivityFilters() {
  return useQuery({
    queryKey: ['filters'],
    queryFn: async () => {
      const res = await api.get<Filters>('/api/activities/filters')
      return res.data
    },
    staleTime: 300_000,
  })
}
