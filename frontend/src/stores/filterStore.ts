import { create } from 'zustand'
import type { GroupFilters } from '@/types'

interface FilterStore extends GroupFilters {
  setFilter: <K extends keyof GroupFilters>(key: K, value: GroupFilters[K]) => void
  setFilters: (filters: Partial<GroupFilters>) => void
}

export const useFilterStore = create<FilterStore>((set) => ({
  dias: 10,
  pastas: [],
  status: [],
  min_sim: 90,
  min_containment: 55,
  use_cnj: true,
  hide_closed: true,

  setFilter: (key, value) => set({ [key]: value }),
  setFilters: (filters) => set(filters),
}))
