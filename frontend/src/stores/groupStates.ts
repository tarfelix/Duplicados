import { create } from 'zustand'

interface GroupState {
  principalId: string
  cancelados: Set<string>
}

interface GroupStatesStore {
  states: Record<string, GroupState>
  ignoredGroups: Set<string>
  getState: (groupId: string, defaultPrincipalId: string) => GroupState
  setPrincipal: (groupId: string, principalId: string) => void
  toggleCancel: (groupId: string, activityId: string) => void
  markAllForCancel: (groupId: string, allIds: string[], principalId: string) => void
  clearGroup: (groupId: string) => void
  ignoreGroup: (groupId: string) => void
  unignoreGroup: (groupId: string) => void
  isIgnored: (groupId: string) => boolean
  getIgnoredCount: () => number
  clearAll: () => void
  getMarkedCount: () => number
  getAllCancelItems: () => { activity_id: string; principal_id: string }[]
}

export const useGroupStates = create<GroupStatesStore>((set, get) => ({
  states: {},
  ignoredGroups: new Set(),

  getState: (groupId, defaultPrincipalId) => {
    const existing = get().states[groupId]
    if (existing) return existing
    const newState: GroupState = { principalId: defaultPrincipalId, cancelados: new Set() }
    set((s) => ({ states: { ...s.states, [groupId]: newState } }))
    return newState
  },

  setPrincipal: (groupId, principalId) => {
    set((s) => {
      const current = s.states[groupId] || { principalId, cancelados: new Set() }
      const cancelados = new Set(current.cancelados)
      cancelados.delete(principalId)
      return { states: { ...s.states, [groupId]: { principalId, cancelados } } }
    })
  },

  toggleCancel: (groupId, activityId) => {
    set((s) => {
      const current = s.states[groupId]
      if (!current) return s
      const cancelados = new Set(current.cancelados)
      if (cancelados.has(activityId)) {
        cancelados.delete(activityId)
      } else {
        cancelados.add(activityId)
      }
      return { states: { ...s.states, [groupId]: { ...current, cancelados } } }
    })
  },

  markAllForCancel: (groupId, allIds, principalId) => {
    set((s) => {
      const cancelados = new Set(allIds.filter((id) => id !== principalId))
      return {
        states: { ...s.states, [groupId]: { principalId, cancelados } },
      }
    })
  },

  clearGroup: (groupId) => {
    set((s) => {
      const { [groupId]: _, ...rest } = s.states
      return { states: rest }
    })
  },

  ignoreGroup: (groupId) => {
    set((s) => {
      const { [groupId]: _, ...rest } = s.states
      const ignored = new Set(s.ignoredGroups)
      ignored.add(groupId)
      return { states: rest, ignoredGroups: ignored }
    })
  },

  unignoreGroup: (groupId) => {
    set((s) => {
      const ignored = new Set(s.ignoredGroups)
      ignored.delete(groupId)
      return { ignoredGroups: ignored }
    })
  },

  isIgnored: (groupId) => get().ignoredGroups.has(groupId),

  getIgnoredCount: () => get().ignoredGroups.size,

  clearAll: () => set({ states: {}, ignoredGroups: new Set() }),

  getMarkedCount: () => {
    const states = get().states
    return Object.values(states).reduce((acc, s) => acc + s.cancelados.size, 0)
  },

  getAllCancelItems: () => {
    const states = get().states
    const items: { activity_id: string; principal_id: string }[] = []
    for (const state of Object.values(states)) {
      for (const activityId of state.cancelados) {
        items.push({ activity_id: activityId, principal_id: state.principalId })
      }
    }
    return items
  },
}))
