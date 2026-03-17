import { useState } from 'react'
import type { Group } from '@/types'
import { useGroupStates } from '@/stores/groupStates'
import ActivityCard from './ActivityCard'
import { ChevronDown, ChevronRight, CheckSquare, XCircle } from 'lucide-react'

interface Props {
  group: Group
  minSim: number
}

export default function GroupCard({ group, minSim }: Props) {
  const [expanded, setExpanded] = useState(false)
  const getState = useGroupStates((s) => s.getState)
  const markAllForCancel = useGroupStates((s) => s.markAllForCancel)
  const clearGroup = useGroupStates((s) => s.clearGroup)
  const state = getState(group.group_id, group.best_principal_id)

  const cancelCount = state.cancelados.size

  return (
    <div className="mb-2 rounded-lg border bg-card overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-accent/50 text-left"
      >
        <div className="flex items-center gap-2">
          {expanded ? <ChevronDown className="h-4 w-4 shrink-0" /> : <ChevronRight className="h-4 w-4 shrink-0" />}
          <span className="font-medium text-sm">
            {group.items.length} itens ({group.open_count} abertas)
          </span>
          <span className="text-xs text-muted-foreground">Pasta: {group.folder}</span>
          <span className="text-xs text-muted-foreground">Manter: #{state.principalId}</span>
          {cancelCount > 0 && (
            <span className="text-xs bg-destructive/10 text-destructive px-2 py-0.5 rounded">
              {cancelCount} marcados
            </span>
          )}
        </div>
      </button>

      {/* Expanded content */}
      {expanded && (
        <div className="px-4 pb-4 space-y-3">
          {/* Actions */}
          <div className="flex gap-2 text-xs">
            <button
              onClick={() => markAllForCancel(group.group_id, group.items.map((i) => i.activity_id), state.principalId)}
              className="flex items-center gap-1 px-2 py-1 rounded border hover:bg-accent"
            >
              <CheckSquare className="h-3 w-3" />
              Marcar todos
            </button>
            <button
              onClick={() => clearGroup(group.group_id)}
              className="flex items-center gap-1 px-2 py-1 rounded border hover:bg-accent"
            >
              <XCircle className="h-3 w-3" />
              Ignorar grupo
            </button>
          </div>

          <div className="border-t pt-3 space-y-2">
            <p className="text-xs text-muted-foreground">
              1. Escolha qual item manter — 2. Marque os repetidos — 3. Use "Processar Marcados" abaixo
            </p>
            {group.items.map((item) => (
              <ActivityCard
                key={item.activity_id}
                item={item}
                groupId={group.group_id}
                principalItem={group.items.find((i) => i.activity_id === state.principalId)}
                minSim={minSim}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
