import { useState } from 'react'
import type { Group } from '@/types'
import { useGroupStates } from '@/stores/groupStates'
import ActivityCard from './ActivityCard'
import { ChevronDown, ChevronRight, CheckSquare, EyeOff, Eye } from 'lucide-react'

interface Props {
  group: Group
  minSim: number
}

export default function GroupCard({ group, minSim }: Props) {
  const [expanded, setExpanded] = useState(false)
  const getState = useGroupStates((s) => s.getState)
  const markAllForCancel = useGroupStates((s) => s.markAllForCancel)
  const ignoreGroup = useGroupStates((s) => s.ignoreGroup)
  const unignoreGroup = useGroupStates((s) => s.unignoreGroup)
  const isIgnored = useGroupStates((s) => s.isIgnored(group.group_id))
  const state = getState(group.group_id, group.best_principal_id)

  const cancelCount = state.cancelados.size

  if (isIgnored && !expanded) {
    return (
      <div className="mb-2 rounded-xl border border-dashed bg-muted/30 overflow-hidden opacity-60">
        <button
          onClick={() => setExpanded(true)}
          className="w-full flex items-center justify-between px-4 py-2.5 hover:bg-accent/30 text-left"
        >
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <EyeOff className="h-3.5 w-3.5" />
            <span>Grupo ignorado — {group.items.length} itens | Pasta: {group.folder || 'N/A'}</span>
          </div>
          <button
            onClick={(e) => { e.stopPropagation(); unignoreGroup(group.group_id) }}
            className="text-[10px] px-2 py-0.5 rounded border hover:bg-accent"
          >
            Restaurar
          </button>
        </button>
      </div>
    )
  }

  return (
    <div className={`mb-2 rounded-xl border bg-card overflow-hidden transition-shadow hover:shadow-sm ${isIgnored ? 'opacity-60' : ''}`}>
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-accent/30 text-left transition-colors"
      >
        <div className="flex items-center gap-2 flex-wrap">
          {expanded ? <ChevronDown className="h-4 w-4 shrink-0 text-primary" /> : <ChevronRight className="h-4 w-4 shrink-0 text-muted-foreground" />}
          <span className="font-medium text-sm">
            {group.items.length} itens
            <span className="text-primary ml-1">({group.open_count} abertas)</span>
          </span>
          <span className="text-xs text-muted-foreground bg-muted px-2 py-0.5 rounded">
            {group.folder || 'N/A'}
          </span>
          {cancelCount > 0 && (
            <span className="text-xs bg-destructive/10 text-destructive px-2 py-0.5 rounded font-medium">
              {cancelCount} marcados
            </span>
          )}
        </div>
      </button>

      {/* Expanded content */}
      {expanded && (
        <div className="px-4 pb-4 space-y-3">
          {/* Actions */}
          <div className="flex gap-2 text-xs flex-wrap">
            <button
              onClick={() => markAllForCancel(group.group_id, group.items.map((i) => i.activity_id), state.principalId)}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border hover:bg-accent transition-colors"
            >
              <CheckSquare className="h-3 w-3" />
              Marcar todos
            </button>
            <button
              onClick={() => ignoreGroup(group.group_id)}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border hover:bg-accent transition-colors"
            >
              <EyeOff className="h-3 w-3" />
              Ignorar grupo
            </button>
            {isIgnored && (
              <button
                onClick={() => unignoreGroup(group.group_id)}
                className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border hover:bg-accent transition-colors text-primary"
              >
                <Eye className="h-3 w-3" />
                Restaurar
              </button>
            )}
          </div>

          <div className="border-t pt-3 space-y-2">
            <p className="text-[10px] text-muted-foreground bg-muted/50 rounded-lg px-3 py-2">
              1. Escolha qual item <strong>manter</strong> — 2. Marque os <strong>repetidos</strong> — 3. Use <strong>"Processar"</strong> acima
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
