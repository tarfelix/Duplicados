import { useState } from 'react'
import type { ActivityItem } from '@/types'
import { useGroupStates } from '@/stores/groupStates'
import SimilarityBadge from './SimilarityBadge'
import DiffDialog from './DiffDialog'
import { formatDate } from '@/lib/utils'
import { Star, Eye, ArrowUpRight } from 'lucide-react'

interface Props {
  item: ActivityItem
  groupId: string
  principalItem?: ActivityItem
  minSim: number
}

export default function ActivityCard({ item, groupId, principalItem, minSim }: Props) {
  const state = useGroupStates((s) => s.getState(groupId, principalItem?.activity_id || item.activity_id))
  const setPrincipal = useGroupStates((s) => s.setPrincipal)
  const toggleCancel = useGroupStates((s) => s.toggleCancel)
  const [diffOpen, setDiffOpen] = useState(false)

  const isPrincipal = item.activity_id === state.principalId
  const isCancelled = state.cancelados.has(item.activity_id)

  return (
    <>
      <div className={`rounded-lg border-l-4 border bg-card p-3 transition-colors ${
        isPrincipal
          ? 'border-l-green-500 bg-green-50/50 dark:bg-green-950/20'
          : isCancelled
          ? 'border-l-destructive/40 bg-muted/40 opacity-75'
          : 'border-l-border'
      }`}>
        <div className="flex gap-3">
          {/* Content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="font-mono text-[11px] text-muted-foreground">#{item.activity_id}</span>
              <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${
                item.activity_status === 'Aberta'
                  ? 'bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300'
                  : 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400'
              }`}>
                {item.activity_status || 'N/A'}
              </span>
              {isPrincipal && (
                <span className="text-[10px] bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-300 px-2 py-0.5 rounded font-medium flex items-center gap-1">
                  <Star className="h-3 w-3" /> Manter
                </span>
              )}
              {isCancelled && (
                <span className="text-[10px] bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300 px-2 py-0.5 rounded font-medium">
                  Cancelar
                </span>
              )}
            </div>
            <p className="text-[11px] text-muted-foreground mt-1">
              {formatDate(item.activity_date)} | {item.user_profile_name || 'N/A'}
            </p>

            {!isPrincipal && item.score != null && <SimilarityBadge score={item.score} details={item.score_details} threshold={minSim} />}

            <div className="mt-2 max-h-20 overflow-y-auto text-[11px] bg-muted/30 rounded-lg p-2.5 font-mono whitespace-pre-wrap leading-relaxed">
              {item.texto || '(sem texto)'}
            </div>
          </div>

          {/* Actions */}
          {!isPrincipal && (
            <div className="flex flex-col gap-1.5 shrink-0">
              <label className="flex items-center gap-1.5 text-[11px] cursor-pointer select-none group">
                <input
                  type="checkbox"
                  checked={isCancelled}
                  onChange={() => toggleCancel(groupId, item.activity_id)}
                  className="rounded w-3.5 h-3.5"
                />
                <span className="group-hover:text-foreground transition-colors">Descartar</span>
              </label>
              <button
                onClick={() => setPrincipal(groupId, item.activity_id)}
                className="text-[11px] px-2 py-1 rounded-lg border hover:bg-accent transition-colors whitespace-nowrap flex items-center gap-1"
              >
                <ArrowUpRight className="h-3 w-3" />
                Manter
              </button>
              {principalItem && (
                <button
                  onClick={() => setDiffOpen(true)}
                  className="text-[11px] px-2 py-1 rounded-lg border hover:bg-accent transition-colors flex items-center gap-1"
                >
                  <Eye className="h-3 w-3" /> Diff
                </button>
              )}
            </div>
          )}
        </div>
      </div>

      {diffOpen && principalItem && (
        <DiffDialog
          open={diffOpen}
          onClose={() => setDiffOpen(false)}
          principalText={principalItem.texto}
          principalId={principalItem.activity_id}
          comparedText={item.texto}
          comparedId={item.activity_id}
        />
      )}
    </>
  )
}
