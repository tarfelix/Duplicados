import { useState } from 'react'
import type { ActivityItem } from '@/types'
import { useGroupStates } from '@/stores/groupStates'
import SimilarityBadge from './SimilarityBadge'
import DiffDialog from './DiffDialog'
import { formatDate } from '@/lib/utils'
import { Star, Eye } from 'lucide-react'

interface Props {
  item: ActivityItem
  groupId: string
  principalItem?: ActivityItem
  minSim: number
}

export default function ActivityCard({ item, groupId, principalItem, minSim }: Props) {
  const state = useGroupStates((s) => s.getState(groupId, ''))
  const setPrincipal = useGroupStates((s) => s.setPrincipal)
  const toggleCancel = useGroupStates((s) => s.toggleCancel)
  const [diffOpen, setDiffOpen] = useState(false)

  const isPrincipal = item.activity_id === state.principalId
  const isCancelled = state.cancelados.has(item.activity_id)

  const borderColor = isPrincipal ? 'border-l-green-500' : isCancelled ? 'border-l-gray-300' : 'border-l-gray-200'
  const bgColor = isCancelled ? 'bg-muted/50' : ''

  return (
    <>
      <div className={`rounded-md border border-l-4 ${borderColor} ${bgColor} p-3`}>
        <div className="flex gap-3">
          {/* Content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="font-mono text-xs">ID: {item.activity_id}</span>
              {isPrincipal && (
                <span className="text-xs bg-green-100 text-green-800 px-2 py-0.5 rounded flex items-center gap-1">
                  <Star className="h-3 w-3" /> Manter
                </span>
              )}
              {isCancelled && (
                <span className="text-xs bg-red-100 text-red-800 px-2 py-0.5 rounded">Marcado para cancelar</span>
              )}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              {formatDate(item.activity_date)} | {item.activity_status} | {item.user_profile_name}
            </p>

            {!isPrincipal && item.score != null && <SimilarityBadge score={item.score} details={item.score_details} threshold={minSim} />}

            <div className="mt-2 max-h-24 overflow-y-auto text-xs bg-muted/30 rounded p-2 font-mono whitespace-pre-wrap">
              {item.texto || '(sem texto)'}
            </div>
          </div>

          {/* Actions */}
          {!isPrincipal && (
            <div className="flex flex-col gap-1.5 shrink-0">
              <label className="flex items-center gap-1.5 text-xs cursor-pointer">
                <input
                  type="checkbox"
                  checked={isCancelled}
                  onChange={() => toggleCancel(groupId, item.activity_id)}
                  className="rounded"
                />
                Descartar
              </label>
              <button
                onClick={() => setPrincipal(groupId, item.activity_id)}
                className="text-xs px-2 py-1 rounded border hover:bg-accent whitespace-nowrap"
              >
                Manter este
              </button>
              {principalItem && (
                <button
                  onClick={() => setDiffOpen(true)}
                  className="text-xs px-2 py-1 rounded border hover:bg-accent flex items-center gap-1"
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
