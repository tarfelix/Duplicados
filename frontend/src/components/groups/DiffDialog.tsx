import { useEffect } from 'react'
import { useDiff, useExplainDiff } from '@/hooks/useGroups'
import { Loader2, X, Sparkles } from 'lucide-react'

interface Props {
  open: boolean
  onClose: () => void
  principalText: string
  principalId: string
  comparedText: string
  comparedId: string
}

export default function DiffDialog({ open, onClose, principalText, principalId, comparedText, comparedId }: Props) {
  const diff = useDiff()
  const explain = useExplainDiff()

  useEffect(() => {
    if (open) {
      diff.mutate({ text_a: principalText, text_b: comparedText })
    }
  }, [open])

  // Close on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={onClose}>
      <div
        className="bg-card rounded-lg shadow-xl w-[95vw] max-w-6xl max-h-[90vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b">
          <h3 className="font-semibold">Ver Diferenças</h3>
          <button onClick={onClose} className="p-1 rounded hover:bg-accent">
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Legend */}
        <div className="px-4 py-2 text-xs flex gap-4 border-b">
          <span>
            <strong>Legenda:</strong>
          </span>
          <span className="bg-green-200 px-2 py-0.5 rounded">Texto adicionado</span>
          <span className="bg-red-200 px-2 py-0.5 rounded">Texto removido</span>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-4">
          {diff.isPending ? (
            <div className="flex items-center justify-center h-48">
              <Loader2 className="h-6 w-6 animate-spin" />
            </div>
          ) : diff.data ? (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-xs font-medium mb-2">Texto mantido (principal) — ID {principalId}</p>
                <div
                  className="text-xs font-mono whitespace-pre-wrap bg-muted/30 rounded p-3 max-h-96 overflow-auto border"
                  dangerouslySetInnerHTML={{ __html: diff.data.html_a }}
                />
              </div>
              <div>
                <p className="text-xs font-medium mb-2">Outro item — ID {comparedId}</p>
                <div
                  className="text-xs font-mono whitespace-pre-wrap bg-muted/30 rounded p-3 max-h-96 overflow-auto border"
                  dangerouslySetInnerHTML={{ __html: diff.data.html_b }}
                />
              </div>
            </div>
          ) : null}

          {/* AI Explanation */}
          <div className="mt-4 border-t pt-4">
            <button
              onClick={() => explain.mutate({ text_a: principalText, text_b: comparedText })}
              disabled={explain.isPending}
              className="flex items-center gap-2 text-xs px-3 py-1.5 rounded border hover:bg-accent disabled:opacity-50"
            >
              <Sparkles className="h-3.5 w-3.5" />
              {explain.isPending ? 'Gerando...' : 'Explicar diferenças com IA'}
            </button>
            {explain.data?.explanation && (
              <div className="mt-3 p-3 bg-blue-50 dark:bg-blue-950/30 rounded text-sm whitespace-pre-wrap">
                {explain.data.explanation}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
