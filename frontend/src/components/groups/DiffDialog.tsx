import { useEffect, useCallback } from 'react'
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, principalText, comparedText])

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape') onClose()
  }, [onClose])

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div
        className="bg-card rounded-xl shadow-2xl w-[95vw] max-w-6xl max-h-[90vh] flex flex-col border"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b">
          <h3 className="font-semibold text-sm">Comparar Textos</h3>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-accent transition-colors">
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Legend */}
        <div className="px-5 py-2 text-[11px] flex gap-4 border-b bg-muted/30">
          <span className="font-medium">Legenda:</span>
          <span className="diff-ins px-2 py-0.5 rounded">Adicionado</span>
          <span className="diff-del px-2 py-0.5 rounded">Removido</span>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-5">
          {diff.isPending ? (
            <div className="flex items-center justify-center h-48 gap-2">
              <Loader2 className="h-5 w-5 animate-spin text-primary" />
              <span className="text-sm text-muted-foreground">Calculando diferenças...</span>
            </div>
          ) : diff.isError ? (
            <div className="text-center text-sm text-destructive py-8">Erro ao gerar diff.</div>
          ) : diff.data ? (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-[11px] font-medium mb-2 text-green-700 dark:text-green-400">Texto mantido (principal) — #{principalId}</p>
                <div
                  className="text-[11px] font-mono whitespace-pre-wrap bg-muted/20 rounded-lg p-3 max-h-96 overflow-auto border leading-relaxed"
                  dangerouslySetInnerHTML={{ __html: diff.data.html_a }}
                />
              </div>
              <div>
                <p className="text-[11px] font-medium mb-2 text-muted-foreground">Outro item — #{comparedId}</p>
                <div
                  className="text-[11px] font-mono whitespace-pre-wrap bg-muted/20 rounded-lg p-3 max-h-96 overflow-auto border leading-relaxed"
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
              className="flex items-center gap-2 text-xs px-3 py-2 rounded-lg border hover:bg-accent transition-colors disabled:opacity-50"
            >
              <Sparkles className="h-3.5 w-3.5 text-primary" />
              {explain.isPending ? 'Gerando...' : 'Explicar diferenças com IA'}
            </button>
            {explain.data?.explanation && (
              <div className="mt-3 p-4 bg-primary/5 border border-primary/10 rounded-lg text-sm whitespace-pre-wrap leading-relaxed">
                {explain.data.explanation}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
