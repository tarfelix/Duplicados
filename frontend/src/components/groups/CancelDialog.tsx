import { X, AlertTriangle, Loader2 } from 'lucide-react'

interface Props {
  open: boolean
  onClose: () => void
  onConfirm: () => void
  items: { activity_id: string; principal_id: string }[]
  isPending: boolean
  dryRun: boolean
}

export default function CancelDialog({ open, onClose, onConfirm, items, isPending, dryRun }: Props) {
  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={onClose}>
      <div
        className="bg-card rounded-lg shadow-xl w-full max-w-lg max-h-[80vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b">
          <h3 className="font-semibold flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-yellow-500" />
            Confirmar Cancelamento
          </h3>
          <button onClick={onClose} className="p-1 rounded hover:bg-accent">
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-4 overflow-auto">
          {dryRun && (
            <div className="bg-yellow-50 dark:bg-yellow-950/30 border border-yellow-200 dark:border-yellow-800 rounded p-3 text-sm">
              Modo Teste (Dry-run) — nenhum cancelamento será enviado à API.
            </div>
          )}
          <div className="bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800 rounded p-3 text-sm">
            Atenção: esta ação é irreversível.
          </div>
          <p className="text-sm">
            Você está prestes a cancelar <strong>{items.length}</strong> atividade(s).
          </p>

          {/* Table */}
          <div className="max-h-48 overflow-auto border rounded">
            <table className="w-full text-xs">
              <thead className="bg-muted sticky top-0">
                <tr>
                  <th className="text-left px-3 py-2">ID a Cancelar</th>
                  <th className="text-left px-3 py-2">Principal</th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {items.map((item) => (
                  <tr key={item.activity_id}>
                    <td className="px-3 py-1.5 font-mono">{item.activity_id}</td>
                    <td className="px-3 py-1.5 font-mono">{item.principal_id}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Footer */}
        <div className="flex gap-3 px-4 py-3 border-t">
          <button
            onClick={onClose}
            disabled={isPending}
            className="flex-1 rounded-md border px-4 py-2 text-sm hover:bg-accent disabled:opacity-50"
          >
            Voltar
          </button>
          <button
            onClick={onConfirm}
            disabled={isPending}
            className="flex-1 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50 flex items-center justify-center gap-2"
          >
            {isPending ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" /> Processando...
              </>
            ) : (
              'Confirmar e Processar'
            )}
          </button>
        </div>
      </div>
    </div>
  )
}
