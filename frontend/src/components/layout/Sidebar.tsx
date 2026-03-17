import { useActivityFilters } from '@/hooks/useFilters'
import { useFilterStore } from '@/stores/filterStore'
import { X, RefreshCw, HelpCircle } from 'lucide-react'
import { useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'

export default function Sidebar({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { data: filterOptions } = useActivityFilters()
  const filters = useFilterStore()
  const queryClient = useQueryClient()
  const [helpOpen, setHelpOpen] = useState(false)

  const handleRefresh = () => {
    queryClient.invalidateQueries({ queryKey: ['groups'] })
    queryClient.invalidateQueries({ queryKey: ['filters'] })
  }

  return (
    <>
      {open && <div className="fixed inset-0 bg-black/50 z-30 md:hidden" onClick={onClose} />}
      <aside
        className={`fixed md:static inset-y-0 left-0 z-40 w-72 bg-card border-r transform transition-transform duration-200 ${
          open ? 'translate-x-0' : '-translate-x-full md:translate-x-0'
        } flex flex-col overflow-y-auto`}
      >
        <div className="p-4 border-b flex items-center justify-between">
          <h2 className="font-semibold text-sm">Filtros</h2>
          <div className="flex gap-1">
            <button onClick={handleRefresh} className="p-1.5 rounded hover:bg-accent" title="Atualizar dados">
              <RefreshCw className="h-4 w-4" />
            </button>
            <button onClick={onClose} className="p-1.5 rounded hover:bg-accent md:hidden">
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        <div className="flex-1 p-4 space-y-5">
          {/* Dias */}
          <div>
            <label className="text-xs font-medium text-muted-foreground uppercase">Dias de Histórico</label>
            <input
              type="number"
              min={1}
              max={365}
              value={filters.dias}
              onChange={(e) => filters.setFilter('dias', Number(e.target.value) || 10)}
              className="mt-1 w-full rounded-md border bg-background px-3 py-1.5 text-sm"
            />
          </div>

          {/* Pastas */}
          <div>
            <label className="text-xs font-medium text-muted-foreground uppercase">Pastas</label>
            <select
              multiple
              value={filters.pastas}
              onChange={(e) => filters.setFilter('pastas', Array.from(e.target.selectedOptions, (o) => o.value))}
              className="mt-1 w-full rounded-md border bg-background px-2 py-1.5 text-sm h-24"
            >
              {filterOptions?.pastas.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </div>

          {/* Status */}
          <div>
            <label className="text-xs font-medium text-muted-foreground uppercase">Status</label>
            <select
              multiple
              value={filters.status}
              onChange={(e) => filters.setFilter('status', Array.from(e.target.selectedOptions, (o) => o.value))}
              className="mt-1 w-full rounded-md border bg-background px-2 py-1.5 text-sm h-24"
            >
              {filterOptions?.status.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>

          {/* Similarity slider */}
          <div>
            <label className="text-xs font-medium text-muted-foreground uppercase">
              Similaridade mínima: {filters.min_sim}%
            </label>
            <input
              type="range"
              min={0}
              max={100}
              value={filters.min_sim}
              onChange={(e) => filters.setFilter('min_sim', Number(e.target.value))}
              className="mt-1 w-full"
            />
          </div>

          {/* Containment slider */}
          <div>
            <label className="text-xs font-medium text-muted-foreground uppercase">
              Containment mínimo: {filters.min_containment}%
            </label>
            <input
              type="range"
              min={0}
              max={100}
              value={filters.min_containment}
              onChange={(e) => filters.setFilter('min_containment', Number(e.target.value))}
              className="mt-1 w-full"
            />
          </div>

          {/* Toggles */}
          <div className="space-y-3">
            <label className="flex items-center gap-2 text-sm cursor-pointer">
              <input
                type="checkbox"
                checked={filters.use_cnj}
                onChange={(e) => filters.setFilter('use_cnj', e.target.checked)}
                className="rounded"
              />
              Filtrar por CNJ
            </label>
            <label className="flex items-center gap-2 text-sm cursor-pointer">
              <input
                type="checkbox"
                checked={filters.hide_closed}
                onChange={(e) => filters.setFilter('hide_closed', e.target.checked)}
                className="rounded"
              />
              Ocultar grupos sem atividades abertas
            </label>
          </div>

          {/* Help */}
          <div>
            <button onClick={() => setHelpOpen(!helpOpen)} className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground">
              <HelpCircle className="h-3.5 w-3.5" />
              Como funciona?
            </button>
            {helpOpen && (
              <div className="mt-2 text-xs text-muted-foreground space-y-1 bg-muted p-3 rounded-md">
                <p><strong>Duplicata:</strong> mesma publicação apareceu mais de uma vez.</p>
                <p><strong>Principal:</strong> o item mantido; os outros podem ser cancelados.</p>
                <p><strong>Similaridade:</strong> quanto maior o %, mais parecidos os textos.</p>
              </div>
            )}
          </div>
        </div>

        <div className="p-4 border-t text-xs text-muted-foreground">
          v2.0 React | Coolify Ready
        </div>
      </aside>
    </>
  )
}
