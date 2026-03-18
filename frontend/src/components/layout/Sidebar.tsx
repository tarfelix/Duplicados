import { useActivityFilters } from '@/hooks/useFilters'
import { useFilterStore } from '@/stores/filterStore'
import { X, RefreshCw, HelpCircle, SlidersHorizontal } from 'lucide-react'
import { useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'

export default function Sidebar({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { data: filterOptions, isError } = useActivityFilters()
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
          <div className="flex items-center gap-2">
            <SlidersHorizontal className="h-4 w-4 text-primary" />
            <h2 className="font-semibold text-sm">Filtros</h2>
          </div>
          <div className="flex gap-1">
            <button onClick={handleRefresh} className="p-1.5 rounded hover:bg-accent transition-colors" title="Atualizar dados">
              <RefreshCw className="h-4 w-4" />
            </button>
            <button onClick={onClose} className="p-1.5 rounded hover:bg-accent md:hidden transition-colors">
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        <div className="flex-1 p-4 space-y-5">
          {isError && (
            <div className="text-xs text-destructive bg-destructive/10 rounded-lg p-2.5">
              Erro ao carregar filtros. Verifique a conexão MySQL.
            </div>
          )}

          {/* Dias */}
          <div>
            <label htmlFor="filter-dias" className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
              Dias de Histórico
            </label>
            <input
              id="filter-dias"
              type="number"
              min={1}
              max={365}
              value={filters.dias}
              onChange={(e) => {
                const v = Number(e.target.value)
                if (v >= 1 && v <= 365) filters.setFilter('dias', v)
              }}
              className="mt-1.5 w-full rounded-lg border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 transition-colors"
            />
          </div>

          {/* Pastas */}
          <div>
            <label className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
              Pastas {filters.pastas.length > 0 && <span className="text-primary">({filters.pastas.length})</span>}
            </label>
            <select
              multiple
              value={filters.pastas}
              onChange={(e) => filters.setFilter('pastas', Array.from(e.target.selectedOptions, (o) => o.value))}
              className="mt-1.5 w-full rounded-lg border bg-background px-2 py-1.5 text-sm h-24 focus:outline-none focus:ring-2 focus:ring-primary/50 transition-colors"
            >
              {filterOptions?.pastas.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </div>

          {/* Status */}
          <div>
            <label className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
              Status {filters.status.length > 0 && <span className="text-primary">({filters.status.length})</span>}
            </label>
            <select
              multiple
              value={filters.status}
              onChange={(e) => filters.setFilter('status', Array.from(e.target.selectedOptions, (o) => o.value))}
              className="mt-1.5 w-full rounded-lg border bg-background px-2 py-1.5 text-sm h-24 focus:outline-none focus:ring-2 focus:ring-primary/50 transition-colors"
            >
              {filterOptions?.status.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>

          {/* Similarity slider */}
          <div>
            <label className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
              Similaridade mínima
            </label>
            <div className="flex items-center gap-2 mt-1.5">
              <input
                type="range"
                min={0}
                max={100}
                value={filters.min_sim}
                onChange={(e) => filters.setFilter('min_sim', Number(e.target.value))}
                className="flex-1"
              />
              <span className="text-sm font-semibold text-primary w-10 text-right">{filters.min_sim}%</span>
            </div>
          </div>

          {/* Containment slider */}
          <div>
            <label className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
              Containment mínimo
            </label>
            <div className="flex items-center gap-2 mt-1.5">
              <input
                type="range"
                min={0}
                max={100}
                value={filters.min_containment}
                onChange={(e) => filters.setFilter('min_containment', Number(e.target.value))}
                className="flex-1"
              />
              <span className="text-sm font-semibold text-primary w-10 text-right">{filters.min_containment}%</span>
            </div>
          </div>

          {/* Toggles */}
          <div className="space-y-3">
            <label className="flex items-center gap-2.5 text-sm cursor-pointer group">
              <input
                type="checkbox"
                checked={filters.use_cnj}
                onChange={(e) => filters.setFilter('use_cnj', e.target.checked)}
                className="rounded w-4 h-4"
              />
              <span className="group-hover:text-foreground transition-colors">Filtrar por CNJ</span>
            </label>
            <label className="flex items-center gap-2.5 text-sm cursor-pointer group">
              <input
                type="checkbox"
                checked={filters.hide_closed}
                onChange={(e) => filters.setFilter('hide_closed', e.target.checked)}
                className="rounded w-4 h-4"
              />
              <span className="group-hover:text-foreground transition-colors">Ocultar grupos sem abertas</span>
            </label>
          </div>

          {/* Help */}
          <div>
            <button onClick={() => setHelpOpen(!helpOpen)} className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors">
              <HelpCircle className="h-3.5 w-3.5" />
              Como funciona?
            </button>
            {helpOpen && (
              <div className="mt-2 text-xs text-muted-foreground space-y-1.5 bg-muted p-3 rounded-lg">
                <p><strong>Duplicata:</strong> mesma publicação apareceu mais de uma vez.</p>
                <p><strong>Principal:</strong> o item mantido; os outros podem ser cancelados.</p>
                <p><strong>Similaridade:</strong> quanto maior o %, mais parecidos os textos.</p>
                <p><strong>Containment:</strong> % de palavras em comum entre os textos.</p>
              </div>
            )}
          </div>
        </div>

        <div className="p-4 border-t">
          <div className="flex items-center gap-2">
            <img src="/logo.png" alt="" className="h-5 opacity-50" />
            <span className="text-[10px] text-muted-foreground">v2.1 | Soares Picon</span>
          </div>
        </div>
      </aside>
    </>
  )
}
