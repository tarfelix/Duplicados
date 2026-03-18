import { useGroups, useCancelActivities } from '@/hooks/useGroups'
import { useGroupStates } from '@/stores/groupStates'
import { useFilterStore } from '@/stores/filterStore'
import GroupList from '@/components/groups/GroupList'
import CancelDialog from '@/components/groups/CancelDialog'
import { Download, Rocket, Loader2, BarChart3, FolderOpen, CheckCircle2, AlertTriangle, EyeOff } from 'lucide-react'
import { useState } from 'react'
import { toast } from 'sonner'
import api from '@/api/client'

export default function DashboardPage() {
  const { data, isLoading, error } = useGroups()
  const markedCount = useGroupStates((s) => s.getMarkedCount())
  const getAllCancelItems = useGroupStates((s) => s.getAllCancelItems)
  const clearAll = useGroupStates((s) => s.clearAll)
  const ignoredCount = useGroupStates((s) => s.getIgnoredCount())
  const cancelMutation = useCancelActivities()
  const filters = useFilterStore()
  const [cancelOpen, setCancelOpen] = useState(false)
  const [dryRun, setDryRun] = useState(false)

  const handleExport = async () => {
    try {
      const params = new URLSearchParams()
      params.set('dias', String(filters.dias))
      params.set('min_sim', String(filters.min_sim))
      params.set('min_containment', String(filters.min_containment))
      params.set('use_cnj', String(filters.use_cnj))
      params.set('hide_closed', String(filters.hide_closed))
      if (filters.pastas.length) params.set('pastas', filters.pastas.join(','))
      if (filters.status.length) params.set('status', filters.status.join(','))

      const res = await api.get(`/api/groups/export-csv?${params}`, { responseType: 'blob' })
      const url = window.URL.createObjectURL(new Blob([res.data]))
      const a = document.createElement('a')
      a.href = url
      a.download = 'duplicatas.csv'
      a.click()
      window.URL.revokeObjectURL(url)
      toast.success('CSV exportado!')
    } catch {
      toast.error('Erro ao exportar CSV.')
    }
  }

  const handleCancel = () => {
    const items = getAllCancelItems()
    if (items.length === 0) {
      toast.info('Nenhuma atividade marcada para cancelar.')
      return
    }
    setCancelOpen(true)
  }

  const handleConfirmCancel = () => {
    const items = getAllCancelItems()
    cancelMutation.mutate(
      { items, dry_run: dryRun },
      {
        onSuccess: (result) => {
          toast.success(`Concluído! Sucessos: ${result.ok}, Falhas: ${result.err}`)
          if (dryRun) toast.info('Modo Teste — nada foi alterado.')
          clearAll()
          setCancelOpen(false)
        },
        onError: (err: any) => {
          toast.error(err.response?.data?.detail || 'Erro ao processar cancelamentos.')
        },
      }
    )
  }

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-3">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <span className="text-sm text-muted-foreground">Carregando atividades...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-3">
        <AlertTriangle className="h-8 w-8 text-destructive" />
        <p className="text-sm text-destructive">Erro ao carregar dados: {(error as Error).message}</p>
      </div>
    )
  }

  if (!data || data.groups.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-3">
        <FolderOpen className="h-10 w-10 text-muted-foreground/50" />
        <p className="text-muted-foreground">Nenhum grupo de duplicatas encontrado.</p>
        <p className="text-xs text-muted-foreground">Ajuste os filtros na barra lateral.</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="rounded-xl border bg-card p-4 flex items-center gap-3">
          <div className="p-2 rounded-lg bg-primary/10">
            <BarChart3 className="h-5 w-5 text-primary" />
          </div>
          <div>
            <p className="text-2xl font-bold">{data.total_groups}</p>
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Grupos</p>
          </div>
        </div>
        <div className="rounded-xl border bg-card p-4 flex items-center gap-3">
          <div className="p-2 rounded-lg bg-blue-500/10">
            <FolderOpen className="h-5 w-5 text-blue-500" />
          </div>
          <div>
            <p className="text-2xl font-bold">{data.total_abertas}</p>
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Abertas</p>
          </div>
        </div>
        <div className="rounded-xl border bg-card p-4 flex items-center gap-3">
          <div className="p-2 rounded-lg bg-green-500/10">
            <CheckCircle2 className="h-5 w-5 text-green-500" />
          </div>
          <div>
            <p className="text-2xl font-bold">{markedCount}</p>
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Marcados</p>
          </div>
        </div>
        <div className="rounded-xl border bg-card p-4 flex items-center gap-3">
          <div className="p-2 rounded-lg bg-muted">
            <EyeOff className="h-5 w-5 text-muted-foreground" />
          </div>
          <div>
            <p className="text-2xl font-bold">{ignoredCount}</p>
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Ignorados</p>
          </div>
        </div>
      </div>

      {/* Controls bar */}
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <label className="flex items-center gap-2 text-sm cursor-pointer select-none">
          <input
            type="checkbox"
            checked={dryRun}
            onChange={(e) => setDryRun(e.target.checked)}
            className="rounded w-4 h-4"
          />
          <span>Modo Teste</span>
          {dryRun && <span className="text-xs text-yellow-600 dark:text-yellow-400 font-medium">(nenhum cancelamento será enviado)</span>}
        </label>

        <div className="flex gap-2">
          <button
            onClick={handleExport}
            className="flex items-center gap-2 rounded-lg border bg-card px-3 py-2 text-xs font-medium hover:bg-accent transition-colors"
          >
            <Download className="h-3.5 w-3.5" />
            CSV
          </button>
          <button
            onClick={handleCancel}
            disabled={markedCount === 0}
            className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-xs font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-40 transition-colors"
          >
            <Rocket className="h-3.5 w-3.5" />
            Processar ({markedCount})
          </button>
        </div>
      </div>

      {/* Group list */}
      <GroupList groups={data.groups} minSim={filters.min_sim} />

      {/* Cancel confirmation dialog */}
      <CancelDialog
        open={cancelOpen}
        onClose={() => setCancelOpen(false)}
        onConfirm={handleConfirmCancel}
        items={getAllCancelItems()}
        isPending={cancelMutation.isPending}
        dryRun={dryRun}
      />
    </div>
  )
}
