import { useGroups, useCancelActivities } from '@/hooks/useGroups'
import { useGroupStates } from '@/stores/groupStates'
import { useFilterStore } from '@/stores/filterStore'
import GroupList from '@/components/groups/GroupList'
import CancelDialog from '@/components/groups/CancelDialog'
import { Download, Rocket, Loader2 } from 'lucide-react'
import { useState } from 'react'
import { toast } from 'sonner'
import api from '@/api/client'

export default function DashboardPage() {
  const { data, isLoading, error } = useGroups()
  const markedCount = useGroupStates((s) => s.getMarkedCount())
  const getAllCancelItems = useGroupStates((s) => s.getAllCancelItems)
  const clearAll = useGroupStates((s) => s.clearAll)
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
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        <span className="ml-3 text-muted-foreground">Carregando atividades...</span>
      </div>
    )
  }

  if (error) {
    return <div className="text-center text-destructive py-8">Erro ao carregar dados: {(error as Error).message}</div>
  }

  if (!data || data.groups.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        <p className="text-lg">Nenhum grupo de duplicatas encontrado.</p>
        <p className="text-sm mt-2">Ajuste os filtros na barra lateral.</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Metrics */}
      <div className="grid grid-cols-3 gap-4">
        <div className="rounded-lg border bg-card p-4 text-center">
          <p className="text-2xl font-bold">{data.total_groups}</p>
          <p className="text-xs text-muted-foreground uppercase">Grupos</p>
        </div>
        <div className="rounded-lg border bg-card p-4 text-center">
          <p className="text-2xl font-bold">{data.total_abertas}</p>
          <p className="text-xs text-muted-foreground uppercase">Abertas</p>
        </div>
        <div className="rounded-lg border bg-card p-4 text-center">
          <p className="text-2xl font-bold">{markedCount}</p>
          <p className="text-xs text-muted-foreground uppercase">Marcados</p>
        </div>
      </div>

      {/* Dry run toggle */}
      <label className="flex items-center gap-2 text-sm">
        <input
          type="checkbox"
          checked={dryRun}
          onChange={(e) => setDryRun(e.target.checked)}
          className="rounded"
        />
        Modo Teste (Dry-run)
        {dryRun && <span className="text-yellow-600 font-medium">— nenhum cancelamento será enviado</span>}
      </label>

      {/* Group list */}
      <GroupList groups={data.groups} minSim={filters.min_sim} />

      {/* Bottom actions */}
      <div className="flex gap-3 pt-4 border-t">
        <button
          onClick={handleExport}
          className="flex-1 flex items-center justify-center gap-2 rounded-md border bg-card px-4 py-2.5 text-sm font-medium hover:bg-accent"
        >
          <Download className="h-4 w-4" />
          Baixar CSV
        </button>
        <button
          onClick={handleCancel}
          className="flex-1 flex items-center justify-center gap-2 rounded-md bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90"
        >
          <Rocket className="h-4 w-4" />
          Processar Marcados ({markedCount})
        </button>
      </div>

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
