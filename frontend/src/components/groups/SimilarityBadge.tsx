import { getSimilarityColor, getSimilarityLabel } from '@/lib/utils'

interface Props {
  score: number
  details?: Record<string, number>
  threshold: number
}

export default function SimilarityBadge({ score, details, threshold }: Props) {
  const color = getSimilarityColor(score, threshold)
  const label = getSimilarityLabel(score, threshold)

  const tooltip = details
    ? `Set ${details.set?.toFixed(0) ?? '?'}% | Sort ${details.sort?.toFixed(0) ?? '?'}% | Contain ${details.contain?.toFixed(0) ?? '?'}% | Bônus ${details.bonus ?? 0}`
    : ''

  return (
    <span className={`inline-flex items-center gap-1 mt-1.5 text-[10px] font-semibold px-2 py-0.5 rounded-full ${color}`} title={tooltip}>
      <span className="inline-block w-1.5 h-1.5 rounded-full bg-current opacity-70" />
      {label} — {score.toFixed(0)}%
    </span>
  )
}
