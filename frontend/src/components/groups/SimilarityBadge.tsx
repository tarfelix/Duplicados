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
    <span className={`inline-block mt-1 text-xs font-semibold px-2 py-0.5 rounded ${color}`} title={tooltip}>
      {label} — {score.toFixed(0)}%
    </span>
  )
}
