import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatDate(dateStr?: string): string {
  if (!dateStr) return 'N/A'
  try {
    const d = new Date(dateStr)
    if (isNaN(d.getTime())) return 'N/A'
    return d.toLocaleDateString('pt-BR', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  } catch {
    return 'N/A'
  }
}

export function getSimilarityColor(score: number, threshold: number): string {
  if (score >= 95) return 'bg-green-100 text-green-800 dark:bg-green-950/50 dark:text-green-300'
  if (score >= threshold) return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-950/50 dark:text-yellow-300'
  return 'bg-red-100 text-red-800 dark:bg-red-950/50 dark:text-red-300'
}

export function getSimilarityLabel(score: number, threshold: number): string {
  if (score >= 95) return 'Muito parecido'
  if (score >= threshold) return 'Parecido'
  return 'Atenção'
}
