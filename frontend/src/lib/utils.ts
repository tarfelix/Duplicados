import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatDate(dateStr?: string): string {
  if (!dateStr) return 'N/A'
  try {
    const d = new Date(dateStr)
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
  if (score >= threshold + 5) return 'bg-green-200 text-green-900'
  if (score >= threshold) return 'bg-yellow-200 text-yellow-900'
  return 'bg-red-200 text-red-900'
}

export function getSimilarityLabel(score: number, threshold: number): string {
  if (score >= 95) return 'Muito parecido'
  if (score >= threshold) return 'Parecido'
  return 'Atenção: pode ter diferenças'
}
