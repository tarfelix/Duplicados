import { useRef } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'
import type { Group } from '@/types'
import GroupCard from './GroupCard'

interface Props {
  groups: Group[]
  minSim: number
}

export default function GroupList({ groups, minSim }: Props) {
  const parentRef = useRef<HTMLDivElement>(null)

  const virtualizer = useVirtualizer({
    count: groups.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 200,
    overscan: 5,
  })

  return (
    <div ref={parentRef} className="h-[calc(100vh-320px)] overflow-auto">
      <div style={{ height: `${virtualizer.getTotalSize()}px`, width: '100%', position: 'relative' }}>
        {virtualizer.getVirtualItems().map((virtualRow) => {
          const group = groups[virtualRow.index]
          return (
            <div
              key={group.group_id}
              data-index={virtualRow.index}
              ref={virtualizer.measureElement}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                transform: `translateY(${virtualRow.start}px)`,
              }}
            >
              <GroupCard group={group} minSim={minSim} />
            </div>
          )
        })}
      </div>
    </div>
  )
}
