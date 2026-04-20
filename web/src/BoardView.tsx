import React, { useMemo } from 'react'
import type { BoardData, ActionData } from './types'

const TERRAIN_COLORS: Record<string, string> = {
  hills: '#c25b3a',
  forest: '#2d6a2e',
  mountains: '#6b6b6b',
  fields: '#d4a017',
  pasture: '#7ec850',
  desert: '#d4c692',
}

const PLAYER_COLORS = ['#e74c3c', '#3498db', '#ecf0f1', '#f39c12']

const HEX_SIZE = 52
const SQRT3 = Math.sqrt(3)

interface BoardViewProps {
  board: BoardData
  legalActions: ActionData[]
  onAction: (action: ActionData) => void
  robberHex: number
  currentPhase: string
}

export default function BoardView({
  board,
  legalActions,
  onAction,
  robberHex,
  currentPhase,
}: BoardViewProps) {
  // Compute pixel positions
  const hexPixels = useMemo(() => {
    return board.hexes.map((h) => ({
      ...h,
      px: h.cx * HEX_SIZE * 1.05,
      py: h.cy * HEX_SIZE * 1.05,
    }))
  }, [board.hexes])

  const vertexPixels = useMemo(() => {
    return board.vertices.map((v) => ({
      ...v,
      px: v.x * HEX_SIZE * 1.05,
      py: v.y * HEX_SIZE * 1.05,
    }))
  }, [board.vertices])

  const edgePixels = useMemo(() => {
    return board.edges.map((e) => ({
      ...e,
      px1: e.x1 * HEX_SIZE * 1.05,
      py1: e.y1 * HEX_SIZE * 1.05,
      px2: e.x2 * HEX_SIZE * 1.05,
      py2: e.y2 * HEX_SIZE * 1.05,
    }))
  }, [board.edges])

  // Build sets of clickable items
  const clickableVertices = useMemo(() => {
    const set = new Set<number>()
    for (const a of legalActions) {
      if (a.vertex !== undefined && (
        a.type === 'PLACE_SETUP_SETTLEMENT' ||
        a.type === 'BUILD_SETTLEMENT' ||
        a.type === 'BUILD_CITY'
      )) {
        set.add(a.vertex)
      }
    }
    return set
  }, [legalActions])

  const clickableEdges = useMemo(() => {
    const set = new Set<number>()
    for (const a of legalActions) {
      if (a.edge !== undefined && (
        a.type === 'PLACE_SETUP_ROAD' ||
        a.type === 'BUILD_ROAD' ||
        a.type === 'PLACE_ROAD_BUILDING_ROAD'
      )) {
        set.add(a.edge)
      }
    }
    return set
  }, [legalActions])

  const clickableHexes = useMemo(() => {
    const set = new Set<number>()
    for (const a of legalActions) {
      if (a.hexId !== undefined && (
        a.type === 'PLACE_ROBBER' ||
        a.type === 'PLAY_KNIGHT'
      )) {
        set.add(a.hexId)
      }
    }
    return set
  }, [legalActions])

  const handleHexClick = (hexId: number) => {
    const action = legalActions.find(
      (a) => a.hexId === hexId && (a.type === 'PLACE_ROBBER' || a.type === 'PLAY_KNIGHT')
    )
    if (action) onAction(action)
  }

  const handleVertexClick = (vid: number) => {
    const action = legalActions.find(
      (a) => a.vertex === vid && (
        a.type === 'PLACE_SETUP_SETTLEMENT' ||
        a.type === 'BUILD_SETTLEMENT' ||
        a.type === 'BUILD_CITY'
      )
    )
    if (action) onAction(action)
  }

  const handleEdgeClick = (eid: number) => {
    const action = legalActions.find(
      (a) => a.edge === eid && (
        a.type === 'PLACE_SETUP_ROAD' ||
        a.type === 'BUILD_ROAD' ||
        a.type === 'PLACE_ROAD_BUILDING_ROAD'
      )
    )
    if (action) onAction(action)
  }

  // Generate hex polygon points
  const hexPoints = (size: number) => {
    const pts: string[] = []
    for (let i = 0; i < 6; i++) {
      const angle = (Math.PI / 180) * (60 * i)
      pts.push(`${size * Math.cos(angle)},${size * Math.sin(angle)}`)
    }
    return pts.join(' ')
  }

  // Number dot colors (6 and 8 are red)
  const numberColor = (n: number) => (n === 6 || n === 8 ? '#e74c3c' : '#fff')
  const numberDots = (n: number) => {
    // Probability dots: 2->1, 3->2, 4->3, 5->4, 6->5, 8->5, 9->4, 10->3, 11->2, 12->1
    const dots = n <= 7 ? n - 1 : 13 - n
    return '•'.repeat(dots)
  }

  const viewBox = '-350 -300 700 600'

  return (
    <svg viewBox={viewBox} width="100%" height="100%" style={{ maxWidth: 700, maxHeight: 600 }}>
      {/* Hex tiles */}
      {hexPixels.map((h) => (
        <g key={`hex-${h.id}`} transform={`translate(${h.px}, ${h.py})`}>
          <polygon
            points={hexPoints(HEX_SIZE * 0.95)}
            fill={TERRAIN_COLORS[h.terrain] || '#555'}
            stroke={clickableHexes.has(h.id) ? '#fff' : '#1a1a2e'}
            strokeWidth={clickableHexes.has(h.id) ? 3 : 2}
            opacity={clickableHexes.has(h.id) ? 1 : 0.9}
            style={{ cursor: clickableHexes.has(h.id) ? 'pointer' : 'default' }}
            onClick={() => handleHexClick(h.id)}
          />
          {h.number > 0 && (
            <>
              <circle r={16} fill="rgba(0,0,0,0.7)" />
              <text
                textAnchor="middle"
                dominantBaseline="central"
                dy={-2}
                fill={numberColor(h.number)}
                fontSize={16}
                fontWeight="bold"
              >
                {h.number}
              </text>
              <text
                textAnchor="middle"
                dominantBaseline="central"
                dy={10}
                fill={numberColor(h.number)}
                fontSize={6}
              >
                {numberDots(h.number)}
              </text>
            </>
          )}
          {h.hasRobber && (
            <text textAnchor="middle" dominantBaseline="central" fontSize={28} dy={h.number > 0 ? -24 : 0}>
              🥷
            </text>
          )}
        </g>
      ))}

      {/* Edges / roads */}
      {edgePixels.map((e) => (
        <line
          key={`edge-${e.id}`}
          x1={e.px1}
          y1={e.py1}
          x2={e.px2}
          y2={e.py2}
          stroke={
            e.road !== null
              ? PLAYER_COLORS[e.road]
              : clickableEdges.has(e.id)
              ? 'rgba(255,255,255,0.5)'
              : 'transparent'
          }
          strokeWidth={e.road !== null ? 5 : clickableEdges.has(e.id) ? 4 : 1}
          strokeLinecap="round"
          style={{ cursor: clickableEdges.has(e.id) ? 'pointer' : 'default' }}
          onClick={() => handleEdgeClick(e.id)}
        />
      ))}

      {/* Vertices / buildings */}
      {vertexPixels.map((v) => {
        const isClickable = clickableVertices.has(v.id)
        const hasBuilding = v.building !== null

        if (!hasBuilding && !isClickable) {
          // Show harbor indicator
          if (v.harbor) {
            const label = v.harbor === 'generic' ? '3:1' : '2:1'
            const color = v.harbor === 'generic' ? '#4ecdc4' : '#f39c12'
            return (
              <g key={`v-${v.id}`} transform={`translate(${v.px}, ${v.py})`}>
                <circle r={10} fill={color} opacity={0.8} stroke="#000" strokeWidth={0.5} />
                <text
                  textAnchor="middle"
                  dominantBaseline="central"
                  fontSize={7}
                  fontWeight="bold"
                  fill="#000"
                >{label}</text>
              </g>
            )
          }
          return null
        }

        return (
          <g
            key={`v-${v.id}`}
            transform={`translate(${v.px}, ${v.py})`}
            style={{ cursor: isClickable ? 'pointer' : 'default' }}
            onClick={() => handleVertexClick(v.id)}
          >
            {v.building === 'settlement' && (
              <polygon
                points="-7,5 7,5 7,-2 0,-8 -7,-2"
                fill={v.owner !== null ? PLAYER_COLORS[v.owner] : '#fff'}
                stroke="#000"
                strokeWidth={1.5}
              />
            )}
            {v.building === 'city' && (
              <polygon
                points="-10,6 10,6 10,-2 4,-2 4,-8 -4,-8 -4,-2 -10,-2"
                fill={v.owner !== null ? PLAYER_COLORS[v.owner] : '#fff'}
                stroke="#000"
                strokeWidth={1.5}
              />
            )}
            {isClickable && !hasBuilding && (
              <circle
                r={7}
                fill="rgba(255,255,255,0.3)"
                stroke="#fff"
                strokeWidth={2}
                strokeDasharray="3,3"
              />
            )}
          </g>
        )
      })}
    </svg>
  )
}
