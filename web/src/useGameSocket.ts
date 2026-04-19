import { useCallback, useEffect, useRef, useState } from 'react'
import type { ActionData, GameStateData, ServerMessage } from './types'

export function useGameSocket() {
  const wsRef = useRef<WebSocket | null>(null)
  const [connected, setConnected] = useState(false)
  const [gameState, setGameState] = useState<GameStateData | null>(null)
  const [legalActions, setLegalActions] = useState<ActionData[]>([])
  const [aiActions, setAiActions] = useState<Array<{ player: number; action: ActionData }>>([])
  const [log, setLog] = useState<string[]>([])

  const connect = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/game`)

    ws.onopen = () => setConnected(true)
    ws.onclose = () => {
      setConnected(false)
      // Retry after 2s
      setTimeout(connect, 2000)
    }
    ws.onmessage = (event) => {
      const msg: ServerMessage = JSON.parse(event.data)
      if (msg.type === 'game_state') {
        if (msg.state) setGameState(msg.state)
        if (msg.legalActions) setLegalActions(msg.legalActions)
        if (msg.aiActions) {
          setAiActions(msg.aiActions)
          // Add to log
          const entries = msg.aiActions.map(
            (a) => `Player ${a.player} → ${formatAction(a.action)}`
          )
          setLog((prev) => [...prev.slice(-100), ...entries])
        }
      }
    }
    wsRef.current = ws
  }, [])

  useEffect(() => {
    connect()
    return () => wsRef.current?.close()
  }, [connect])

  const send = useCallback((data: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data))
    }
  }, [])

  const newGame = useCallback((difficulty: string) => {
    setLog([])
    send({ type: 'new_game', difficulty, humanPlayer: 0 })
  }, [send])

  const performAction = useCallback((action: ActionData) => {
    setLog((prev) => [...prev.slice(-100), `You → ${formatAction(action)}`])
    send({ type: 'action', action })
  }, [send])

  return { connected, gameState, legalActions, aiActions, log, newGame, performAction }
}

function formatAction(a: ActionData): string {
  const parts = [a.type.replace(/_/g, ' ').toLowerCase()]
  if (a.vertex !== undefined) parts.push(`v${a.vertex}`)
  if (a.edge !== undefined) parts.push(`e${a.edge}`)
  if (a.hexId !== undefined) parts.push(`hex${a.hexId}`)
  if (a.resource) parts.push(a.resource)
  if (a.giveResource && a.getResource) parts.push(`${a.giveResource}→${a.getResource}`)
  return parts.join(' ')
}
