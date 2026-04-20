import { useCallback, useEffect, useRef, useState } from 'react'
import BoardView from './BoardView'
import type { GameStateData, ActionData, PlayerData } from './types'

const RESOURCE_EMOJI: Record<string, string> = {
  brick: '🧱',
  lumber: '🪵',
  ore: '⛏️',
  grain: '🌾',
  wool: '🐑',
}

interface EpochSummary {
  epoch: number
  totalEpochs: number
  gamesPlayed: number
  avgGameLength: number
  policyLoss: number
  valueLoss: number
  entropy: number
  wins: Record<number, number>
}

export default function SpectatorView({ onBack }: { onBack: () => void }) {
  const wsRef = useRef<WebSocket | null>(null)
  const [connected, setConnected] = useState(false)
  const [gameState, setGameState] = useState<GameStateData | null>(null)
  const [lastAction, setLastAction] = useState<ActionData | null>(null)
  const [epoch, setEpoch] = useState(0)
  const [totalEpochs, setTotalEpochs] = useState(0)
  const [gameIdx, setGameIdx] = useState(0)
  const [trainingRunning, setTrainingRunning] = useState(false)
  const [epochLog, setEpochLog] = useState<EpochSummary[]>([])
  const [speed, setSpeed] = useState<'all' | 'turns' | 'slow'>('turns')
  const [paused, setPaused] = useState(false)

  // Throttle state updates based on speed setting
  const lastUpdateRef = useRef(0)
  const pausedRef = useRef(paused)
  pausedRef.current = paused
  const speedRef = useRef(speed)
  speedRef.current = speed

  const connect = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/spectate`)

    ws.onopen = () => setConnected(true)
    ws.onclose = () => {
      setConnected(false)
      setTimeout(connect, 2000)
    }
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data)

      if (msg.type === 'training_status') {
        setTrainingRunning(msg.running)
        setEpoch(msg.epoch || 0)
        setTotalEpochs(msg.total_epochs || 0)
      } else if (msg.type === 'spectate_state') {
        if (pausedRef.current) return

        const now = Date.now()
        const minInterval = speedRef.current === 'slow' ? 300 : speedRef.current === 'turns' ? 50 : 0

        if (now - lastUpdateRef.current >= minInterval) {
          setGameState(msg.state)
          setLastAction(msg.lastAction)
          setEpoch(msg.epoch)
          setGameIdx(msg.gameIdx)
          lastUpdateRef.current = now
        }
      } else if (msg.type === 'epoch_summary') {
        setEpochLog((prev) => [...prev.slice(-50), msg])
        setEpoch(msg.epoch)
        setTotalEpochs(msg.totalEpochs)
      }
    }
    wsRef.current = ws
  }, [])

  useEffect(() => {
    connect()
    return () => wsRef.current?.close()
  }, [connect])

  const send = useCallback(
    (data: Record<string, unknown>) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify(data))
      }
    },
    [],
  )

  const startTraining = (epochs: number) => {
    send({ type: 'start_training', epochs, gamesPerEpoch: 20 })
  }

  const noOp = () => {}

  // ---- Not connected or not training ----
  if (!connected) {
    return (
      <div className="start-screen">
        <h1>Live Spectator</h1>
        <p style={{ color: 'var(--text-dim)' }}>Connecting to server...</p>
        <button className="action-btn" onClick={onBack}>
          ← Back
        </button>
      </div>
    )
  }

  if (!trainingRunning && !gameState) {
    return (
      <div className="start-screen">
        <h1>Live Spectator</h1>
        <p style={{ color: 'var(--text-dim)', marginBottom: 24 }}>
          Launch training and watch AI agents play in real-time
        </p>
        <div className="difficulty-select">
          {[
            { epochs: 20, label: 'Quick', desc: '20 epochs' },
            { epochs: 50, label: 'Short', desc: '50 epochs' },
            { epochs: 200, label: 'Full', desc: '200 epochs' },
          ].map((opt) => (
            <button
              key={opt.epochs}
              className="difficulty-btn"
              onClick={() => startTraining(opt.epochs)}
            >
              <div className="label">{opt.label}</div>
              <div className="desc">{opt.desc}</div>
            </button>
          ))}
        </div>
        <button
          className="action-btn"
          onClick={onBack}
          style={{ marginTop: 24 }}
        >
          ← Back
        </button>
      </div>
    )
  }

  const latestEpoch = epochLog.length > 0 ? epochLog[epochLog.length - 1] : null

  return (
    <div className="app">
      {/* Sidebar */}
      <div className="sidebar">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h1 style={{ fontSize: 18 }}>🔴 Live Training</h1>
          <button className="action-btn" onClick={onBack} style={{ fontSize: 12, padding: '4px 10px' }}>
            ← Back
          </button>
        </div>

        {/* Training progress */}
        <div className="game-info" style={{ flexDirection: 'column', gap: 4 }}>
          <div>
            Epoch <strong>{epoch}</strong> / {totalEpochs}
            {' · '}Game {gameIdx + 1}
          </div>
          <div
            style={{
              width: '100%',
              height: 6,
              background: 'var(--surface)',
              borderRadius: 3,
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                width: `${totalEpochs > 0 ? (epoch / totalEpochs) * 100 : 0}%`,
                height: '100%',
                background: 'var(--accent)',
                transition: 'width 0.3s',
              }}
            />
          </div>
        </div>

        {/* Speed controls */}
        <section>
          <h3>Playback</h3>
          <div className="actions" style={{ gap: 6 }}>
            <button
              className={`action-btn ${speed === 'all' ? 'primary' : ''}`}
              onClick={() => setSpeed('all')}
            >
              Max
            </button>
            <button
              className={`action-btn ${speed === 'turns' ? 'primary' : ''}`}
              onClick={() => setSpeed('turns')}
            >
              Normal
            </button>
            <button
              className={`action-btn ${speed === 'slow' ? 'primary' : ''}`}
              onClick={() => setSpeed('slow')}
            >
              Slow
            </button>
            <button
              className={`action-btn ${paused ? 'primary' : ''}`}
              onClick={() => setPaused(!paused)}
            >
              {paused ? '▶ Resume' : '⏸ Pause'}
            </button>
          </div>
        </section>

        {/* Current game info */}
        {gameState && (
          <>
            <div className="game-info">
              <span className="phase-indicator">{gameState.phase.replace(/_/g, ' ')}</span>
              {' '}Turn {gameState.turnNumber}
              {gameState.diceRoll && (
                <span className="dice-display" style={{ marginLeft: 12 }}>
                  ⚄ {gameState.diceRoll[0]} + {gameState.diceRoll[1]} ={' '}
                  {gameState.diceRoll[0] + gameState.diceRoll[1]}
                </span>
              )}
            </div>

            {/* Players */}
            <section>
              <h3>Players</h3>
              {gameState.players.map((p) => (
                <SpectatorPlayerCard
                  key={p.index}
                  player={p}
                  isActive={p.index === gameState.currentPlayer}
                />
              ))}
            </section>

            {/* Last action */}
            {lastAction && (
              <section>
                <h3>Last Action</h3>
                <div className="log-entry" style={{ fontSize: 13 }}>
                  P{gameState.currentPlayer}:{' '}
                  {lastAction.type.replace(/_/g, ' ').toLowerCase()}
                  {lastAction.vertex !== undefined && ` v${lastAction.vertex}`}
                  {lastAction.edge !== undefined && ` e${lastAction.edge}`}
                  {lastAction.resource && ` ${lastAction.resource}`}
                </div>
              </section>
            )}

            {/* Winner */}
            {gameState.winner !== null && (
              <div
                style={{
                  padding: 12,
                  background: 'var(--accent)',
                  borderRadius: 8,
                  textAlign: 'center',
                  color: '#fff',
                  fontWeight: 'bold',
                }}
              >
                🏆 Player {gameState.winner} wins with{' '}
                {gameState.players[gameState.winner].victoryPoints} VP!
              </div>
            )}
          </>
        )}

        {/* Epoch log */}
        {epochLog.length > 0 && (
          <section>
            <h3>Epoch Log</h3>
            <div className="log" style={{ maxHeight: 200 }}>
              {epochLog
                .slice()
                .reverse()
                .map((e, i) => (
                  <div key={i} className="log-entry" style={{ fontSize: 12 }}>
                    <strong>E{e.epoch}</strong> — Avg len: {e.avgGameLength} · P_loss:{' '}
                    {e.policyLoss} · Wins: [
                    {Object.entries(e.wins)
                      .map(([p, w]) => `P${p}:${w}`)
                      .join(' ')}
                    ]
                  </div>
                ))}
            </div>
          </section>
        )}
      </div>

      {/* Board */}
      <div className="board-area">
        {gameState ? (
          <BoardView
            board={gameState.board}
            legalActions={[]}
            onAction={noOp}
            robberHex={gameState.robberHex}
            currentPhase={gameState.phase}
          />
        ) : (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              color: 'var(--text-dim)',
            }}
          >
            Waiting for first game...
          </div>
        )}
      </div>
    </div>
  )
}

function SpectatorPlayerCard({
  player,
  isActive,
}: {
  player: PlayerData
  isActive: boolean
}) {
  return (
    <div
      className={`player-card ${isActive ? 'active' : ''}`}
      style={{ borderLeftColor: player.color }}
    >
      <div className="name">
        🤖 Agent {player.index}
        <span className="vp">{player.victoryPoints} VP</span>
      </div>
      <div className="stats">
        <span>🏠{player.numSettlements}</span>
        <span>🏙️{player.numCities}</span>
        <span>🛤️{player.numRoads}</span>
        <span>🃏{player.numDevCards}</span>
        <span>⚔️{player.knightsPlayed}</span>
        {player.hasLongestRoad && <span>🛣️LR</span>}
        {player.hasLargestArmy && <span>⚔️LA</span>}
      </div>
      {player.resources && (
        <div className="stats" style={{ fontSize: 11, marginTop: 2 }}>
          {Object.entries(player.resources).map(([res, count]) => (
            <span key={res}>
              {RESOURCE_EMOJI[res]}{count}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
