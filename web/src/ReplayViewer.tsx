import { useCallback, useEffect, useRef, useState } from 'react'
import BoardView from './BoardView'
import type { GameStateData, PlayerData } from './types'

const RESOURCE_EMOJI: Record<string, string> = {
  brick: '🧱',
  lumber: '🪵',
  ore: '⛏️',
  grain: '🌾',
  wool: '🐑',
}

interface ReplaySummary {
  filename: string
  seed: number
  winner: number | null
  numTurns: number
  finalVPs: number[]
  epoch: number
  gameIdx: number
  numFrames: number
}

interface ReplayFrame {
  player: number
  actionType: string
  phase?: string
  turnNumber?: number
  vps?: number[]
  vertex?: number
  edge?: number
  hexId?: number
  targetPlayer?: number
  resource?: string
  resource2?: string
  giveResource?: string
  getResource?: string
  diceRoll?: number[]
  discard?: Record<string, number>
}

interface ReplayDetail {
  seed: number
  winner: number | null
  numTurns: number
  finalVPs: number[]
  epoch: number
  gameIdx: number
  frames: ReplayFrame[]
}

const PLAYER_COLORS = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71']

export default function ReplayViewer({ onBack }: { onBack: () => void }) {
  const [replays, setReplays] = useState<ReplaySummary[]>([])
  const [loading, setLoading] = useState(true)
  const [replay, setReplay] = useState<ReplayDetail | null>(null)
  const [currentFile, setCurrentFile] = useState<string | null>(null)
  const [frameIdx, setFrameIdx] = useState(0)
  const [gameState, setGameState] = useState<GameStateData | null>(null)
  const [loadingState, setLoadingState] = useState(false)
  const [playing, setPlaying] = useState(false)
  const [playSpeed, setPlaySpeed] = useState(200) // ms per frame
  const playRef = useRef(false)

  // Fetch replay list
  useEffect(() => {
    fetch('/api/replays')
      .then((r) => r.json())
      .then((data) => {
        setReplays(data.replays || [])
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  // Load a replay
  const loadReplay = useCallback(async (filename: string) => {
    setCurrentFile(filename)
    setFrameIdx(0)
    setGameState(null)
    const res = await fetch(`/api/replays/${filename}`)
    const data = await res.json()
    setReplay(data)
    // Load initial state (frame 0 = before any actions)
    fetchState(filename, 0)
  }, [])

  // Fetch game state at a specific frame
  const fetchState = async (filename: string, frame: number) => {
    setLoadingState(true)
    const res = await fetch(`/api/replays/${filename}/state/${frame}`)
    const state = await res.json()
    setGameState(state)
    setLoadingState(false)
  }

  // Navigate to a frame
  const goToFrame = useCallback(
    (idx: number) => {
      if (!currentFile || !replay) return
      const clamped = Math.max(0, Math.min(idx, replay.frames.length))
      setFrameIdx(clamped)
      fetchState(currentFile, clamped)
    },
    [currentFile, replay],
  )

  // Auto-play
  useEffect(() => {
    playRef.current = playing
    if (!playing || !replay || !currentFile) return

    let timeoutId: ReturnType<typeof setTimeout>
    const step = () => {
      if (!playRef.current) return
      setFrameIdx((prev) => {
        const next = prev + 1
        if (next > replay.frames.length) {
          setPlaying(false)
          return prev
        }
        fetchState(currentFile, next)
        return next
      })
      timeoutId = setTimeout(step, playSpeed)
    }
    timeoutId = setTimeout(step, playSpeed)
    return () => clearTimeout(timeoutId)
  }, [playing, replay, currentFile, playSpeed])

  const noOp = () => {}

  // ---- Replay list ----
  if (!replay) {
    return (
      <div className="start-screen">
        <h1>📼 Game Replays</h1>
        <p style={{ color: 'var(--text-dim)', marginBottom: 24 }}>
          Step through recorded games turn-by-turn
        </p>
        {loading ? (
          <p style={{ color: 'var(--text-dim)' }}>Loading...</p>
        ) : replays.length === 0 ? (
          <div>
            <p style={{ color: 'var(--text-dim)', marginBottom: 16 }}>
              No replays yet. Train the AI to generate won games.
            </p>
          </div>
        ) : (
          <div style={{ maxWidth: 600, width: '100%' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
              <thead>
                <tr style={{ color: 'var(--text-dim)', textAlign: 'left' }}>
                  <th style={{ padding: '8px 12px' }}>Epoch</th>
                  <th style={{ padding: '8px 12px' }}>Winner</th>
                  <th style={{ padding: '8px 12px' }}>Final VPs</th>
                  <th style={{ padding: '8px 12px' }}>Turns</th>
                  <th style={{ padding: '8px 12px' }}>Actions</th>
                  <th style={{ padding: '8px 12px' }}></th>
                </tr>
              </thead>
              <tbody>
                {replays.map((r) => (
                  <tr
                    key={r.filename}
                    style={{
                      borderTop: '1px solid var(--surface)',
                      cursor: 'pointer',
                    }}
                    onClick={() => loadReplay(r.filename)}
                  >
                    <td style={{ padding: '8px 12px' }}>E{r.epoch}</td>
                    <td style={{ padding: '8px 12px' }}>
                      <span style={{ color: r.winner !== null ? PLAYER_COLORS[r.winner] : 'var(--text-dim)' }}>
                        {r.winner !== null ? `Player ${r.winner}` : 'Draw'}
                      </span>
                    </td>
                    <td style={{ padding: '8px 12px' }}>
                      {r.finalVPs.map((vp, i) => (
                        <span
                          key={i}
                          style={{
                            color: PLAYER_COLORS[i],
                            marginRight: 6,
                            fontWeight: i === r.winner ? 'bold' : 'normal',
                          }}
                        >
                          {vp}
                        </span>
                      ))}
                    </td>
                    <td style={{ padding: '8px 12px' }}>{r.numTurns}</td>
                    <td style={{ padding: '8px 12px' }}>{r.numFrames}</td>
                    <td style={{ padding: '8px 12px' }}>
                      <button className="action-btn" style={{ fontSize: 12, padding: '4px 10px' }}>
                        View
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <button className="action-btn" onClick={onBack} style={{ marginTop: 24 }}>
          ← Back
        </button>
      </div>
    )
  }

  // ---- Replay viewer ----
  const currentFrame = frameIdx > 0 ? replay.frames[frameIdx - 1] : null
  const totalFrames = replay.frames.length

  return (
    <div className="app">
      {/* Sidebar */}
      <div className="sidebar">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h1 style={{ fontSize: 18 }}>📼 Replay</h1>
          <button
            className="action-btn"
            onClick={() => {
              setReplay(null)
              setCurrentFile(null)
              setGameState(null)
              setPlaying(false)
            }}
            style={{ fontSize: 12, padding: '4px 10px' }}
          >
            ← List
          </button>
        </div>

        {/* Replay info */}
        <div className="game-info" style={{ flexDirection: 'column', gap: 4 }}>
          <div>
            Epoch <strong>{replay.epoch}</strong> · Game {replay.gameIdx}
            {replay.winner !== null && (
              <span style={{ color: PLAYER_COLORS[replay.winner], marginLeft: 8 }}>
                Winner: P{replay.winner}
              </span>
            )}
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-dim)' }}>
            Seed: {replay.seed} · {replay.numTurns} turns · {totalFrames} actions
          </div>
        </div>

        {/* Frame slider */}
        <section>
          <h3>
            Frame {frameIdx} / {totalFrames}
            {currentFrame && (
              <span style={{ fontWeight: 'normal', fontSize: 12, marginLeft: 8, color: 'var(--text-dim)' }}>
                Turn {currentFrame.turnNumber}
              </span>
            )}
          </h3>
          <input
            type="range"
            min={0}
            max={totalFrames}
            value={frameIdx}
            onChange={(e) => goToFrame(parseInt(e.target.value))}
            style={{ width: '100%' }}
          />
        </section>

        {/* Playback controls */}
        <section>
          <h3>Playback</h3>
          <div className="actions" style={{ gap: 6 }}>
            <button className="action-btn" onClick={() => goToFrame(0)} title="Start">
              ⏮
            </button>
            <button className="action-btn" onClick={() => goToFrame(frameIdx - 1)} title="Previous">
              ◀
            </button>
            <button
              className={`action-btn ${playing ? 'primary' : ''}`}
              onClick={() => setPlaying(!playing)}
            >
              {playing ? '⏸' : '▶'}
            </button>
            <button className="action-btn" onClick={() => goToFrame(frameIdx + 1)} title="Next">
              ▶
            </button>
            <button className="action-btn" onClick={() => goToFrame(totalFrames)} title="End">
              ⏭
            </button>
          </div>
          <div className="actions" style={{ gap: 6, marginTop: 6 }}>
            {[
              { ms: 500, label: '0.5×' },
              { ms: 200, label: '1×' },
              { ms: 50, label: '4×' },
              { ms: 10, label: '20×' },
            ].map((s) => (
              <button
                key={s.ms}
                className={`action-btn ${playSpeed === s.ms ? 'primary' : ''}`}
                onClick={() => setPlaySpeed(s.ms)}
                style={{ fontSize: 12 }}
              >
                {s.label}
              </button>
            ))}
          </div>
        </section>

        {/* Current action */}
        {currentFrame && (
          <section>
            <h3>Action</h3>
            <div
              className="log-entry"
              style={{
                fontSize: 13,
                padding: 8,
                background: 'var(--surface)',
                borderRadius: 6,
                borderLeft: `3px solid ${PLAYER_COLORS[currentFrame.player]}`,
              }}
            >
              <strong style={{ color: PLAYER_COLORS[currentFrame.player] }}>
                P{currentFrame.player}
              </strong>
              {': '}
              {formatAction(currentFrame)}
            </div>
          </section>
        )}

        {/* Players */}
        {gameState && (
          <section>
            <h3>Players</h3>
            {gameState.players.map((p) => (
              <ReplayPlayerCard
                key={p.index}
                player={p}
                isActive={currentFrame ? p.index === currentFrame.player : false}
              />
            ))}
          </section>
        )}

        {/* VP chart */}
        {currentFrame && currentFrame.vps && (
          <section>
            <h3>Victory Points</h3>
            <div style={{ display: 'flex', gap: 12, alignItems: 'flex-end', height: 60 }}>
              {currentFrame.vps.map((vp, i) => (
                <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flex: 1 }}>
                  <div style={{ fontSize: 12, fontWeight: 'bold', marginBottom: 4 }}>{vp}</div>
                  <div
                    style={{
                      width: '100%',
                      height: Math.max(4, (vp / 10) * 50),
                      background: PLAYER_COLORS[i],
                      borderRadius: 3,
                      transition: 'height 0.2s',
                    }}
                  />
                  <div style={{ fontSize: 10, color: 'var(--text-dim)', marginTop: 4 }}>P{i}</div>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Final result */}
        {frameIdx === totalFrames && replay.winner !== null && (
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
            🏆 Player {replay.winner} wins with {replay.finalVPs[replay.winner]} VP!
          </div>
        )}

        {/* Action log — last 20 frames around current */}
        <section>
          <h3>Action Log</h3>
          <div className="log" style={{ maxHeight: 200 }}>
            {replay.frames
              .slice(Math.max(0, frameIdx - 10), frameIdx)
              .map((f, i) => {
                const absIdx = Math.max(0, frameIdx - 10) + i
                return (
                  <div
                    key={absIdx}
                    className="log-entry"
                    style={{
                      fontSize: 12,
                      cursor: 'pointer',
                      opacity: absIdx === frameIdx - 1 ? 1 : 0.6,
                      borderLeft: `2px solid ${PLAYER_COLORS[f.player]}`,
                      paddingLeft: 6,
                    }}
                    onClick={() => goToFrame(absIdx + 1)}
                  >
                    <span style={{ color: 'var(--text-dim)' }}>#{absIdx + 1}</span>{' '}
                    <strong style={{ color: PLAYER_COLORS[f.player] }}>P{f.player}</strong>{' '}
                    {formatAction(f)}
                  </div>
                )
              })}
          </div>
        </section>
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
            {loadingState ? 'Loading state...' : 'Select a frame to view'}
          </div>
        )}
      </div>
    </div>
  )
}

function ReplayPlayerCard({ player, isActive }: { player: PlayerData; isActive: boolean }) {
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
        <div className="stats" style={{ fontSize: 11 }}>
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

function formatAction(frame: ReplayFrame): string {
  const type = frame.actionType.replace(/_/g, ' ').toLowerCase()
  const parts = [type]
  if (frame.vertex !== undefined) parts.push(`v${frame.vertex}`)
  if (frame.edge !== undefined) parts.push(`e${frame.edge}`)
  if (frame.hexId !== undefined) parts.push(`hex ${frame.hexId}`)
  if (frame.resource) parts.push(frame.resource)
  if (frame.resource2) parts.push(`+ ${frame.resource2}`)
  if (frame.giveResource && frame.getResource) {
    parts.push(`${frame.giveResource} → ${frame.getResource}`)
  }
  if (frame.targetPlayer !== undefined) parts.push(`→ P${frame.targetPlayer}`)
  if (frame.diceRoll) parts.push(`🎲 ${frame.diceRoll[0]}+${frame.diceRoll[1]}`)
  return parts.join(' ')
}
