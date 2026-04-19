import { useState } from 'react'
import BoardView from './BoardView'
import TrainingDashboard from './TrainingDashboard'
import { useGameSocket } from './useGameSocket'
import type { ActionData, PlayerData } from './types'

const RESOURCE_EMOJI: Record<string, string> = {
  brick: '🧱',
  lumber: '🪵',
  ore: '⛏️',
  grain: '🌾',
  wool: '🐑',
}

const RESOURCE_LABELS: Record<string, string> = {
  brick: 'Brick',
  lumber: 'Lumber',
  ore: 'Ore',
  grain: 'Grain',
  wool: 'Wool',
}

type Tab = 'play' | 'training'

export default function App() {
  const { connected, gameState, legalActions, log, newGame, performAction } = useGameSocket()
  const [tab, setTab] = useState<Tab>('play')
  const [selectedAction, setSelectedAction] = useState<string | null>(null)

  // ---- Start screen ----
  if (!gameState && tab === 'play') {
    return (
      <div className="start-screen">
        <h1>Catan Sim</h1>
        <p>AI-powered Settlers of Catan</p>
        <div className="difficulty-select">
          {[
            { key: 'easy', label: 'Easy', desc: 'Early training checkpoint' },
            { key: 'medium', label: 'Medium', desc: 'Mid training checkpoint' },
            { key: 'hard', label: 'Hard', desc: 'Fully trained agent' },
            { key: 'random', label: 'Random', desc: 'Random bot opponents' },
          ].map((d) => (
            <button
              key={d.key}
              className="difficulty-btn"
              onClick={() => newGame(d.key)}
              disabled={!connected}
            >
              <div className="label">{d.label}</div>
              <div className="desc">{d.desc}</div>
            </button>
          ))}
        </div>
        <div style={{ marginTop: 20 }}>
          <button
            className="difficulty-btn"
            onClick={() => setTab('training')}
            style={{ fontSize: 14, padding: '12px 24px' }}
          >
            <div className="label">📊 Training Dashboard</div>
            <div className="desc">View AI training progress</div>
          </button>
        </div>
        {!connected && (
          <p style={{ color: 'var(--accent)', marginTop: 16 }}>
            Connecting to server...
          </p>
        )}
      </div>
    )
  }

  if (tab === 'training') {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
        <div className="tab-bar">
          <button onClick={() => setTab('play')} className={tab === 'play' ? 'active' : ''}>
            Play
          </button>
          <button onClick={() => setTab('training')} className={tab === 'training' ? 'active' : ''}>
            Training
          </button>
        </div>
        <TrainingDashboard />
      </div>
    )
  }

  if (!gameState) return null

  const myPlayer = gameState.players[0] // human is always player 0
  const isMyTurn = gameState.currentPlayer === 0
  const phase = gameState.phase

  // Group actions by type for the action panel
  const actionGroups = groupActions(legalActions)

  // Filter actions to show based on selected action type
  const boardActions = legalActions.filter((a) =>
    ['PLACE_SETUP_SETTLEMENT', 'PLACE_SETUP_ROAD', 'BUILD_SETTLEMENT',
     'BUILD_CITY', 'BUILD_ROAD', 'PLACE_ROBBER', 'PLAY_KNIGHT',
     'PLACE_ROAD_BUILDING_ROAD'].includes(a.type)
  )

  return (
    <div className="app">
      {/* Sidebar */}
      <div className="sidebar">
        <h1>Catan Sim</h1>

        {/* Game info */}
        <div className="game-info">
          <span className="phase-indicator">{formatPhase(phase)}</span>
          {' '}Turn {gameState.turnNumber}
          {gameState.diceRoll && (
            <span className="dice-display" style={{ marginLeft: 12 }}>
              ⚄ {gameState.diceRoll[0]} + {gameState.diceRoll[1]} = {gameState.diceRoll[0] + gameState.diceRoll[1]}
            </span>
          )}
        </div>

        {/* Players */}
        <section>
          <h3>Players</h3>
          {gameState.players.map((p) => (
            <PlayerCard
              key={p.index}
              player={p}
              isActive={p.index === gameState.currentPlayer}
              isHuman={p.index === 0}
            />
          ))}
        </section>

        {/* Resources */}
        <section>
          <h3>Your Resources</h3>
          <div className="resources">
            {myPlayer.resources &&
              Object.entries(myPlayer.resources).map(([res, count]) => (
                <div key={res} className="resource-item">
                  <div style={{ fontSize: 20 }}>{RESOURCE_EMOJI[res]}</div>
                  <div className="count">{count}</div>
                  <div>{RESOURCE_LABELS[res]}</div>
                </div>
              ))}
          </div>
        </section>

        {/* Dev cards */}
        {myPlayer.devCards && myPlayer.devCards.length > 0 && (
          <section>
            <h3>Development Cards</h3>
            <div className="dev-cards">
              {myPlayer.devCards.map((c, i) => (
                <span key={i} className="dev-card-chip">{formatDevCard(c)}</span>
              ))}
            </div>
          </section>
        )}

        {/* Actions */}
        {isMyTurn && legalActions.length > 0 && (
          <section>
            <h3>Actions</h3>
            <div className="actions">
              {actionGroups.map((group) => (
                <button
                  key={group.label}
                  className={`action-btn ${group.primary ? 'primary' : ''}`}
                  onClick={() => {
                    if (group.actions.length === 1) {
                      performAction(group.actions[0])
                    } else {
                      setSelectedAction(selectedAction === group.label ? null : group.label)
                    }
                  }}
                >
                  {group.label}
                  {group.actions.length > 1 && ` (${group.actions.length})`}
                </button>
              ))}
            </div>
            {/* Sub-actions for trades etc */}
            {selectedAction && (
              <div className="actions" style={{ marginTop: 8 }}>
                {actionGroups
                  .find((g) => g.label === selectedAction)
                  ?.actions.map((a, i) => (
                    <button
                      key={i}
                      className="action-btn"
                      onClick={() => { performAction(a); setSelectedAction(null) }}
                    >
                      {formatSubAction(a)}
                    </button>
                  ))}
              </div>
            )}
          </section>
        )}

        {/* Steal target selection */}
        {phase === 'STEAL' && isMyTurn && (
          <section>
            <h3>Steal from</h3>
            <div className="actions">
              {legalActions
                .filter((a) => a.type === 'STEAL_FROM')
                .map((a, i) => (
                  <button key={i} className="action-btn" onClick={() => performAction(a)}>
                    {a.targetPlayer !== null && a.targetPlayer !== undefined
                      ? `Player ${a.targetPlayer} (${gameState.colors[a.targetPlayer]})`
                      : 'No one'}
                  </button>
                ))}
            </div>
          </section>
        )}

        {/* Log */}
        <section>
          <h3>Game Log</h3>
          <div className="log">
            {log.map((entry, i) => (
              <div key={i} className="log-entry">{entry}</div>
            ))}
          </div>
        </section>
      </div>

      {/* Board */}
      <div className="board-area">
        <BoardView
          board={gameState.board}
          legalActions={isMyTurn ? boardActions : []}
          onAction={performAction}
          robberHex={gameState.robberHex}
          currentPhase={phase}
        />
      </div>

      {/* Winner overlay */}
      {gameState.winner !== null && (
        <div className="winner-overlay" onClick={() => newGame('medium')}>
          <div className="winner-box">
            <h2>
              {gameState.winner === 0 ? '🎉 You Win!' : `Player ${gameState.winner} Wins!`}
            </h2>
            <p style={{ color: 'var(--text-dim)', marginBottom: 16 }}>
              {gameState.players[gameState.winner].victoryPoints} Victory Points
            </p>
            <button className="action-btn primary" style={{ fontSize: 16, padding: '12px 24px' }}>
              Play Again
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

function PlayerCard({ player, isActive, isHuman }: {
  player: PlayerData
  isActive: boolean
  isHuman: boolean
}) {
  return (
    <div
      className={`player-card ${isActive ? 'active' : ''}`}
      style={{ borderLeftColor: player.color }}
    >
      <div className="name">
        {isHuman ? '👤 You' : `🤖 Bot ${player.index}`}
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
    </div>
  )
}

interface ActionGroup {
  label: string
  actions: ActionData[]
  primary: boolean
}

function groupActions(actions: ActionData[]): ActionGroup[] {
  const groups: ActionGroup[] = []
  const byType = new Map<string, ActionData[]>()

  for (const a of actions) {
    // Skip board-click actions (handled by board view)
    if (['PLACE_SETUP_SETTLEMENT', 'PLACE_SETUP_ROAD', 'PLACE_ROBBER',
         'PLACE_ROAD_BUILDING_ROAD'].includes(a.type)) continue

    const key = a.type
    if (!byType.has(key)) byType.set(key, [])
    byType.get(key)!.push(a)
  }

  const labels: Record<string, string> = {
    ROLL_DICE: '🎲 Roll Dice',
    END_TURN: '⏭️ End Turn',
    BUILD_ROAD: '🛤️ Build Road',
    BUILD_SETTLEMENT: '🏠 Build Settlement',
    BUILD_CITY: '🏙️ Upgrade to City',
    BUY_DEV_CARD: '🃏 Buy Dev Card',
    PLAY_KNIGHT: '⚔️ Play Knight',
    PLAY_ROAD_BUILDING: '🛤️ Road Building',
    PLAY_YEAR_OF_PLENTY: '🎁 Year of Plenty',
    PLAY_MONOPOLY: '💰 Monopoly',
    TRADE_BANK: '🏦 Bank Trade',
    STEAL_FROM: '🥷 Steal',
    DISCARD_RESOURCES: '♻️ Discard',
  }

  for (const [type, acts] of byType) {
    groups.push({
      label: labels[type] || type,
      actions: acts,
      primary: type === 'ROLL_DICE',
    })
  }

  return groups
}

function formatPhase(phase: string): string {
  const map: Record<string, string> = {
    SETUP_SETTLEMENT_1: 'Setup: Place Settlement',
    SETUP_ROAD_1: 'Setup: Place Road',
    SETUP_SETTLEMENT_2: 'Setup: Place Settlement (2)',
    SETUP_ROAD_2: 'Setup: Place Road (2)',
    ROLL_DICE: 'Roll Dice',
    DISCARD: 'Discard Cards',
    MOVE_ROBBER: 'Move Robber',
    STEAL: 'Steal Resource',
    MAIN_TURN: 'Your Turn',
    GAME_OVER: 'Game Over',
    ROAD_BUILDING_1: 'Road Building (1/2)',
    ROAD_BUILDING_2: 'Road Building (2/2)',
  }
  return map[phase] || phase
}

function formatDevCard(card: string): string {
  const map: Record<string, string> = {
    knight: '⚔️ Knight',
    victory_point: '⭐ VP',
    road_building: '🛤️ Roads',
    year_of_plenty: '🎁 Plenty',
    monopoly: '💰 Monopoly',
  }
  return map[card] || card
}

function formatSubAction(a: ActionData): string {
  if (a.type === 'TRADE_BANK' && a.giveResource && a.getResource) {
    return `${a.giveResource} → ${a.getResource}`
  }
  if (a.type === 'PLAY_YEAR_OF_PLENTY') {
    return `${a.resource}${a.resource2 ? ' + ' + a.resource2 : ''}`
  }
  if (a.type === 'PLAY_MONOPOLY') {
    return `Take all ${a.resource}`
  }
  if (a.type === 'BUILD_CITY' && a.vertex !== undefined) {
    return `City @ v${a.vertex}`
  }
  if (a.type === 'PLAY_KNIGHT' && a.hexId !== undefined) {
    return `Hex ${a.hexId}`
  }
  return a.type
}
