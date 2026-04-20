// Type definitions for the Catan web client

export interface HexData {
  id: number
  q: number
  r: number
  terrain: string
  number: number
  cx: number
  cy: number
  hasRobber: boolean
  vertices: number[]
}

export interface VertexData {
  id: number
  x: number
  y: number
  building: string | null
  owner: number | null
  harbor: string | null
}

export interface EdgeData {
  id: number
  v1: number
  v2: number
  x1: number
  y1: number
  x2: number
  y2: number
  road: number | null
}

export interface BoardData {
  hexes: HexData[]
  vertices: VertexData[]
  edges: EdgeData[]
}

export interface PlayerData {
  index: number
  color: string
  victoryPoints: number
  numSettlements: number
  numCities: number
  numRoads: number
  knightsPlayed: number
  hasLongestRoad: boolean
  hasLargestArmy: boolean
  longestRoadLength: number
  totalResources: number
  numDevCards: number
  harbors: string[]
  resources?: Record<string, number>
  devCards?: string[]
  newDevCards?: string[]
  tradeRatios?: Record<string, number>
}

export interface GameStateData {
  board: BoardData
  players: PlayerData[]
  currentPlayer: number
  phase: string
  turnNumber: number
  diceRoll: number[] | null
  winner: number | null
  robberHex: number
  colors: string[]
}

export interface ActionData {
  type: string
  vertex?: number
  edge?: number
  hexId?: number
  targetPlayer?: number
  resource?: string
  resource2?: string
  giveResource?: string
  getResource?: string
  discard?: Record<string, number>
}

export interface ServerMessage {
  type: string
  state?: GameStateData
  legalActions?: ActionData[]
  aiActions?: Array<{ player: number; action: ActionData }>
}

export interface EpochStats {
  epoch: number
  avg_game_length: number
  avg_reward: number
  policy_loss: number
  value_loss: number
  entropy: number
  wins: Record<string, number>
  avg_settlements: number
  avg_cities: number
  avg_roads: number
  avg_dev_cards: number
  avg_knights: number
  longest_road_wins: number
  largest_army_wins: number
}
