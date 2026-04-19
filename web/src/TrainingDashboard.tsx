import { useEffect, useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  BarChart, Bar, ResponsiveContainer, AreaChart, Area,
} from 'recharts'
import type { EpochStats } from './types'

export default function TrainingDashboard() {
  const [stats, setStats] = useState<EpochStats[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('/api/training/stats')
      .then((r) => r.json())
      .then((data) => {
        setStats(data.epochs || [])
        setLoading(false)
      })
      .catch(() => setLoading(false))

    // Poll every 10s
    const interval = setInterval(() => {
      fetch('/api/training/stats')
        .then((r) => r.json())
        .then((data) => setStats(data.epochs || []))
        .catch(() => {})
    }, 10000)
    return () => clearInterval(interval)
  }, [])

  if (loading) return <div className="training-dashboard"><p>Loading training data...</p></div>
  if (stats.length === 0) {
    return (
      <div className="training-dashboard">
        <h2 style={{ marginBottom: 16 }}>Training Dashboard</h2>
        <p style={{ color: 'var(--text-dim)' }}>
          No training data yet. Run <code>python scripts/train.py</code> to start training.
        </p>
      </div>
    )
  }

  return (
    <div className="training-dashboard">
      <h2 style={{ marginBottom: 24 }}>Training Dashboard — {stats.length} Epochs</h2>

      <div className="chart-container">
        <h3>Loss Curves</h3>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={stats}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="epoch" stroke="#666" />
            <YAxis stroke="#666" />
            <Tooltip contentStyle={{ background: '#16213e', border: '1px solid #333' }} />
            <Legend />
            <Line type="monotone" dataKey="policy_loss" stroke="#e94560" dot={false} name="Policy Loss" />
            <Line type="monotone" dataKey="value_loss" stroke="#3498db" dot={false} name="Value Loss" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-container">
        <h3>Average Reward & Entropy</h3>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={stats}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="epoch" stroke="#666" />
            <YAxis stroke="#666" />
            <Tooltip contentStyle={{ background: '#16213e', border: '1px solid #333' }} />
            <Legend />
            <Line type="monotone" dataKey="avg_reward" stroke="#4caf50" dot={false} name="Avg Reward" />
            <Line type="monotone" dataKey="entropy" stroke="#ff9800" dot={false} name="Entropy" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-container">
        <h3>Game Length</h3>
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={stats}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="epoch" stroke="#666" />
            <YAxis stroke="#666" />
            <Tooltip contentStyle={{ background: '#16213e', border: '1px solid #333' }} />
            <Area type="monotone" dataKey="avg_game_length" stroke="#9b59b6" fill="#9b59b640" name="Avg Game Length" />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-container">
        <h3>Winner Strategy Profile</h3>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={stats}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="epoch" stroke="#666" />
            <YAxis stroke="#666" />
            <Tooltip contentStyle={{ background: '#16213e', border: '1px solid #333' }} />
            <Legend />
            <Line type="monotone" dataKey="avg_settlements" stroke="#f39c12" dot={false} name="Settlements" />
            <Line type="monotone" dataKey="avg_cities" stroke="#e74c3c" dot={false} name="Cities" />
            <Line type="monotone" dataKey="avg_roads" stroke="#3498db" dot={false} name="Roads" />
            <Line type="monotone" dataKey="avg_knights" stroke="#9b59b6" dot={false} name="Knights" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-container">
        <h3>Special Victory Methods</h3>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={stats.slice(-20)}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="epoch" stroke="#666" />
            <YAxis stroke="#666" />
            <Tooltip contentStyle={{ background: '#16213e', border: '1px solid #333' }} />
            <Legend />
            <Bar dataKey="longest_road_wins" fill="#3498db" name="Longest Road Wins" />
            <Bar dataKey="largest_army_wins" fill="#9b59b6" name="Largest Army Wins" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
