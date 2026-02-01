import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { TrendingUp, TrendingDown, DollarSign, Activity, AlertCircle, BarChart3, Settings, User, LogOut } from 'lucide-react';

// Main Dashboard Component
export default function StockPredictionDashboard() {
  const [darkMode, setDarkMode] = useState(true);
  const [activeTab, setActiveTab] = useState('predictions');
  const [selectedStock, setSelectedStock] = useState('AAPL');
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiKey, setApiKey] = useState('demo_key_12345');
  
  const popularStocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX'];
  
  const theme = {
    bg: darkMode ? '#0f172a' : '#ffffff',
    cardBg: darkMode ? '#1e293b' : '#f8fafc',
    text: darkMode ? '#e2e8f0' : '#1e293b',
    textMuted: darkMode ? '#94a3b8' : '#64748b',
    border: darkMode ? '#334155' : '#e2e8f0',
    primary: '#3b82f6',
    success: '#10b981',
    danger: '#ef4444',
    warning: '#f59e0b'
  };

  // Simulated API call
  const fetchPredictions = async (symbol) => {
    setLoading(true);
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    // Generate mock data
    const mockData = generateMockPredictions(symbol);
    setPredictions(mockData);
    setLoading(false);
  };

  const generateMockPredictions = (symbol) => {
    const basePrice = Math.random() * 200 + 100;
    const data = [];
    
    for (let i = 0; i < 30; i++) {
      const trend = Math.sin(i / 5) * 10;
      const noise = (Math.random() - 0.5) * 5;
      const actual = basePrice + trend + noise;
      const predicted = actual + (Math.random() - 0.5) * 8;
      
      data.push({
        date: `Day ${i + 1}`,
        actual: actual.toFixed(2),
        predicted: predicted.toFixed(2),
        lower: (predicted - 5).toFixed(2),
        upper: (predicted + 5).toFixed(2)
      });
    }
    
    return {
      symbol,
      data,
      metrics: {
        rmse: (Math.random() * 5 + 2).toFixed(2),
        mape: (Math.random() * 3 + 1).toFixed(2),
        sharpe: (Math.random() * 1.5 + 0.5).toFixed(2),
        accuracy: (Math.random() * 5 + 92).toFixed(1)
      }
    };
  };

  useEffect(() => {
    fetchPredictions(selectedStock);
  }, [selectedStock]);

  return (
    <div style={{ 
      minHeight: '100vh', 
      background: theme.bg,
      color: theme.text,
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    }}>
      {/* Header */}
      <header style={{
        background: theme.cardBg,
        borderBottom: `1px solid ${theme.border}`,
        padding: '1rem 2rem',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <TrendingUp size={32} color={theme.primary} />
          <h1 style={{ fontSize: '1.5rem', fontWeight: 'bold', margin: 0 }}>
            Stock Predictor AI
          </h1>
        </div>
        
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <button
            onClick={() => setDarkMode(!darkMode)}
            style={{
              background: theme.cardBg,
              border: `1px solid ${theme.border}`,
              padding: '0.5rem 1rem',
              borderRadius: '0.5rem',
              color: theme.text,
              cursor: 'pointer'
            }}
          >
            {darkMode ? '☀️' : '🌙'}
          </button>
          
          <button style={{
            background: theme.primary,
            border: 'none',
            padding: '0.5rem 1rem',
            borderRadius: '0.5rem',
            color: 'white',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            <User size={18} />
            Account
          </button>
        </div>
      </header>

      {/* Main Content */}
      <div style={{ padding: '2rem' }}>
        {/* Stats Cards */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
          gap: '1.5rem',
          marginBottom: '2rem'
        }}>
          <StatsCard
            icon={<DollarSign />}
            title="Portfolio Value"
            value="$124,563"
            change="+12.5%"
            positive={true}
            theme={theme}
          />
          <StatsCard
            icon={<Activity />}
            title="Active Predictions"
            value="47"
            change="+3"
            positive={true}
            theme={theme}
          />
          <StatsCard
            icon={<BarChart3 />}
            title="Avg Accuracy"
            value="94.2%"
            change="+2.1%"
            positive={true}
            theme={theme}
          />
          <StatsCard
            icon={<TrendingUp />}
            title="Total Return"
            value="+28.4%"
            change="This month"
            positive={true}
            theme={theme}
          />
        </div>

        {/* Stock Selector */}
        <div style={{
          background: theme.cardBg,
          padding: '1.5rem',
          borderRadius: '1rem',
          marginBottom: '2rem',
          border: `1px solid ${theme.border}`
        }}>
          <h3 style={{ marginTop: 0, marginBottom: '1rem' }}>Select Stock</h3>
          <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
            {popularStocks.map(stock => (
              <button
                key={stock}
                onClick={() => setSelectedStock(stock)}
                style={{
                  padding: '0.75rem 1.5rem',
                  borderRadius: '0.5rem',
                  border: `2px solid ${selectedStock === stock ? theme.primary : theme.border}`,
                  background: selectedStock === stock ? theme.primary + '20' : theme.bg,
                  color: selectedStock === stock ? theme.primary : theme.text,
                  cursor: 'pointer',
                  fontWeight: selectedStock === stock ? 'bold' : 'normal',
                  transition: 'all 0.2s'
                }}
              >
                {stock}
              </button>
            ))}
          </div>
        </div>

        {/* Main Chart */}
        {loading ? (
          <div style={{
            background: theme.cardBg,
            padding: '4rem',
            borderRadius: '1rem',
            textAlign: 'center',
            border: `1px solid ${theme.border}`
          }}>
            <div style={{
              width: '50px',
              height: '50px',
              border: `4px solid ${theme.border}`,
              borderTop: `4px solid ${theme.primary}`,
              borderRadius: '50%',
              animation: 'spin 1s linear infinite',
              margin: '0 auto'
            }} />
            <p style={{ marginTop: '1rem', color: theme.textMuted }}>Loading predictions...</p>
            <style>{`
              @keyframes spin {
                to { transform: rotate(360deg); }
              }
            `}</style>
          </div>
        ) : predictions && (
          <div style={{
            background: theme.cardBg,
            padding: '1.5rem',
            borderRadius: '1rem',
            border: `1px solid ${theme.border}`
          }}>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '1.5rem'
            }}>
              <div>
                <h2 style={{ margin: 0, fontSize: '1.5rem' }}>{predictions.symbol}</h2>
                <p style={{ color: theme.textMuted, margin: '0.25rem 0 0 0' }}>
                  30-Day Prediction with Confidence Intervals
                </p>
              </div>
              
              <div style={{ display: 'flex', gap: '2rem' }}>
                <MetricBadge label="RMSE" value={predictions.metrics.rmse} theme={theme} />
                <MetricBadge label="MAPE" value={`${predictions.metrics.mape}%`} theme={theme} />
                <MetricBadge label="Accuracy" value={`${predictions.metrics.accuracy}%`} theme={theme} />
              </div>
            </div>

            <ResponsiveContainer width="100%" height={400}>
              <AreaChart data={predictions.data}>
                <defs>
                  <linearGradient id="confidenceGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={theme.primary} stopOpacity={0.2}/>
                    <stop offset="95%" stopColor={theme.primary} stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke={theme.border} />
                <XAxis dataKey="date" stroke={theme.textMuted} />
                <YAxis stroke={theme.textMuted} />
                <Tooltip 
                  contentStyle={{ 
                    background: theme.cardBg, 
                    border: `1px solid ${theme.border}`,
                    borderRadius: '0.5rem'
                  }}
                />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="upper"
                  stroke="none"
                  fill="url(#confidenceGradient)"
                  name="Upper Bound"
                />
                <Area
                  type="monotone"
                  dataKey="lower"
                  stroke="none"
                  fill="url(#confidenceGradient)"
                  name="Lower Bound"
                />
                <Line
                  type="monotone"
                  dataKey="actual"
                  stroke={theme.success}
                  strokeWidth={2}
                  dot={false}
                  name="Actual Price"
                />
                <Line
                  type="monotone"
                  dataKey="predicted"
                  stroke={theme.primary}
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  dot={false}
                  name="Predicted Price"
                />
              </AreaChart>
            </ResponsiveContainer>

            {/* Action Buttons */}
            <div style={{
              marginTop: '1.5rem',
              display: 'flex',
              gap: '1rem'
            }}>
              <button style={{
                flex: 1,
                padding: '0.75rem',
                background: theme.primary,
                color: 'white',
                border: 'none',
                borderRadius: '0.5rem',
                cursor: 'pointer',
                fontWeight: 'bold',
                fontSize: '0.95rem'
              }}>
                Train New Model
              </button>
              <button style={{
                flex: 1,
                padding: '0.75rem',
                background: theme.success,
                color: 'white',
                border: 'none',
                borderRadius: '0.5rem',
                cursor: 'pointer',
                fontWeight: 'bold',
                fontSize: '0.95rem'
              }}>
                Run Backtest
              </button>
              <button style={{
                flex: 1,
                padding: '0.75rem',
                background: theme.cardBg,
                color: theme.text,
                border: `1px solid ${theme.border}`,
                borderRadius: '0.5rem',
                cursor: 'pointer',
                fontWeight: 'bold',
                fontSize: '0.95rem'
              }}>
                Export Data
              </button>
            </div>
          </div>
        )}

        {/* Recent Predictions Table */}
        <div style={{
          background: theme.cardBg,
          padding: '1.5rem',
          borderRadius: '1rem',
          marginTop: '2rem',
          border: `1px solid ${theme.border}`
        }}>
          <h3 style={{ marginTop: 0, marginBottom: '1rem' }}>Recent Predictions</h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: `1px solid ${theme.border}` }}>
                  <th style={{ padding: '1rem', textAlign: 'left', color: theme.textMuted, fontWeight: '600' }}>Symbol</th>
                  <th style={{ padding: '1rem', textAlign: 'left', color: theme.textMuted, fontWeight: '600' }}>Date</th>
                  <th style={{ padding: '1rem', textAlign: 'left', color: theme.textMuted, fontWeight: '600' }}>Predicted</th>
                  <th style={{ padding: '1rem', textAlign: 'left', color: theme.textMuted, fontWeight: '600' }}>Actual</th>
                  <th style={{ padding: '1rem', textAlign: 'left', color: theme.textMuted, fontWeight: '600' }}>Accuracy</th>
                  <th style={{ padding: '1rem', textAlign: 'left', color: theme.textMuted, fontWeight: '600' }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'].map((symbol, idx) => {
                  const predicted = (Math.random() * 200 + 100).toFixed(2);
                  const actual = (Math.random() * 200 + 100).toFixed(2);
                  const accuracy = (Math.random() * 5 + 92).toFixed(1);
                  
                  return (
                    <tr key={idx} style={{ borderBottom: `1px solid ${theme.border}` }}>
                      <td style={{ padding: '1rem', fontWeight: 'bold' }}>{symbol}</td>
                      <td style={{ padding: '1rem', color: theme.textMuted }}>
                        {new Date(Date.now() - idx * 86400000).toLocaleDateString()}
                      </td>
                      <td style={{ padding: '1rem' }}>${predicted}</td>
                      <td style={{ padding: '1rem' }}>${actual}</td>
                      <td style={{ padding: '1rem' }}>
                        <span style={{
                          color: parseFloat(accuracy) > 90 ? theme.success : theme.warning,
                          fontWeight: 'bold'
                        }}>
                          {accuracy}%
                        </span>
                      </td>
                      <td style={{ padding: '1rem' }}>
                        <span style={{
                          padding: '0.25rem 0.75rem',
                          borderRadius: '1rem',
                          background: theme.success + '20',
                          color: theme.success,
                          fontSize: '0.875rem',
                          fontWeight: '500'
                        }}>
                          Completed
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

// Stats Card Component
function StatsCard({ icon, title, value, change, positive, theme }) {
  return (
    <div 
      style={{
        background: theme.cardBg,
        padding: '1.5rem',
        borderRadius: '1rem',
        border: `1px solid ${theme.border}`,
        transition: 'transform 0.2s',
        cursor: 'pointer'
      }}
      onMouseEnter={(e) => e.currentTarget.style.transform = 'translateY(-4px)'}
      onMouseLeave={(e) => e.currentTarget.style.transform = 'translateY(0)'}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
        <div style={{
          padding: '0.75rem',
          borderRadius: '0.75rem',
          background: theme.primary + '20',
          color: theme.primary
        }}>
          {React.cloneElement(icon, { size: 24 })}
        </div>
        <span style={{
          color: positive ? theme.success : theme.danger,
          fontSize: '0.875rem',
          fontWeight: 'bold'
        }}>
          {change}
        </span>
      </div>
      <h3 style={{
        margin: '1rem 0 0.5rem 0',
        fontSize: '1.75rem',
        fontWeight: 'bold'
      }}>
        {value}
      </h3>
      <p style={{
        margin: 0,
        color: theme.textMuted,
        fontSize: '0.875rem'
      }}>
        {title}
      </p>
    </div>
  );
}

// Metric Badge Component
function MetricBadge({ label, value, theme }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{
        fontSize: '0.75rem',
        color: theme.textMuted,
        marginBottom: '0.25rem',
        textTransform: 'uppercase',
        fontWeight: '600'
      }}>
        {label}
      </div>
      <div style={{
        fontSize: '1.25rem',
        fontWeight: 'bold',
        color: theme.primary
      }}>
        {value}
      </div>
    </div>
  );
}