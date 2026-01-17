import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import {
  TrendingUp,
  Moon,
  Sun,
  Activity,
  BarChart3,
  RefreshCw,
  Search,
  AlertCircle,
  CheckCircle,
  Loader
} from 'lucide-react';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const API_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

// Modern color palette
const colors = {
  primary: '#667eea',
  secondary: '#764ba2',
  accent: '#f5576c',
  success: '#10b981',
  error: '#ef4444',
};

function App() {
  const [darkMode, setDarkMode] = useState(true);
  const [stocks, setStocks] = useState([]);
  const [selectedStock, setSelectedStock] = useState('');
  const [stockInfo, setStockInfo] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [startDate, setStartDate] = useState('2020-01-01');
  const [endDate, setEndDate] = useState('');
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [training, setTraining] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [trainedModels, setTrainedModels] = useState([]);

  useEffect(() => {
    fetchStocks();
    fetchTrainedModels();
  }, []);

  useEffect(() => {
    if (selectedStock) {
      fetchStockInfo(selectedStock);
    }
  }, [selectedStock]);

  const fetchStocks = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/stocks`);
      setStocks(response.data.stocks);
    } catch (err) {
      setError('Failed to fetch stocks');
    }
  };

  const fetchStockInfo = async (symbol) => {
    try {
      const response = await axios.get(`${API_URL}/api/stock/${symbol}`);
      setStockInfo(response.data);
    } catch (err) {
      console.error('Failed to fetch stock info', err);
    }
  };

  const fetchTrainedModels = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/models`);
      setTrainedModels(response.data.models);
    } catch (err) {
      console.error('Failed to fetch trained models');
    }
  };

  const handlePredict = async () => {
    if (!selectedStock) {
      setError('Please select a stock');
      return;
    }

    setLoading(true);
    setError('');
    setPredictions(null);

    try {
      const response = await axios.post(`${API_URL}/api/predict`, {
        symbol: selectedStock,
        start_date: startDate,
        end_date: endDate || undefined,
        seq_len: 60
      });

      setPredictions(response.data);
      setSuccess('Predictions generated successfully!');
      setTimeout(() => setSuccess(''), 3000);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to generate predictions. Train the model first.');
    } finally {
      setLoading(false);
    }
  };

  const handleTrain = async () => {
    if (!selectedStock) {
      setError('Please select a stock');
      return;
    }

    setTraining(true);
    setError('');

    try {
      await axios.post(`${API_URL}/api/train`, {
        symbol: selectedStock,
        start_date: startDate,
        end_date: endDate || undefined,
        epochs: 10,
        seq_len: 60
      });

      setSuccess(`Training started for ${selectedStock}. This will take a few minutes...`);
      setTimeout(() => {
        setSuccess('');
        fetchTrainedModels();
      }, 5000);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to start training');
    } finally {
      setTraining(false);
    }
  };

  const filteredStocks = stocks.filter(stock =>
    stock.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
    stock.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const chartData = predictions ? {
    labels: Array.from({ length: predictions.actual.length }, (_, i) => `Day ${i + 1}`),
    datasets: [
      {
        label: 'Actual Price',
        data: predictions.actual,
        borderColor: colors.primary,
        backgroundColor: `${colors.primary}20`,
        fill: true,
        tension: 0.4,
        pointRadius: 2,
        pointHoverRadius: 6,
        borderWidth: 3,
      },
      {
        label: 'Predicted Price',
        data: predictions.predicted,
        borderColor: colors.accent,
        backgroundColor: `${colors.accent}20`,
        fill: true,
        tension: 0.4,
        pointRadius: 2,
        pointHoverRadius: 6,
        borderWidth: 3,
      }
    ]
  } : null;

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: darkMode ? '#e2e8f0' : '#1e293b',
          font: { size: 13, weight: '600' },
          padding: 20,
          usePointStyle: true,
        }
      },
      title: {
        display: true,
        text: `${selectedStock} Stock Price Prediction`,
        color: darkMode ? '#f1f5f9' : '#1e293b',
        font: { size: 20, weight: 'bold' },
        padding: 25,
      },
      tooltip: {
        backgroundColor: darkMode ? '#1e293b' : '#ffffff',
        titleColor: darkMode ? '#f1f5f9' : '#1e293b',
        bodyColor: darkMode ? '#cbd5e1' : '#475569',
        borderColor: darkMode ? '#334155' : '#e2e8f0',
        borderWidth: 1,
        padding: 15,
        displayColors: true,
        callbacks: {
          label: function(context) {
            return `${context.dataset.label}: $${context.parsed.y.toFixed(2)}`;
          }
        }
      }
    },
    scales: {
      x: {
        grid: { color: darkMode ? '#334155' : '#e2e8f0', drawBorder: false },
        ticks: { color: darkMode ? '#94a3b8' : '#64748b', font: { size: 11 } }
      },
      y: {
        grid: { color: darkMode ? '#334155' : '#e2e8f0', drawBorder: false },
        ticks: {
          color: darkMode ? '#94a3b8' : '#64748b',
          font: { size: 11 },
          callback: (value) => '$' + value.toFixed(0)
        }
      }
    },
    interaction: { intersect: false, mode: 'index' },
  };

  const bgColor = darkMode ? '#0f172a' : '#f8fafc';
  const cardBg = darkMode ? '#1e293b' : '#ffffff';
  const textPrimary = darkMode ? '#f1f5f9' : '#1e293b';
  const textSecondary = darkMode ? '#94a3b8' : '#64748b';
  const borderColor = darkMode ? '#334155' : '#e2e8f0';

  return (
    <div style={{ minHeight: '100vh', backgroundColor: bgColor, transition: 'all 0.3s' }}>
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes slideIn { from { opacity: 0; transform: translateY(-10px); } to { opacity: 1; transform: translateY(0); } }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif; }
        .gradient-text { background: linear-gradient(135deg, ${colors.primary} 0%, ${colors.secondary} 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .btn-hover:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(0,0,0,0.2); }
        .card-hover:hover { transform: translateY(-4px); box-shadow: 0 12px 24px rgba(0,0,0,0.15); }
      `}</style>

      {/* Header */}
      <header style={{ backgroundColor: cardBg, boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderBottom: `1px solid ${borderColor}` }}>
        <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '0 1.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '1.5rem 0' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <TrendingUp size={40} color={colors.primary} />
              <div>
                <h1 className="gradient-text" style={{ fontSize: '1.875rem', fontWeight: 'bold', margin: 0 }}>Stock Prediction AI</h1>
                <p style={{ fontSize: '0.875rem', color: textSecondary, margin: '0.25rem 0 0 0' }}>Powered by Deep Learning GRU Models</p>
              </div>
            </div>
            <button
              onClick={() => setDarkMode(!darkMode)}
              style={{ padding: '0.75rem', borderRadius: '0.75rem', backgroundColor: darkMode ? '#334155' : '#f1f5f9', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', transition: 'all 0.3s' }}
              onMouseEnter={(e) => e.target.style.transform = 'scale(1.1)'}
              onMouseLeave={(e) => e.target.style.transform = 'scale(1)'}
            >
              {darkMode ? <Sun size={20} color="#f59e0b" /> : <Moon size={20} color="#64748b" />}
            </button>
          </div>
        </div>
      </header>

      {/* Main */}
      <main style={{ maxWidth: '1400px', margin: '0 auto', padding: '2rem 1.5rem' }}>
        {error && (
          <div style={{ marginBottom: '1.5rem', padding: '1rem 1.25rem', borderRadius: '0.75rem', display: 'flex', alignItems: 'flex-start', gap: '0.75rem', animation: 'slideIn 0.3s', backgroundColor: darkMode ? 'rgba(239,68,68,0.1)' : '#fee2e2', border: `1px solid ${darkMode ? '#dc2626' : '#fca5a5'}`, color: darkMode ? '#fca5a5' : '#991b1b' }}>
            <AlertCircle size={20} />
            <p style={{ margin: 0 }}>{error}</p>
          </div>
        )}
        
        {success && (
          <div style={{ marginBottom: '1.5rem', padding: '1rem 1.25rem', borderRadius: '0.75rem', display: 'flex', alignItems: 'flex-start', gap: '0.75rem', animation: 'slideIn 0.3s', backgroundColor: darkMode ? 'rgba(34,197,94,0.1)' : '#d1fae5', border: `1px solid ${darkMode ? '#16a34a' : '#86efac'}`, color: darkMode ? '#86efac' : '#065f46' }}>
            <CheckCircle size={20} />
            <p style={{ margin: 0 }}>{success}</p>
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: window.innerWidth >= 1024 ? 'minmax(350px, 1fr) 2fr' : '1fr', gap: '1.5rem' }}>
          {/* Left Panel */}
          <div>
            {stockInfo && (
              <div className="card-hover" style={{ backgroundColor: cardBg, borderRadius: '1rem', boxShadow: '0 4px 6px rgba(0,0,0,0.1)', padding: '1.5rem', marginBottom: '1.5rem', border: `1px solid ${borderColor}`, transition: 'all 0.3s' }}>
                <h3 style={{ fontSize: '1.125rem', fontWeight: '600', color: textPrimary, marginBottom: '1rem' }}>{stockInfo.name}</h3>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '1rem', backgroundColor: darkMode ? '#0f172a' : '#f8fafc', borderRadius: '0.75rem', marginBottom: '0.75rem' }}>
                  <span style={{ fontSize: '0.875rem', color: textSecondary }}>Current Price</span>
                  <span className="gradient-text" style={{ fontSize: '2rem', fontWeight: 'bold' }}>
                    ${stockInfo.current_price?.toFixed(2) || 'N/A'}
                  </span>
                </div>
                {stockInfo.change_percent !== null && (
                  <div style={{ textAlign: 'center', marginTop: '0.5rem' }}>
                    <span style={{ fontSize: '1.25rem', fontWeight: '600', color: stockInfo.change_percent >= 0 ? colors.success : colors.error }}>
                      {stockInfo.change_percent >= 0 ? '+' : ''}{stockInfo.change_percent?.toFixed(2)}%
                    </span>
                  </div>
                )}
              </div>
            )}

            <div className="card-hover" style={{ backgroundColor: cardBg, borderRadius: '1rem', boxShadow: '0 4px 6px rgba(0,0,0,0.1)', padding: '1.5rem', marginBottom: '1.5rem', border: `1px solid ${borderColor}`, transition: 'all 0.3s' }}>
              <h3 style={{ fontSize: '1.125rem', fontWeight: '600', color: textPrimary, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Activity size={20} color={colors.primary} /> Select Stock
              </h3>
              
              <div style={{ position: 'relative', marginBottom: '1rem' }}>
                <Search size={18} style={{ position: 'absolute', left: '1rem', top: '50%', transform: 'translateY(-50%)', color: '#94a3b8', pointerEvents: 'none' }} />
                <input
                  type="text"
                  placeholder="Search stocks..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  style={{ width: '100%', padding: '0.75rem 1rem 0.75rem 3rem', border: `1px solid ${borderColor}`, borderRadius: '0.5rem', backgroundColor: darkMode ? '#0f172a' : '#ffffff', color: textPrimary, fontSize: '0.95rem', outline: 'none' }}
                />
              </div>

              <select
                value={selectedStock}
                onChange={(e) => setSelectedStock(e.target.value)}
                style={{ width: '100%', padding: '0.75rem 1rem', border: `1px solid ${borderColor}`, borderRadius: '0.5rem', backgroundColor: darkMode ? '#0f172a' : '#ffffff', color: textPrimary, fontSize: '0.95rem', cursor: 'pointer', outline: 'none' }}
              >
                <option value="">Choose a stock...</option>
                {filteredStocks.map((stock) => (
                  <option key={stock.symbol} value={stock.symbol}>
                    {stock.symbol} - {stock.name}
                  </option>
                ))}
              </select>

              <div style={{ marginTop: '1.5rem' }}>
                <div style={{ marginBottom: '1rem' }}>
                  <label style={{ display: 'block', fontSize: '0.875rem', fontWeight: '500', color: textSecondary, marginBottom: '0.5rem' }}>Start Date</label>
                  <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} style={{ width: '100%', padding: '0.75rem 1rem', border: `1px solid ${borderColor}`, borderRadius: '0.5rem', backgroundColor: darkMode ? '#0f172a' : '#ffffff', color: textPrimary, fontSize: '0.95rem', outline: 'none' }} />
                </div>
                <div>
                  <label style={{ display: 'block', fontSize: '0.875rem', fontWeight: '500', color: textSecondary, marginBottom: '0.5rem' }}>End Date (Optional)</label>
                  <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} style={{ width: '100%', padding: '0.75rem 1rem', border: `1px solid ${borderColor}`, borderRadius: '0.5rem', backgroundColor: darkMode ? '#0f172a' : '#ffffff', color: textPrimary, fontSize: '0.95rem', outline: 'none' }} />
                </div>
              </div>
            </div>

            <div className="card-hover" style={{ backgroundColor: cardBg, borderRadius: '1rem', boxShadow: '0 4px 6px rgba(0,0,0,0.1)', padding: '1.5rem', marginBottom: '1.5rem', border: `1px solid ${borderColor}`, transition: 'all 0.3s' }}>
              <button
                onClick={handlePredict}
                disabled={loading || !selectedStock}
                className="btn-hover"
                style={{ width: '100%', padding: '0.875rem 1.5rem', fontWeight: '600', fontSize: '0.95rem', borderRadius: '0.75rem', border: 'none', cursor: loading || !selectedStock ? 'not-allowed' : 'pointer', transition: 'all 0.3s', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', opacity: loading || !selectedStock ? 0.6 : 1, background: `linear-gradient(135deg, ${colors.primary} 0%, ${colors.secondary} 100%)`, color: '#ffffff', boxShadow: '0 4px 12px rgba(0,0,0,0.15)' }}
              >
                {loading ? (
                  <>
                    <Loader size={20} style={{ animation: 'spin 1s linear infinite' }} />
                    <span>Predicting...</span>
                  </>
                ) : (
                  <>
                    <BarChart3 size={20} />
                    <span>Generate Prediction</span>
                  </>
                )}
              </button>

              <button
                onClick={handleTrain}
                disabled={training || !selectedStock}
                className="btn-hover"
                style={{ width: '100%', padding: '0.875rem 1.5rem', fontWeight: '600', fontSize: '0.95rem', borderRadius: '0.75rem', border: 'none', cursor: training || !selectedStock ? 'not-allowed' : 'pointer', transition: 'all 0.3s', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', opacity: training || !selectedStock ? 0.6 : 1, background: `linear-gradient(135deg, #f093fb 0%, ${colors.accent} 100%)`, color: '#ffffff', boxShadow: '0 4px 12px rgba(0,0,0,0.15)', marginTop: '1rem' }}
              >
                {training ? (
                  <>
                    <Loader size={20} style={{ animation: 'spin 1s linear infinite' }} />
                    <span>Training...</span>
                  </>
                ) : (
                  <>
                    <RefreshCw size={20} />
                    <span>Train New Model</span>
                  </>
                )}
              </button>
            </div>

            <div className="card-hover" style={{ backgroundColor: cardBg, borderRadius: '1rem', boxShadow: '0 4px 6px rgba(0,0,0,0.1)', padding: '1.5rem', marginBottom: '1.5rem', border: `1px solid ${borderColor}`, transition: 'all 0.3s' }}>
              <h3 style={{ fontSize: '1.125rem', fontWeight: '600', color: textPrimary, marginBottom: '1rem' }}>Trained Models ({trainedModels.length})</h3>
              <div style={{ maxHeight: '250px', overflowY: 'auto' }}>
                {trainedModels.length === 0 ? (
                  <p style={{ color: textSecondary, fontSize: '0.875rem' }}>No models trained yet</p>
                ) : (
                  trainedModels.map((model) => (
                    <div key={model.symbol} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0.75rem 1rem', backgroundColor: darkMode ? '#0f172a' : '#f8fafc', borderRadius: '0.5rem', marginBottom: '0.5rem', border: `1px solid ${borderColor}` }}>
                      <span style={{ fontSize: '0.875rem', fontWeight: '600', color: textPrimary }}>{model.symbol}</span>
                      <CheckCircle size={16} color={colors.success} />
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>

          {/* Right Panel */}
          <div>
            {predictions ? (
              <>
                <div className="card-hover" style={{ backgroundColor: cardBg, borderRadius: '1rem', boxShadow: '0 4px 6px rgba(0,0,0,0.1)', padding: '1.5rem', marginBottom: '1.5rem', border: `1px solid ${borderColor}`, transition: 'all 0.3s' }}>
                  <div style={{ height: '450px', position: 'relative' }}>
                    <Line data={chartData} options={chartOptions} />
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
                  <div className="card-hover" style={{ padding: '1.5rem', backgroundColor: darkMode ? '#0f172a' : '#f8fafc', borderRadius: '0.75rem', border: `1px solid ${borderColor}`, transition: 'all 0.3s' }}>
                    <h4 style={{ fontSize: '0.875rem', color: textSecondary, marginBottom: '0.5rem' }}>RMSE</h4>
                    <p className="gradient-text" style={{ fontSize: '1.875rem', fontWeight: 'bold' }}>{predictions.metrics.rmse.toFixed(4)}</p>
                    <p style={{ fontSize: '0.75rem', color: textSecondary, marginTop: '0.5rem' }}>Root Mean Square Error</p>
                  </div>

                  <div className="card-hover" style={{ padding: '1.5rem', backgroundColor: darkMode ? '#0f172a' : '#f8fafc', borderRadius: '0.75rem', border: `1px solid ${borderColor}`, transition: 'all 0.3s' }}>
                    <h4 style={{ fontSize: '0.875rem', color: textSecondary, marginBottom: '0.5rem' }}>MAPE</h4>
                    <p className="gradient-text" style={{ fontSize: '1.875rem', fontWeight: 'bold' }}>{predictions.metrics.mape.toFixed(2)}%</p>
                    <p style={{ fontSize: '0.75rem', color: textSecondary, marginTop: '0.5rem' }}>Mean Absolute % Error</p>
                  </div>

                  <div className="card-hover" style={{ padding: '1.5rem', backgroundColor: darkMode ? '#0f172a' : '#f8fafc', borderRadius: '0.75rem', border: `1px solid ${borderColor}`, transition: 'all 0.3s' }}>
                    <h4 style={{ fontSize: '0.875rem', color: textSecondary, marginBottom: '0.5rem' }}>Data Points</h4>
                    <p className="gradient-text" style={{ fontSize: '1.875rem', fontWeight: 'bold' }}>{predictions.length}</p>
                    <p style={{ fontSize: '0.75rem', color: textSecondary, marginTop: '0.5rem' }}>Predictions Generated</p>
                  </div>
                </div>
              </>
            ) : (
              <div className="card-hover" style={{ backgroundColor: cardBg, borderRadius: '1rem', boxShadow: '0 4px 6px rgba(0,0,0,0.1)', padding: '4rem 2rem', textAlign: 'center', border: `1px solid ${borderColor}`, transition: 'all 0.3s' }}>
                <BarChart3 size={64} color={darkMode ? '#475569' : '#cbd5e1'} style={{ margin: '0 auto 1rem' }} />
                <h3 style={{ fontSize: '1.5rem', fontWeight: '600', marginBottom: '0.5rem', color: textPrimary }}>No Predictions Yet</h3>
                <p style={{ color: textSecondary }}>Select a stock and generate predictions to see the interactive chart</p>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer style={{ backgroundColor: cardBg, marginTop: '3rem', borderTop: `1px solid ${borderColor}`, padding: '1.5rem 0', textAlign: 'center', color: textSecondary, fontSize: '0.875rem' }}>
        <p>© 2025 Stock Prediction AI. Powered by GRU Deep Learning Models.</p>
      </footer>
    </div>
  );
}

export default App;