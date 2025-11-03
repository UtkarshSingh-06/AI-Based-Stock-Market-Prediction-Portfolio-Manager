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
import './App.css';

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

function App() {
  const [darkMode, setDarkMode] = useState(false);
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
  const [activeTab, setActiveTab] = useState('predict'); // predict, train, history

  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode);
  }, [darkMode]);

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
      setError(err.response?.data?.detail || 'Failed to generate predictions. Please train the model first.');
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
      const response = await axios.post(`${API_URL}/api/train`, {
        symbol: selectedStock,
        start_date: startDate,
        end_date: endDate || undefined,
        epochs: 10,
        seq_len: 60
      });

      setSuccess(`Training started for ${selectedStock}. This may take a few minutes...`);
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
    labels: Array.from({ length: predictions.actual.length }, (_, i) => i + 1),
    datasets: [
      {
        label: 'Actual Price',
        data: predictions.actual,
        borderColor: darkMode ? '#60a5fa' : '#3b82f6',
        backgroundColor: darkMode ? 'rgba(96, 165, 250, 0.1)' : 'rgba(59, 130, 246, 0.1)',
        fill: true,
        tension: 0.4,
        pointRadius: 1,
      },
      {
        label: 'Predicted Price',
        data: predictions.predicted,
        borderColor: darkMode ? '#f87171' : '#ef4444',
        backgroundColor: darkMode ? 'rgba(248, 113, 113, 0.1)' : 'rgba(239, 68, 68, 0.1)',
        fill: true,
        tension: 0.4,
        pointRadius: 1,
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
          color: darkMode ? '#e5e7eb' : '#1f2937',
          font: {
            size: 12
          }
        }
      },
      title: {
        display: true,
        text: `${selectedStock} Stock Price Prediction`,
        color: darkMode ? '#e5e7eb' : '#1f2937',
        font: {
          size: 16,
          weight: 'bold'
        }
      }
    },
    scales: {
      x: {
        grid: {
          color: darkMode ? '#374151' : '#e5e7eb'
        },
        ticks: {
          color: darkMode ? '#9ca3af' : '#6b7280'
        }
      },
      y: {
        grid: {
          color: darkMode ? '#374151' : '#e5e7eb'
        },
        ticks: {
          color: darkMode ? '#9ca3af' : '#6b7280'
        }
      }
    }
  };

  return (
    <div className={`min-h-screen ${darkMode ? 'dark bg-gray-900' : 'bg-gray-50'}`}>
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 shadow-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-3">
              <TrendingUp className="w-8 h-8 text-blue-500" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                  Stock Prediction AI
                </h1>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Powered by Deep Learning GRU Models
                </p>
              </div>
            </div>
            <button
              onClick={() => setDarkMode(!darkMode)}
              className="p-2 rounded-lg bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
              data-testid="theme-toggle"
            >
              {darkMode ? <Sun className="w-5 h-5 text-yellow-400" /> : <Moon className="w-5 h-5 text-gray-700" />}
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Alerts */}
        {error && (
          <div className="mb-6 p-4 bg-red-100 dark:bg-red-900/30 border border-red-400 dark:border-red-700 rounded-lg flex items-start space-x-3 fade-in" data-testid="error-alert">
            <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
            <p className="text-red-700 dark:text-red-300">{error}</p>
          </div>
        )}
        
        {success && (
          <div className="mb-6 p-4 bg-green-100 dark:bg-green-900/30 border border-green-400 dark:border-green-700 rounded-lg flex items-start space-x-3 fade-in" data-testid="success-alert">
            <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400 flex-shrink-0 mt-0.5" />
            <p className="text-green-700 dark:text-green-300">{success}</p>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Panel - Stock Selection & Controls */}
          <div className="lg:col-span-1 space-y-6">
            {/* Stock Info Card */}
            {stockInfo && (
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 fade-in" data-testid="stock-info-card">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  {stockInfo.name}
                </h3>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600 dark:text-gray-400">Price</span>
                    <span className="text-2xl font-bold text-gray-900 dark:text-white">
                      ${stockInfo.current_price?.toFixed(2) || 'N/A'}
                    </span>
                  </div>
                  {stockInfo.change_percent !== null && (
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600 dark:text-gray-400">Change</span>
                      <span className={`font-semibold ${stockInfo.change_percent >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                        {stockInfo.change_percent >= 0 ? '+' : ''}{stockInfo.change_percent?.toFixed(2)}%
                      </span>
                    </div>
                  )}
                  <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600 dark:text-gray-400">Sector</span>
                      <span className="text-gray-900 dark:text-white">{stockInfo.sector}</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Stock Selection */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <Activity className="w-5 h-5 mr-2 text-blue-500" />
                Select Stock
              </h3>
              
              {/* Search */}
              <div className="relative mb-4">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search stocks..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  data-testid="stock-search"
                />
              </div>

              {/* Stock Dropdown */}
              <select
                value={selectedStock}
                onChange={(e) => setSelectedStock(e.target.value)}
                className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                data-testid="stock-select"
              >
                <option value="">Choose a stock...</option>
                {filteredStocks.map((stock) => (
                  <option key={stock.symbol} value={stock.symbol}>
                    {stock.symbol} - {stock.name}
                  </option>
                ))}
              </select>

              {/* Date Range */}
              <div className="mt-4 space-y-3">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Start Date
                  </label>
                  <input
                    type="date"
                    value={startDate}
                    onChange={(e) => setStartDate(e.target.value)}
                    className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
                    data-testid="start-date"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    End Date (Optional)
                  </label>
                  <input
                    type="date"
                    value={endDate}
                    onChange={(e) => setEndDate(e.target.value)}
                    className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
                    data-testid="end-date"
                  />
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 space-y-3">
              <button
                onClick={handlePredict}
                disabled={loading || !selectedStock}
                className="w-full px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-semibold rounded-lg transition-colors flex items-center justify-center space-x-2"
                data-testid="predict-button"
              >
                {loading ? (
                  <>
                    <Loader className="w-5 h-5 animate-spin" />
                    <span>Predicting...</span>
                  </>
                ) : (
                  <>
                    <BarChart3 className="w-5 h-5" />
                    <span>Generate Prediction</span>
                  </>
                )}
              </button>

              <button
                onClick={handleTrain}
                disabled={training || !selectedStock}
                className="w-full px-6 py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 text-white font-semibold rounded-lg transition-colors flex items-center justify-center space-x-2"
                data-testid="train-button"
              >
                {training ? (
                  <>
                    <Loader className="w-5 h-5 animate-spin" />
                    <span>Training...</span>
                  </>
                ) : (
                  <>
                    <RefreshCw className="w-5 h-5" />
                    <span>Train Model</span>
                  </>
                )}
              </button>
            </div>

            {/* Trained Models */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Trained Models ({trainedModels.length})
              </h3>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {trainedModels.length === 0 ? (
                  <p className="text-gray-500 dark:text-gray-400 text-sm">No models trained yet</p>
                ) : (
                  trainedModels.map((model) => (
                    <div
                      key={model.symbol}
                      className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-700 rounded"
                    >
                      <span className="text-sm font-medium text-gray-900 dark:text-white">
                        {model.symbol}
                      </span>
                      <CheckCircle className="w-4 h-4 text-green-500" />
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>

          {/* Right Panel - Chart & Metrics */}
          <div className="lg:col-span-2 space-y-6">
            {predictions ? (
              <>
                {/* Chart */}
                <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 fade-in" data-testid="prediction-chart">
                  <div className="h-96">
                    <Line data={chartData} options={chartOptions} />
                  </div>
                </div>

                {/* Metrics */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6" data-testid="metrics-card">
                    <h4 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">
                      RMSE
                    </h4>
                    <p className="text-2xl font-bold text-gray-900 dark:text-white">
                      {predictions.metrics.rmse.toFixed(4)}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      Root Mean Square Error
                    </p>
                  </div>

                  <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
                    <h4 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">
                      MAPE
                    </h4>
                    <p className="text-2xl font-bold text-gray-900 dark:text-white">
                      {predictions.metrics.mape.toFixed(2)}%
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      Mean Absolute Percentage Error
                    </p>
                  </div>

                  <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
                    <h4 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">
                      Data Points
                    </h4>
                    <p className="text-2xl font-bold text-gray-900 dark:text-white">
                      {predictions.length}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      Predictions Generated
                    </p>
                  </div>
                </div>
              </>
            ) : (
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-12 text-center">
                <BarChart3 className="w-16 h-16 mx-auto text-gray-400 dark:text-gray-600 mb-4" />
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                  No Predictions Yet
                </h3>
                <p className="text-gray-600 dark:text-gray-400">
                  Select a stock and generate predictions to see the chart
                </p>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white dark:bg-gray-800 mt-12 border-t border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-gray-600 dark:text-gray-400 text-sm">
            © 2025 Stock Prediction AI. Powered by GRU Deep Learning Models.
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
