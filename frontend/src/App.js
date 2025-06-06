import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Play, Pause, Settings, TrendingUp, TrendingDown, AlertCircle, CheckCircle } from 'lucide-react';
import toast, { Toaster } from 'react-hot-toast';
import axios from 'axios';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || '';

function App() {
  const [backtests, setBacktests] = useState([]);
  const [activeBacktest, setActiveBacktest] = useState(null);
  const [config, setConfig] = useState({
    ticker: 'BTC',
    look_back_period: 6,
    hold_period: 6,
    transaction_cost: 1,
    count_common_threshold: 5,
    stop_loss: null,
    stop_gain: null,
    api_key: '',
    start_date: '2025-02-01',
    num_days: 30
  });
  const [showConfig, setShowConfig] = useState(true);

  // Date validation function
  const validateDates = (startDate, numDays) => {
    const start = new Date(startDate);
    const end = new Date(start);
    end.setDate(start.getDate() + numDays);
    const today = new Date();
    today.setUTCHours(23, 59, 59, 999);
    
    return end <= today;
  };

  // WebSocket connection for real-time updates
  useEffect(() => {
    if (activeBacktest && (activeBacktest.status === 'initializing' || activeBacktest.status === 'running')) {
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsHost = window.location.host;
      const wsUrl = `${wsProtocol}//${wsHost}/ws/${activeBacktest.backtest_id}`;
      console.log('Connecting to WebSocket:', wsUrl);
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        console.log('WebSocket connected');
      };
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('WebSocket update received:', data);
        setActiveBacktest(data);
        
        // Update backtests list
        setBacktests(prev => prev.map(bt => 
          bt.backtest_id === data.backtest_id ? data : bt
        ));
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        toast.error('Connection error');
      };
      
      ws.onclose = () => {
        console.log('WebSocket disconnected');
      };

      return () => {
        ws.close();
      };
    }
  }, [activeBacktest?.backtest_id, activeBacktest?.status]);

  // Polling fallback for status updates
  useEffect(() => {
    if (activeBacktest && (activeBacktest.status === 'initializing' || activeBacktest.status === 'running')) {
      const interval = setInterval(async () => {
        try {
          const response = await axios.get(`${API_URL}/backtest/${activeBacktest.backtest_id}/status`);
          console.log('Polling update:', response.data);
          setActiveBacktest(response.data);
          
          // Update backtests list
          setBacktests(prev => prev.map(bt => 
            bt.backtest_id === response.data.backtest_id ? response.data : bt
          ));
          
          // Stop polling if completed
          if (response.data.status === 'completed' || response.data.status === 'error' || response.data.status === 'cancelled') {
            clearInterval(interval);
          }
        } catch (error) {
          console.error('Polling error:', error);
        }
      }, 2000); // Poll every 2 seconds

      return () => clearInterval(interval);
    }
  }, [activeBacktest?.backtest_id, activeBacktest?.status]);

  const startBacktest = async () => {
    if (!config.api_key) {
      toast.error('Please enter your SentiChain API key');
      return;
    }

    if (!validateDates(config.start_date, config.num_days)) {
      toast.error('Start date + backtest days cannot exceed today');
      return;
    }

    try {
      console.log('Starting backtest with config:', config);
      const response = await axios.post(`${API_URL}/backtest/start`, {
        start_date: config.start_date + 'T00:00:00',
        num_days: config.num_days,
        agent_config: {
          ticker: config.ticker,
          look_back_period: config.look_back_period,
          hold_period: config.hold_period,
          transaction_cost: config.transaction_cost,
          count_common_threshold: config.count_common_threshold,
          stop_loss: config.stop_loss ? parseFloat(config.stop_loss) : null,
          stop_gain: config.stop_gain ? parseFloat(config.stop_gain) : null,
          api_key: config.api_key
        }
      });

      const newBacktest = response.data;
      console.log('Backtest started:', newBacktest);
      setBacktests([newBacktest, ...backtests]);
      setActiveBacktest(newBacktest);
      setShowConfig(false);
      toast.success('Backtest started! You can start multiple backtests.');
    } catch (error) {
      console.error('Failed to start backtest:', error);
      toast.error('Failed to start backtest: ' + (error.response?.data?.detail || error.message));
    }
  };

  const cancelBacktest = async (backtestId) => {
    try {
      await axios.post(`${API_URL}/backtest/${backtestId}/cancel`);
      toast.success('Backtest cancelled');
    } catch (error) {
      toast.error('Failed to cancel backtest');
    }
  };

  const formatPercentage = (value) => `${(value * 100).toFixed(2)}%`;
  const formatNumber = (value) => value?.toFixed(4) || '0';

  return (
    <div className="min-h-screen bg-black text-white overflow-hidden">
      <Toaster position="top-right" />
      
      <header className="p-4 border-b border-cyan-900/50 bg-black/50 backdrop-blur-sm">
        <div className="container mx-auto">
          <h1 className="text-2xl font-bold flex items-center gap-3">
            <TrendingUp className="text-cyan-400" />
            <span className="bg-gradient-to-r from-cyan-400 to-purple-500 bg-clip-text text-transparent">
              Narrative Agent Marketplace
            </span>
          </h1>
          <p className="text-gray-500 text-sm mt-1">AI-powered narrative trading system</p>
        </div>
      </header>

      <main className="container mx-auto p-4 h-[calc(100vh-80px)] flex flex-col">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 flex-1">
          {/* Configuration Panel */}
          <motion.div 
            className="lg:col-span-1"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
          >
            <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl p-4 shadow-xl border border-cyan-900/30 h-full overflow-y-auto">
              <h2 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Settings className="text-purple-400" size={18} />
                Configuration
              </h2>
              
              <div className="space-y-3">
                <div>
                  <label className="block text-xs font-medium text-gray-400 mb-1">Ticker</label>
                  <select 
                    className="w-full bg-black/50 border border-cyan-900/30 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-cyan-500 focus:border-cyan-500"
                    value={config.ticker}
                    onChange={(e) => setConfig({...config, ticker: e.target.value})}
                  >
                    <option value="BTC">BTC</option>
                  </select>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1">
                      Look-back (hrs)
                    </label>
                    <input
                      type="number"
                      className="w-full bg-black/50 border border-cyan-900/30 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-cyan-500 focus:border-cyan-500"
                      value={config.look_back_period}
                      onChange={(e) => setConfig({...config, look_back_period: parseInt(e.target.value)})}
                      min="1"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1">
                      Hold (hrs)
                    </label>
                    <input
                      type="number"
                      className="w-full bg-black/50 border border-cyan-900/30 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-cyan-500 focus:border-cyan-500"
                      value={config.hold_period}
                      onChange={(e) => setConfig({...config, hold_period: parseInt(e.target.value)})}
                      min="1"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1">
                      TX Cost (bps)
                    </label>
                    <input
                      type="number"
                      className="w-full bg-black/50 border border-cyan-900/30 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-cyan-500 focus:border-cyan-500"
                      value={config.transaction_cost}
                      onChange={(e) => setConfig({...config, transaction_cost: parseInt(e.target.value)})}
                      min="0"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1">
                      Keyword Thresh
                    </label>
                    <input
                      type="number"
                      className="w-full bg-black/50 border border-cyan-900/30 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-cyan-500 focus:border-cyan-500"
                      value={config.count_common_threshold}
                      onChange={(e) => setConfig({...config, count_common_threshold: parseInt(e.target.value)})}
                      min="1"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1">
                      Stop Loss (%)
                    </label>
                    <input
                      type="number"
                      className="w-full bg-black/50 border border-cyan-900/30 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-cyan-500 focus:border-cyan-500"
                      value={config.stop_loss || ''}
                      onChange={(e) => setConfig({...config, stop_loss: e.target.value ? parseFloat(e.target.value) : null})}
                      min="0"
                      step="0.1"
                      placeholder="Optional"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1">
                      Stop Gain (%)
                    </label>
                    <input
                      type="number"
                      className="w-full bg-black/50 border border-cyan-900/30 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-cyan-500 focus:border-cyan-500"
                      value={config.stop_gain || ''}
                      onChange={(e) => setConfig({...config, stop_gain: e.target.value ? parseFloat(e.target.value) : null})}
                      min="0"
                      step="0.1"
                      placeholder="Optional"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-xs font-medium text-gray-400 mb-1">
                    Start Date
                  </label>
                  <input
                    type="date"
                    className="w-full bg-black/50 border border-cyan-900/30 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-cyan-500 focus:border-cyan-500"
                    value={config.start_date}
                    onChange={(e) => {
                      const newDate = e.target.value;
                      if (validateDates(newDate, config.num_days)) {
                        setConfig({...config, start_date: newDate});
                      } else {
                        toast.error('Start date + backtest days cannot exceed today');
                      }
                    }}
                    max={new Date().toISOString().split('T')[0]}
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-gray-400 mb-1">
                    Backtest Days
                  </label>
                  <input
                    type="number"
                    className="w-full bg-black/50 border border-cyan-900/30 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-cyan-500 focus:border-cyan-500"
                    value={config.num_days}
                    onChange={(e) => {
                      const newDays = parseInt(e.target.value);
                      if (validateDates(config.start_date, newDays)) {
                        setConfig({...config, num_days: newDays});
                      } else {
                        toast.error('Start date + backtest days cannot exceed today');
                      }
                    }}
                    min="1"
                    max="365"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-gray-400 mb-1">
                    SentiChain API Key
                  </label>
                  <input
                    type="text"
                    className="w-full bg-black/50 border border-cyan-900/30 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-cyan-500 focus:border-cyan-500"
                    value={config.api_key}
                    onChange={(e) => setConfig({...config, api_key: e.target.value})}
                    placeholder="Enter your API key"
                  />
                </div>

                <button
                  onClick={startBacktest}
                  className="w-full bg-gradient-to-r from-cyan-500 to-purple-600 hover:from-cyan-600 hover:to-purple-700 rounded-lg px-4 py-2 font-semibold text-sm transition-all transform hover:scale-[1.02] flex items-center justify-center gap-2 shadow-lg shadow-cyan-500/25"
                >
                  <Play size={16} />
                  Start Backtest
                </button>
              </div>
            </div>
          </motion.div>

          {/* Results Panel */}
          <motion.div 
            className="lg:col-span-3"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
          >
            {activeBacktest ? (
              <div className="grid grid-rows-[auto,1fr,auto] gap-4 h-full relative">
                <button
                  onClick={() => setActiveBacktest(null)}
                  className="absolute top-0 right-0 text-xs text-gray-500 hover:text-gray-300 z-10"
                >
                  Clear View
                </button>
                {/* Status and Activity Row */}
                <div className="grid grid-cols-2 gap-4">
                  {/* Status Card */}
                  <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl p-4 shadow-xl border border-cyan-900/30">
                    <div className="flex items-center justify-between mb-3">
                      <h2 className="text-lg font-semibold">Progress</h2>
                      {activeBacktest.status === 'running' && (
                        <button
                          onClick={() => cancelBacktest(activeBacktest.backtest_id)}
                          className="text-red-400 hover:text-red-300 flex items-center gap-1 text-sm"
                        >
                          <Pause size={16} />
                          Cancel
                        </button>
                      )}
                    </div>
                    
                    <div className="mb-3">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-gray-400 text-sm">Progress</span>
                        <span className="font-mono text-sm">
                          {activeBacktest.current_day}/{activeBacktest.total_days}
                        </span>
                      </div>
                      <div className="w-full bg-black/50 rounded-full h-2">
                        <motion.div 
                          className="bg-gradient-to-r from-cyan-500 to-purple-600 h-2 rounded-full"
                          initial={{ width: 0 }}
                          animate={{ width: `${activeBacktest.progress * 100}%` }}
                          transition={{ ease: "easeOut" }}
                        />
                      </div>
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        {activeBacktest.status === 'running' && (
                          <div className="flex items-center gap-2 text-cyan-400 text-sm">
                            <div className="animate-spin rounded-full h-3 w-3 border-2 border-cyan-400 border-t-transparent" />
                            Running...
                          </div>
                        )}
                        {activeBacktest.status === 'completed' && (
                          <div className="flex items-center gap-2 text-green-400 text-sm">
                            <CheckCircle size={16} />
                            Completed
                          </div>
                        )}
                        {activeBacktest.status === 'error' && (
                          <div className="flex items-center gap-2 text-red-400 text-sm">
                            <AlertCircle size={16} />
                            Error
                          </div>
                        )}
                      </div>
                      {activeBacktest.timestamp && (
                        <span className="text-xs text-gray-500">
                          {new Date(activeBacktest.timestamp).toLocaleDateString()}
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Real-time Activity Log */}
                  <div className={`bg-gray-900/50 backdrop-blur-sm rounded-xl p-4 shadow-xl border ${activeBacktest.status === 'running' ? 'border-purple-900/30' : 'border-gray-800/30'}`}>
                    <h2 className="text-lg font-semibold mb-3 flex items-center gap-2">
                      <TrendingUp className={`${activeBacktest.status === 'running' ? 'text-purple-400' : 'text-gray-600'}`} size={18} />
                      Live Activity
                    </h2>
                    {activeBacktest.status === 'running' ? (
                      <div className="space-y-2 text-sm">
                        <div className="flex items-start gap-2">
                          <div className="w-2 h-2 bg-cyan-400 rounded-full mt-1 animate-pulse"></div>
                          <div>
                            <p className="text-gray-300">
                              Analyzing {activeBacktest.config?.agent_config?.ticker || 'asset'}...
                            </p>
                            <p className="text-gray-500 text-xs">
                              Day {activeBacktest.current_day}: {activeBacktest.config?.agent_config?.look_back_period || 0}h lookback
                            </p>
                          </div>
                        </div>
                        {activeBacktest.metrics && activeBacktest.metrics.total_positions > 0 && (
                          <div className="flex items-start gap-2">
                            <div className="w-2 h-2 bg-green-400 rounded-full mt-1"></div>
                            <div>
                              <p className="text-gray-300">
                                {activeBacktest.metrics.total_positions} position{activeBacktest.metrics.total_positions !== 1 ? 's' : ''}
                              </p>
                              <p className="text-gray-500 text-xs">
                                Return: {formatPercentage(activeBacktest.metrics.total_return)}
                              </p>
                            </div>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="space-y-2 text-sm">
                        <p className="text-gray-500">
                          {activeBacktest.status === 'completed' ? 'Backtest completed' :
                           activeBacktest.status === 'error' ? 'Backtest failed' :
                           activeBacktest.status === 'cancelled' ? 'Backtest cancelled' :
                           'Waiting to start...'}
                        </p>
                        {activeBacktest.status === 'completed' && activeBacktest.metrics && (
                          <p className="text-gray-400 text-xs">
                            Total: {activeBacktest.metrics.total_positions} positions, 
                            {' '}Return: {formatPercentage(activeBacktest.metrics.total_return)}
                          </p>
                        )}
                      </div>
                    )}
                  </div>
                </div>

                {/* Performance Chart */}
                {activeBacktest.status === 'running' || activeBacktest.performance_data ? (
                  <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl p-4 shadow-xl border border-cyan-900/30 h-full">
                    <h2 className="text-lg font-semibold mb-3">Performance</h2>
                    <div className="h-[calc(100%-3rem)]">
                    {activeBacktest.performance_data && activeBacktest.performance_data.length > 0 ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={activeBacktest.performance_data}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                          <XAxis 
                            dataKey="position" 
                            stroke="#64748b"
                            tick={{ fontSize: 12 }}
                          />
                          <YAxis 
                            stroke="#64748b"
                            tickFormatter={formatPercentage}
                            tick={{ fontSize: 12 }}
                          />
                          <Tooltip 
                            formatter={(value) => formatPercentage(value)}
                            contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #0891b2', borderRadius: '8px' }}
                          />
                          <Legend wrapperStyle={{ fontSize: '12px' }} />
                          <Line 
                            type="monotone" 
                            dataKey="cum_return" 
                            stroke="#22d3ee" 
                            name="Cumulative Return"
                            strokeWidth={2}
                            dot={false}
                          />
                          <Line 
                            type="monotone" 
                            dataKey="max_drawdown" 
                            stroke="#f43f5e" 
                            name="Max Drawdown"
                            strokeWidth={2}
                            dot={false}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="h-full flex flex-col items-center justify-center text-gray-500">
                        <div className="mb-4">
                          <div className="animate-pulse flex space-x-4">
                            <div className="h-1 bg-gray-800 rounded w-48"></div>
                          </div>
                        </div>
                        <p className="text-sm">Waiting for positions...</p>
                      </div>
                    )}
                    </div>
                  </div>
                ) : null}

                {/* Live Metrics Grid */}
                {(activeBacktest.status === 'running' || activeBacktest.metrics) && (
                  <div className="grid grid-cols-4 md:grid-cols-7 gap-2">
                    <MetricCard
                      title="Total Return"
                      value={activeBacktest.metrics ? formatPercentage(activeBacktest.metrics.total_return) : '0.00%'}
                      icon={activeBacktest.metrics?.total_return >= 0 ? TrendingUp : TrendingDown}
                      color={activeBacktest.metrics?.total_return >= 0 ? 'cyan' : 'red'}
                      isLoading={!activeBacktest.metrics}
                    />
                    <MetricCard
                      title="Annual Return"
                      value={activeBacktest.metrics ? formatPercentage(activeBacktest.metrics.annualized_return) : '0.00%'}
                      icon={activeBacktest.metrics?.annualized_return >= 0 ? TrendingUp : TrendingDown}
                      color={activeBacktest.metrics?.annualized_return >= 0 ? 'green' : 'red'}
                      isLoading={!activeBacktest.metrics}
                    />
                    <MetricCard
                      title="Win Rate"
                      value={activeBacktest.metrics ? formatPercentage(activeBacktest.metrics.win_rate) : '0.00%'}
                      icon={TrendingUp}
                      color="purple"
                      isLoading={!activeBacktest.metrics}
                    />
                    <MetricCard
                      title="Volatility"
                      value={activeBacktest.metrics ? formatPercentage(activeBacktest.metrics.volatility) : '0.00%'}
                      icon={AlertCircle}
                      color="yellow"
                      isLoading={!activeBacktest.metrics}
                    />
                    <MetricCard
                      title="Sharpe"
                      value={activeBacktest.metrics ? formatNumber(activeBacktest.metrics.sharpe_ratio) : '0.00'}
                      icon={TrendingUp}
                      color="cyan"
                      isLoading={!activeBacktest.metrics}
                    />
                    <MetricCard
                      title="Max DD"
                      value={activeBacktest.metrics ? formatPercentage(activeBacktest.metrics.max_drawdown) : '0.00%'}
                      icon={TrendingDown}
                      color="red"
                      isLoading={!activeBacktest.metrics}
                    />
                    <MetricCard
                      title="Positions"
                      value={activeBacktest.metrics ? activeBacktest.metrics.total_positions : '0'}
                      icon={Settings}
                      color="blue"
                      isLoading={!activeBacktest.metrics}
                    />
                  </div>
                )}
              </div>
            ) : (
              <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl p-12 shadow-xl text-center h-full flex items-center justify-center border border-cyan-900/30">
                <div>
                  <TrendingUp size={64} className="mx-auto text-cyan-900/50 mb-4" />
                  <h3 className="text-xl font-semibold mb-2">No Active Backtest</h3>
                  <p className="text-gray-500">Configure your agent and start a backtest</p>
                </div>
              </div>
            )}
          </motion.div>
        </div>

        {/* Recent Backtests - Compact Table */}
        {backtests.length > 0 && (
          <motion.div 
            className="mt-4 mb-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl shadow-xl overflow-hidden border border-cyan-900/30">
              <div className="px-4 py-2 bg-black/50 flex justify-between items-center">
                <h2 className="text-lg font-semibold">All Backtests ({backtests.length})</h2>
                <span className="text-xs text-gray-500">Click "View" to display results</span>
              </div>
              <div className="overflow-y-auto max-h-40 relative">
                <table className="w-full text-sm">
                  <thead className="bg-black/30 text-xs uppercase sticky top-0 z-10">
                    <tr>
                      <th className="px-4 py-2 text-left text-gray-400 bg-black/50">Ticker</th>
                      <th className="px-4 py-2 text-left text-gray-400 bg-black/50">Config</th>
                      <th className="px-4 py-2 text-left text-gray-400 bg-black/50">Status</th>
                      <th className="px-4 py-2 text-left text-gray-400 bg-black/50">Progress</th>
                      <th className="px-4 py-2 text-left text-gray-400 bg-black/50">Return</th>
                      <th className="px-4 py-2 text-left text-gray-400 bg-black/50">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-800/50">
                    {backtests.map((backtest) => (
                      <tr key={backtest.backtest_id} className={`hover:bg-gray-800/30 transition-colors ${activeBacktest?.backtest_id === backtest.backtest_id ? 'bg-cyan-900/20' : ''}`}>
                        <td className="px-4 py-2 whitespace-nowrap">
                          {backtest.config?.agent_config?.ticker || 'N/A'}
                          {activeBacktest?.backtest_id === backtest.backtest_id && (
                            <span className="ml-2 text-xs text-cyan-400">(viewing)</span>
                          )}
                        </td>
                        <td className="px-4 py-2 text-xs text-gray-400">
                          <div className="space-y-0.5">
                            <div>LB:{backtest.config?.agent_config?.look_back_period}h H:{backtest.config?.agent_config?.hold_period}h</div>
                            <div>TX:{backtest.config?.agent_config?.transaction_cost}bps</div>
                            {(backtest.config?.agent_config?.stop_loss || backtest.config?.agent_config?.stop_gain) && (
                              <div>
                                {backtest.config?.agent_config?.stop_loss && `SL:${backtest.config.agent_config.stop_loss}%`}
                                {backtest.config?.agent_config?.stop_loss && backtest.config?.agent_config?.stop_gain && ' '}
                                {backtest.config?.agent_config?.stop_gain && `SG:${backtest.config.agent_config.stop_gain}%`}
                              </div>
                            )}
                          </div>
                        </td>
                        <td className="px-4 py-2 whitespace-nowrap">
                          <span className={`px-2 py-0.5 text-xs rounded-full ${
                            backtest.status === 'completed' ? 'bg-green-900/50 text-green-400' :
                            backtest.status === 'running' ? 'bg-cyan-900/50 text-cyan-400' :
                            backtest.status === 'error' ? 'bg-red-900/50 text-red-400' :
                            'bg-gray-800 text-gray-400'
                          }`}>
                            {backtest.status}
                          </span>
                        </td>
                        <td className="px-4 py-2 whitespace-nowrap text-xs">
                          {formatPercentage(backtest.progress)}
                        </td>
                        <td className="px-4 py-2 whitespace-nowrap text-xs">
                          {backtest.metrics ? formatPercentage(backtest.metrics.total_return) : '-'}
                        </td>
                        <td className="px-4 py-2 whitespace-nowrap">
                          <button
                            onClick={() => setActiveBacktest(backtest)}
                            className="text-cyan-400 hover:text-cyan-300 text-xs"
                          >
                            View
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </motion.div>
        )}
      </main>
    </div>
  );
}

function MetricCard({ title, value, icon: Icon, color, isLoading }) {
  const colorClasses = {
    cyan: 'text-cyan-400 bg-cyan-900/20 border-cyan-900/50',
    red: 'text-red-400 bg-red-900/20 border-red-900/50',
    purple: 'text-purple-400 bg-purple-900/20 border-purple-900/50',
    yellow: 'text-yellow-400 bg-yellow-900/20 border-yellow-900/50',
    green: 'text-green-400 bg-green-900/20 border-green-900/50',
    blue: 'text-blue-400 bg-blue-900/20 border-blue-900/50'
  };

  return (
    <motion.div 
      className={`bg-gray-900/50 backdrop-blur-sm rounded-lg p-3 shadow-lg border ${colorClasses[color]}`}
      whileHover={{ scale: 1.02 }}
      transition={{ type: "spring", stiffness: 300 }}
    >
      <div className="flex items-center justify-between mb-1">
        <span className="text-gray-400 text-xs">{title}</span>
        <Icon size={14} className={colorClasses[color].split(' ')[0]} />
      </div>
      <div className="text-lg font-bold">
        {isLoading ? (
          <div className="animate-pulse">
            <div className="h-5 bg-gray-800 rounded w-16"></div>
          </div>
        ) : (
          value
        )}
      </div>
    </motion.div>
  );
}

export default App; 