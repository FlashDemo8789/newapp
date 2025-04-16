import React, { useState, useEffect } from 'react';
import ModelServiceClient from './ModelServiceClient';
import './StartupPredictionForm.css';

// Default form values
const defaultFormData = {
  name: '',
  stage: 'seed',
  sector: 'saas',
  monthly_revenue: 50000,
  annual_recurring_revenue: 600000,
  lifetime_value_ltv: 8000,
  gross_margin_percent: 70,
  operating_margin_percent: 15,
  burn_rate: 1.2,
  runway_months: 18,
  cash_on_hand_million: 2.5,
  debt_ratio: 0.1,
  financing_round_count: 1,
  monthly_active_users: 12000
};

const StartupPredictionForm = ({ serverUrl = 'http://localhost:5000' }) => {
  const [formData, setFormData] = useState(defaultFormData);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [serverStatus, setServerStatus] = useState('unknown');
  const [models, setModels] = useState([]);

  // Create client instance
  const modelClient = new ModelServiceClient(serverUrl);

  // Check server status and available models on component mount or when serverUrl changes
  useEffect(() => {
    const checkServer = async () => {
      try {
        await modelClient.healthCheck();
        setServerStatus('online');
        
        // Try to get available models
        const availableModels = await modelClient.listModels();
        setModels(availableModels);
      } catch (err) {
        setServerStatus('offline');
        setError('ModelServer is offline. Please check the server status.');
      }
    };
    
    checkServer();
  }, [serverUrl]);

  // Handle input changes
  const handleChange = (e) => {
    const { name, value } = e.target;
    
    // For numeric fields, convert string to numbers
    if ([
      'monthly_revenue', 'annual_recurring_revenue', 'lifetime_value_ltv',
      'gross_margin_percent', 'operating_margin_percent', 'burn_rate',
      'runway_months', 'cash_on_hand_million', 'debt_ratio',
      'financing_round_count', 'monthly_active_users'
    ].includes(name)) {
      setFormData({
        ...formData,
        [name]: parseFloat(value) || 0
      });
    } else {
      setFormData({
        ...formData,
        [name]: value
      });
    }
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Prepare data to avoid the float() argument dict error
      // Convert all object or complex values to strings
      const cleanedData = {};
      for (const [key, value] of Object.entries(formData)) {
        if (typeof value === 'object' && value !== null) {
          cleanedData[key] = JSON.stringify(value);
        } else {
          cleanedData[key] = value;
        }
      }

      const predictionResult = await modelClient.predictStartupSuccess(cleanedData);
      setResult(predictionResult);
    } catch (err) {
      console.error("Prediction error:", err);
      setError(`Prediction failed: ${err.message || 'Unknown error'}`);
      if (err.response) {
        setError(`Prediction failed: ${err.response.status} - ${err.response.data?.error || err.response.statusText}`);
      }
      
      // Special handling for the known float error
      if (err.message?.includes("float() argument must be a string or a real number, not 'dict'")) {
        setError('The server encountered an error processing a numeric field. This is a known issue with the model server postprocessing function. Please try using SimpleAPI instead.');
      }
    } finally {
      setLoading(false);
    }
  };

  // Try using SimpleAPI as fallback
  const handleUseSimpleApi = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    
    const simpleApiUrl = serverUrl.replace(/:\d+/, ':5001'); // Assume SimpleAPI is on port 5001
    
    try {
      const response = await fetch(`${simpleApiUrl}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });
      
      if (!response.ok) {
        throw new Error(`SimpleAPI responded with status: ${response.status}`);
      }
      
      const predictionResult = await response.json();
      
      // Format the result to match our expected structure
      setResult({
        result: {
          outcome: predictionResult.outcome || 'unknown',
          confidence: predictionResult.confidence || 0,
          success_probability: predictionResult.success_probability || 0
        }
      });
    } catch (err) {
      console.error("SimpleAPI error:", err);
      setError(`SimpleAPI prediction failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="startup-prediction-container">
      <h2>Startup Success Prediction</h2>
      
      {/* Server Status Display */}
      <div className="server-status">
        <p>Server Status: <span className={`status-${serverStatus}`}>{serverStatus}</span></p>
        {models.length > 0 && (
          <p>Available Models: {models.map(model => model.name).join(', ')}</p>
        )}
      </div>

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="prediction-form">
        <div className="form-group">
          <label htmlFor="name">Startup Name</label>
          <input 
            type="text" 
            id="name" 
            name="name" 
            value={formData.name} 
            onChange={handleChange} 
            placeholder="Enter startup name"
          />
        </div>

        <div className="form-group">
          <label htmlFor="stage">Stage</label>
          <select 
            id="stage" 
            name="stage" 
            value={formData.stage} 
            onChange={handleChange}
          >
            <option value="seed">Seed</option>
            <option value="series_a">Series A</option>
            <option value="series_b">Series B</option>
            <option value="series_c">Series C</option>
            <option value="growth">Growth</option>
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="sector">Sector</label>
          <select 
            id="sector" 
            name="sector" 
            value={formData.sector} 
            onChange={handleChange}
          >
            <option value="saas">SaaS</option>
            <option value="fintech">Fintech</option>
            <option value="healthtech">Healthtech</option>
            <option value="ecommerce">E-commerce</option>
            <option value="ai">AI/ML</option>
            <option value="edtech">EdTech</option>
            <option value="consumer">Consumer</option>
            <option value="hardware">Hardware</option>
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="monthly_revenue">Monthly Revenue ($)</label>
          <input 
            type="number" 
            id="monthly_revenue" 
            name="monthly_revenue" 
            value={formData.monthly_revenue} 
            onChange={handleChange} 
            min="0"
          />
        </div>

        <div className="form-group">
          <label htmlFor="annual_recurring_revenue">Annual Recurring Revenue ($)</label>
          <input 
            type="number" 
            id="annual_recurring_revenue" 
            name="annual_recurring_revenue" 
            value={formData.annual_recurring_revenue} 
            onChange={handleChange} 
            min="0"
          />
        </div>

        <div className="form-group">
          <label htmlFor="burn_rate">Burn Rate ($ millions/month)</label>
          <input 
            type="number" 
            id="burn_rate" 
            name="burn_rate" 
            value={formData.burn_rate} 
            onChange={handleChange} 
            step="0.1" 
            min="0"
          />
        </div>

        <div className="form-group">
          <label htmlFor="runway_months">Runway (months)</label>
          <input 
            type="number" 
            id="runway_months" 
            name="runway_months" 
            value={formData.runway_months} 
            onChange={handleChange} 
            min="0"
          />
        </div>

        <div className="form-row">
          <div className="form-group">
            <label htmlFor="gross_margin_percent">Gross Margin (%)</label>
            <input 
              type="number" 
              id="gross_margin_percent" 
              name="gross_margin_percent" 
              value={formData.gross_margin_percent} 
              onChange={handleChange} 
              min="0" 
              max="100"
            />
          </div>

          <div className="form-group">
            <label htmlFor="operating_margin_percent">Operating Margin (%)</label>
            <input 
              type="number" 
              id="operating_margin_percent" 
              name="operating_margin_percent" 
              value={formData.operating_margin_percent} 
              onChange={handleChange} 
              min="-100" 
              max="100"
            />
          </div>
        </div>

        <div className="form-actions">
          <button 
            type="submit" 
            className="predict-button" 
            disabled={loading || serverStatus !== 'online'}
          >
            {loading ? 'Predicting...' : 'Predict Success'}
          </button>
          
          <button 
            type="button" 
            className="simple-api-button" 
            onClick={handleUseSimpleApi}
            disabled={loading}
          >
            Use SimpleAPI Instead
          </button>
        </div>
      </form>

      {/* Error Display */}
      {error && (
        <div className="error-message">
          <p>{error}</p>
        </div>
      )}

      {/* Results Display */}
      {result && (
        <div className="prediction-result">
          <h3>Prediction Result</h3>
          
          {result.result ? (
            <div className="result-container">
              <div className={`outcome ${result.result.outcome === 'pass' ? 'success' : 'failure'}`}>
                <span className="outcome-label">Outcome:</span> 
                <span className="outcome-value">{result.result.outcome === 'pass' ? 'Success' : 'Failure'}</span>
              </div>
              
              <div className="confidence">
                <span>Confidence:</span> {(result.result.confidence * 100).toFixed(1)}%
              </div>
              
              <div className="probability">
                <span>Success Probability:</span> {(result.result.success_probability * 100).toFixed(1)}%
              </div>
            </div>
          ) : (
            <p>Unable to parse prediction result.</p>
          )}
        </div>
      )}
    </div>
  );
};

export default StartupPredictionForm; 