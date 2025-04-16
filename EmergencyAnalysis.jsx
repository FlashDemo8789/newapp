import React, { useState } from 'react';

/**
 * Emergency Analysis component - bare minimal implementation
 * This component bypasses all React complexities and directly connects to the API
 */
const EmergencyAnalysis = () => {
  const [analysisData, setAnalysisData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const fetchAnalysis = () => {
    setLoading(true);
    setError(null);
    
    console.log('Emergency direct fetch to API');
    
    // Direct fetch without any middleware or proxies
    fetch('http://localhost:5001/api/v1/analysis/sample-123', {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      mode: 'cors'
    })
      .then(response => {
        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        console.log('Data received from emergency fetch:', data);
        setAnalysisData(data);
        
        // Save to localStorage for future use
        try {
          localStorage.setItem('latestAnalysis', JSON.stringify(data));
          console.log('Saved analysis to localStorage');
        } catch (e) {
          console.error('Failed to save to localStorage:', e);
        }
      })
      .catch(err => {
        console.error('Emergency fetch failed:', err);
        setError(err.message);
      })
      .finally(() => {
        setLoading(false);
      });
  };
  
  const clearData = () => {
    setAnalysisData(null);
  };
  
  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h2>Emergency Analysis Retrieval</h2>
      <p>This is a bare-bones implementation that connects directly to the backend API.</p>
      
      <div style={{ marginBottom: '20px' }}>
        <button 
          onClick={fetchAnalysis} 
          disabled={loading}
          style={{ 
            padding: '10px 15px', 
            background: '#4caf50', 
            color: 'white', 
            border: 'none',
            borderRadius: '4px',
            marginRight: '10px',
            cursor: loading ? 'not-allowed' : 'pointer'
          }}
        >
          {loading ? 'Loading...' : 'Direct Fetch Analysis'}
        </button>
        
        <button
          onClick={clearData}
          disabled={!analysisData}
          style={{ 
            padding: '10px 15px', 
            background: '#f44336', 
            color: 'white', 
            border: 'none',
            borderRadius: '4px',
            cursor: !analysisData ? 'not-allowed' : 'pointer'
          }}
        >
          Clear Data
        </button>
      </div>
      
      {error && (
        <div style={{ 
          padding: '10px 15px', 
          backgroundColor: '#ffebee', 
          border: '1px solid #f44336',
          borderRadius: '4px',
          marginBottom: '20px'
        }}>
          <strong>Error:</strong> {error}
        </div>
      )}
      
      {analysisData && (
        <div style={{ 
          padding: '15px', 
          backgroundColor: '#f5f5f5', 
          border: '1px solid #ddd',
          borderRadius: '4px'
        }}>
          <h3>{analysisData.name}</h3>
          <p><strong>ID:</strong> {analysisData.id}</p>
          <p><strong>Industry:</strong> {analysisData.industry || analysisData.sector}</p>
          <p><strong>Monthly Revenue:</strong> ${analysisData.monthly_revenue}</p>
          <p><strong>Monthly Active Users:</strong> {analysisData.monthly_active_users}</p>
          <p><strong>CAMP Score:</strong> {analysisData.camp_score}</p>
          
          <div style={{ marginTop: '20px' }}>
            <details>
              <summary>Full JSON Response</summary>
              <pre style={{ 
                overflowX: 'auto', 
                backgroundColor: '#f8f8f8', 
                padding: '10px',
                maxHeight: '300px'
              }}>
                {JSON.stringify(analysisData, null, 2)}
              </pre>
            </details>
          </div>
        </div>
      )}
    </div>
  );
};

export default EmergencyAnalysis; 