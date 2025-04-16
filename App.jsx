import React, { useState } from 'react';
import StartupPredictionForm from './StartupPredictionForm';
import TestAnalysis from './TestAnalysis';
import EmergencyAnalysis from './EmergencyAnalysis';
import './StartupPredictionForm.css';
import './App.css';

function App() {
  const [serverUrl, setServerUrl] = useState('http://localhost:5001');
  const [currentView, setCurrentView] = useState('analysis'); // Changed to 'analysis' so it starts with TestAnalysis component
  
  const handleServerUrlChange = (e) => {
    setServerUrl(e.target.value);
  };
  
  return (
    <div className="App">
      <header className="App-header">
        <h1>FlashDNA ML Infrastructure</h1>
        <div className="server-config">
          <label htmlFor="server-url">Model Server URL:</label>
          <input 
            type="text" 
            id="server-url" 
            value={serverUrl} 
            onChange={handleServerUrlChange}
            placeholder="http://localhost:5001"
          />
        </div>
        <div className="view-tabs">
          <button 
            className={currentView === 'prediction' ? 'active' : ''} 
            onClick={() => setCurrentView('prediction')}
          >
            Prediction Form
          </button>
          <button 
            className={currentView === 'analysis' ? 'active' : ''} 
            onClick={() => setCurrentView('analysis')}
          >
            Test Analysis
          </button>
          <button 
            className={currentView === 'emergency' ? 'active' : ''} 
            onClick={() => setCurrentView('emergency')}
          >
            Emergency Mode
          </button>
        </div>
      </header>
      
      <main className="App-content">
        {currentView === 'prediction' ? (
          <StartupPredictionForm serverUrl={serverUrl} />
        ) : currentView === 'analysis' ? (
          <TestAnalysis />
        ) : (
          <EmergencyAnalysis />
        )}
      </main>
      
      <footer className="App-footer">
        <p>Powered by FlashDNA ML Infrastructure</p>
      </footer>
    </div>
  );
}

export default App; 