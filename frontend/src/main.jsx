import React from 'react'
import ReactDOM from 'react-dom/client' // React 18 concurrent renderer
import App from './App.jsx' // Main Application component
import './index.css' // Global styles and design tokens

// Create the root mount point for the React application
ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
