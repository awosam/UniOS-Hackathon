import axios from 'axios'; // HTTP client for API requests

// Create a pre-configured axios instance with a base URL
const api = axios.create({
  baseURL: 'http://localhost:8000', // Point directly to the backend
});

// --- API Service Endpoints ---

// Fetches upcoming student assignments from Canvas LMS
export const getAssignments = () => api.get('/assignments');

// Generates a strategic academic roadmap using AI (LangGraph)
export const generatePlan = (goal, record) => api.post('/generate-plan', { goal, student_record: record });

// Uploads university policy PDFs to the RAG system
export const ingestPolicy = (file) => {
  const formData = new FormData();
  formData.append('file', file);
  return api.post('/ingest-policy', formData);
};

// Drafts professional documents (e.g., extension requests) using 'Admin Concierge'
export const draftDocument = (type, context, query) => api.post('/draft-document', { doc_type: type, student_context: context, policy_query: query });

// Analyzes input to determine the best empathetic AI persona for the student
export const analyzePersona = (input, event) => api.post('/analyze-persona', { user_input: input, academic_event: event });

// Fetches live 'Campus Pulse' vibes and occupancy data
export const getPeerPulse = () => api.get('/peer-pulse');

// Retrieves long-term student memory context (milestones/struggles)
export const getContext = () => api.get('/personal-context');

// Main conversational endpoint
export const sendMessage = (message) => api.post(`/chat?message=${encodeURIComponent(message)}`);

// DEMO: Resets the student memory for a fresh start
export const resetMemory = () => api.post('/reset-memory');

export default api;
