# Uni-OS 🎓
### Your AI-Powered Academic Lifestyle Companion for University of Waterloo

Uni-OS is a smart AI assistant that understands context and answers questions about courses, exams, food, events, campus locations, and more — pulling **live data from the UWaterloo OpenData API** using Gemini's contextual reasoning.

---

## ✨ Features

- 🧠 **AI-First Routing** — Gemini reads a tool catalog and picks the right API based on *meaning*, not keyword matching
- 📚 **Live Course & Subject Data** — Real-time info from UWaterloo's OpenData API
- 🏛️ **Faculty & Organization Data** — Browse subjects by faculty (Math, Engineering, Arts, etc.)
- 📅 **Class Schedules & Exams** — Section times, rooms, instructors, and exam dates
- 🍔 **Food & Campus Info** — Outlets, locations, events, holidays
- 💾 **Student Memory** — Remembers your name, major, and milestones across conversations
- 🔍 **System Trace** — Transparent AI reasoning shown in every response

---

## 🚀 Quick Start (For Teammates)

### 1. Clone the repo
```bash
git clone https://github.com/awosam/UniOS-Hackathon.git
cd UniOS-Hackathon/uni_os
```

### 2. Run the setup script
```bash
bash setup.sh
```
This installs all Python and Node.js dependencies and creates your `.env` file.

### 3. Fill in your API keys
Open `.env` and add your keys:
```
GOOGLE_API_KEY="your-google-api-key"
GOOGLE_CLOUD_QUOTA_PROJECT=your-gcp-project-id
WATERLOO_API_KEY=your-waterloo-api-key
```

| Key | Where to get it |
|-----|----------------|
| `GOOGLE_API_KEY` | [Google Cloud Console](https://console.cloud.google.com/apis/credentials) |
| `GOOGLE_CLOUD_QUOTA_PROJECT` | Your GCP Project ID |
| `WATERLOO_API_KEY` | [UWaterloo OpenData](https://openapi.data.uwaterloo.ca) (free) |

### 4. Authenticate with Google Cloud
```bash
gcloud auth application-default login
```
> If you don't have `gcloud`, the setup script will tell you how to install it.

### 5. Run the app

**Terminal 1 — Backend:**
```bash
python3 -m backend.main
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```

Then open **http://localhost:3000** 🎉

---

## 🗂️ Project Structure

```
uni_os/
├── backend/
│   ├── agents/
│   │   ├── chat_agent.py      # AI-First contextual router (main brain)
│   │   ├── memory.py          # Student memory engine
│   │   ├── pathfinder.py      # LangGraph academic roadmap generator
│   │   └── policy_decoder.py  # RAG for UW policy PDFs
│   ├── tools/
│   │   └── waterloo_api.py    # UWaterloo OpenData API client + TOOL_CATALOG
│   ├── integrations/
│   │   ├── canvas.py          # Canvas LMS integration
│   │   └── calendar.py        # Study schedule generator
│   └── main.py                # FastAPI server entry point
├── frontend/
│   └── src/
│       ├── App.jsx            # Main React UI
│       ├── api.js             # Backend API client
│       └── index.css          # Styling
├── .env.example               # ← Copy this to .env and fill in keys
├── requirements.txt           # Python dependencies
├── setup.sh                   # One-command setup script
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| AI/LLM | Google Gemini (via Vertex AI) |
| Backend | FastAPI + Python |
| AI Framework | LangChain + LangGraph |
| Frontend | React + Vite |
| Data | UWaterloo OpenData API v3 |
| Memory | JSON-based persistent context |
