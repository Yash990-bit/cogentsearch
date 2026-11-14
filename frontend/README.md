# CogentSearch React Frontend Integration

This module provides a production-grade React client for the **CogentSearch Multimodal Agentic Document Search & Intelligence** system.

---

### Features
1. **Real-Time Agent Thought Streaming (SSE)**: Connects to `/api/agent/stream` to visualize the LangGraph self-correcting state machine live:
   - 🔍 Query Rewriter (`gpt-4o-mini`)
   - ⚡ Hybrid Dual Retrieval (`Qdrant 1536d Cosine` + `BM25 Okapi` with RRF $k=60$)
   - 🎯 Neural Reranking (`ms-marco-MiniLM-L-6-v2` cross-encoder)
   - ⚖️ Evidence Grader (Self-correction & hallucination guards)
   - 📝 Structured Synthesis (Pydantic schema citations)
2. **Visual Provenance (Cropped Bounding Boxes)**: Renders cropped table, figure, and chart images side-by-side with citations.
3. **Audio Transcripts**: Direct support for Whisper ASR JSON and audio meeting transcripts with timestamp ranges.
4. **Dark Glassmorphic UI**: Premium obsidian theme with glowing violet neon badges (`#a855f7`).

---

### Quick Start

#### 1. Start the FastAPI Backend
```bash
# In the project root:
source .venv/bin/activate
uvicorn api:app --reload --port 8000
```
Interactive Swagger API documentation will be available at:
👉 **`http://localhost:8000/docs`**

#### 2. Plug `CogentSearchChat.tsx` into any React App
```tsx
import React from 'react';
import { CogentSearchChat } from './CogentSearchChat';

export function App() {
  return <CogentSearchChat />;
}
```
