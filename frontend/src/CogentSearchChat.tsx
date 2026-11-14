import React, { useState, useEffect, useRef } from "react";

/**
 * CogentSearch Enterprise React Client
 * Directly integrates with FastAPI Server-Sent Events (SSE) streaming endpoint:
 * - Real-time LangGraph Agent thought chain (Rewrite -> Hybrid Search -> Rerank -> Grade -> Answer)
 * - Dynamic step progress pills & badges
 * - Strict source citation cards
 * - Multi-modal visual bounding box image crops (tables, charts, figures)
 */

interface AgentStep {
  step: string;
  title: string;
  status: "pending" | "running" | "completed";
  detail?: string;
}

interface Citation {
  source_file: string;
  page_number: number;
  modality_type: string;
  bounding_box?: {
    x0: number;
    y0: number;
    x1: number;
    y1: number;
  };
}

interface VisualCrop {
  source_file: string;
  page_number: number;
  modality: string;
  image_base64: string;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  steps?: AgentStep[];
  citations?: Citation[];
  crops?: VisualCrop[];
  timestamp: string;
}

const API_BASE = "http://localhost:8000";

export const CogentSearchChat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome-1",
      role: "assistant",
      content:
        "**CogentSearch Autonomous Agent Online.** I parse, verify, and reason across PDFs, complex tables, trend charts, slide presentations (`.pptx`), and Whisper/ASR audio meeting transcripts (`.json`). Ask a question or preload our enterprise benchmark data.",
      timestamp: new Date().toLocaleTimeString(),
    },
  ]);
  const [inputQuery, setInputQuery] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeSteps, setActiveSteps] = useState<AgentStep[]>([]);
  const [docCount, setDocCount] = useState<number>(0);
  const [chunkCount, setChunkCount] = useState<number>(0);
  const chatBottomRef = useRef<HTMLDivElement>(null);

  // Fetch vector DB telemetry on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then((res) => res.json())
      .then((data) => {
        setDocCount(data.indexed_documents_count || 0);
        setChunkCount(data.total_chunks || 0);
      })
      .catch((err) => console.log("FastAPI backend offline:", err));
  }, []);

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, activeSteps]);

  // Handle Preload Enterprise Corpus
  const handlePreload = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/preload`, { method: "POST" });
      const data = await res.json();
      setDocCount(data.total_documents);
      setChunkCount(data.total_chunks);
      alert(`Preloaded ${data.total_documents} documents (${data.total_chunks} chunks)!`);
    } catch (err) {
      alert("Failed connecting to FastAPI backend at " + API_BASE);
    }
  };

  // Execute Agent Query with SSE Streaming
  const handleSend = async (queryText?: string) => {
    const textToSend = queryText || inputQuery;
    if (!textToSend.trim() || isStreaming) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content: textToSend,
      timestamp: new Date().toLocaleTimeString(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputQuery("");
    setIsStreaming(true);

    const initialSteps: AgentStep[] = [
      { step: "rewrite", title: "Query Rewriter (gpt-4o-mini)", status: "pending" },
      { step: "hybrid_search", title: "Dual Retrieval (Qdrant + BM25)", status: "pending" },
      { step: "rerank", title: "Cross-Encoder (ms-marco-MiniLM)", status: "pending" },
      { step: "grade", title: "Evidence Grader & Self-Correction", status: "pending" },
      { step: "generate", title: "Strict Grounded Answer Synthesis", status: "pending" },
    ];
    setActiveSteps(initialSteps);

    try {
      const response = await fetch(`${API_BASE}/api/agent/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: textToSend }),
      });

      if (!response.body) throw new Error("No readable stream from API");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      let answerText = "";
      let citationsList: Citation[] = [];
      let cropsList: VisualCrop[] = [];

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const payload = JSON.parse(line.replace("data: ", ""));
              if (payload.event === "step_start") {
                setActiveSteps((prev) =>
                  prev.map((s) =>
                    s.step === payload.step
                      ? { ...s, status: "running", detail: payload.detail }
                      : s
                  )
                );
              } else if (payload.event === "step_complete") {
                setActiveSteps((prev) =>
                  prev.map((s) =>
                    s.step === payload.step ? { ...s, status: "completed" } : s
                  )
                );
              } else if (payload.event === "answer") {
                answerText = payload.text;
              } else if (payload.event === "citations") {
                citationsList = payload.citations;
              } else if (payload.event === "visual_crops") {
                cropsList = payload.crops;
              }
            } catch (e) {
              // Ignore partial stream fragments
            }
          }
        }
      }

      // Add final assistant message
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: answerText,
          steps: activeSteps,
          citations: citationsList,
          crops: cropsList,
          timestamp: new Date().toLocaleTimeString(),
        },
      ]);
    } catch (err) {
      console.error("Stream error:", err);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          role: "assistant",
          content:
            "⚠️ Error connecting to CogentSearch agent backend. Please ensure `uvicorn api:app` is running on port 8000.",
          timestamp: new Date().toLocaleTimeString(),
        },
      ]);
    } finally {
      setIsStreaming(false);
      setActiveSteps([]);
    }
  };

  return (
    <div style={styles.container}>
      {/* Top Header */}
      <header style={styles.header}>
        <div style={styles.logoRow}>
          <div style={styles.iconBadge}>⚡</div>
          <div>
            <h1 style={styles.title}>CogentSearch React</h1>
            <p style={styles.subtitle}>
              Multimodal Agentic RAG | Qdrant + BM25 | LangGraph Loop
            </p>
          </div>
        </div>

        <div style={styles.headerRight}>
          <div style={styles.telemetryCard}>
            <span style={styles.telemetryNum}>{docCount}</span> docs |{" "}
            <span style={styles.telemetryNum}>{chunkCount}</span> chunks
          </div>
          <button style={styles.preloadBtn} onClick={handlePreload}>
            📂 Preload Corpus
          </button>
        </div>
      </header>

      {/* Suggested Quick Verification Pills */}
      <div style={styles.quickPillsRow}>
        <button
          style={styles.pillBtn}
          onClick={() =>
            handleSend(
              "What are the financial metrics and revenue figures in the quarterly report table?"
            )
          }
        >
          📊 Quarterly Table
        </button>
        <button
          style={styles.pillBtn}
          onClick={() =>
            handleSend(
              "What are the latency benchmarks and system throughput metrics shown in the chart?"
            )
          }
        >
          📈 Latency Charts
        </button>
        <button
          style={styles.pillBtn}
          onClick={() =>
            handleSend(
              "What key financial growth, ARR, and margins were discussed in the earnings call audio transcript?"
            )
          }
        >
          🎙️ Audio Transcript (Whisper)
        </button>
      </div>

      {/* Main Chat Stream */}
      <main style={styles.chatArea}>
        {messages.map((msg) => (
          <div
            key={msg.id}
            style={{
              ...styles.messageRow,
              justifyContent: msg.role === "user" ? "flex-end" : "flex-start",
            }}
          >
            <div
              style={{
                ...styles.messageBubble,
                ...(msg.role === "user"
                  ? styles.userBubble
                  : styles.assistantBubble),
              }}
            >
              <div style={styles.msgHeader}>
                <strong>{msg.role === "user" ? "You" : "🤖 CogentSearch Agent"}</strong>
                <span style={styles.timeTag}>{msg.timestamp}</span>
              </div>
              <div style={styles.msgContent}>{msg.content}</div>

              {/* Citations Badges */}
              {msg.citations && msg.citations.length > 0 && (
                <div style={styles.citationsSection}>
                  <div style={styles.sectionTitle}>📌 Verified Line-of-Sight Sources:</div>
                  <div style={styles.citationBadgeList}>
                    {msg.citations.map((c, idx) => (
                      <span key={idx} style={styles.citationPill}>
                        📄 {c.source_file} (p. {c.page_number}) [
                        {c.modality_type.toUpperCase()}]
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Visual Bounding-Box Image Crops */}
              {msg.crops && msg.crops.length > 0 && (
                <div style={styles.cropSection}>
                  <div style={styles.sectionTitle}>
                    📷 Visual Evidence (Bounding Box Provenance):
                  </div>
                  <div style={styles.cropGrid}>
                    {msg.crops.map((crop, idx) => (
                      <div key={idx} style={styles.cropCard}>
                        <img
                          src={`data:image/png;base64,${crop.image_base64}`}
                          alt={`Evidence Crop ${idx + 1}`}
                          style={styles.cropImg}
                        />
                        <div style={styles.cropCaption}>
                          {crop.source_file} — Page {crop.page_number} (
                          {crop.modality})
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}

        {/* Live Streaming Agent Steps Accordion */}
        {isStreaming && (
          <div style={styles.agentThinkingCard}>
            <div style={styles.agentThinkingHeader}>
              <span style={styles.pulseDot} />
              <strong>⚡ Agent Reasoning in Real-Time (LangGraph State Machine)</strong>
            </div>
            <div style={styles.stepsList}>
              {activeSteps.map((step, idx) => (
                <div key={idx} style={styles.stepItem}>
                  <span
                    style={{
                      ...styles.stepStatusBadge,
                      background:
                        step.status === "completed"
                          ? "#22c55e"
                          : step.status === "running"
                          ? "#a855f7"
                          : "#475569",
                    }}
                  >
                    {step.status === "completed"
                      ? "✓"
                      : step.status === "running"
                      ? "⟳"
                      : "•"}
                  </span>
                  <div>
                    <span style={styles.stepTitle}>{step.title}</span>
                    {step.detail && <div style={styles.stepDetail}>{step.detail}</div>}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        <div ref={chatBottomRef} />
      </main>

      {/* Input Bar */}
      <footer style={styles.footer}>
        <input
          style={styles.input}
          type="text"
          placeholder="Ask anything across PDFs, charts, slides, or audio transcripts..."
          value={inputQuery}
          disabled={isStreaming}
          onChange={(e) => setInputQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
        />
        <button
          style={styles.sendBtn}
          disabled={isStreaming || !inputQuery.trim()}
          onClick={() => handleSend()}
        >
          {isStreaming ? "Thinking..." : "Send ➔"}
        </button>
      </footer>
    </div>
  );
};

// Modern Glassmorphism Obsidian / Violet Neon CSS in JS
const styles: { [key: string]: React.CSSProperties } = {
  container: {
    display: "flex",
    flexDirection: "column",
    height: "100vh",
    backgroundColor: "#080b11",
    color: "#f1f5f9",
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "16px 24px",
    background: "rgba(15, 23, 42, 0.8)",
    borderBottom: "1px solid rgba(168, 85, 247, 0.25)",
    backdropFilter: "blur(12px)",
  },
  logoRow: { display: "flex", alignItems: "center", gap: "12px" },
  iconBadge: {
    width: "38px",
    height: "38px",
    borderRadius: "10px",
    background: "linear-gradient(135deg, #a855f7 0%, #7c3aed 100%)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: "1.3rem",
    boxShadow: "0 0 16px rgba(168, 85, 247, 0.6)",
  },
  title: { margin: 0, fontSize: "1.35rem", fontWeight: 800, color: "#ffffff" },
  subtitle: { margin: "2px 0 0 0", fontSize: "0.78rem", color: "#94a3b8" },
  headerRight: { display: "flex", alignItems: "center", gap: "12px" },
  telemetryCard: {
    fontSize: "0.82rem",
    color: "#cbd5e1",
    background: "rgba(30, 41, 59, 0.6)",
    padding: "6px 12px",
    borderRadius: "8px",
    border: "1px solid rgba(255, 255, 255, 0.1)",
  },
  telemetryNum: { color: "#c084fc", fontWeight: 700 },
  preloadBtn: {
    background: "linear-gradient(135deg, #a855f7 0%, #7c3aed 100%)",
    color: "#ffffff",
    border: "1px solid rgba(216, 180, 254, 0.5)",
    borderRadius: "8px",
    padding: "7px 14px",
    fontSize: "0.82rem",
    fontWeight: 700,
    cursor: "pointer",
    boxShadow: "0 0 16px rgba(168, 85, 247, 0.4)",
  },
  quickPillsRow: {
    display: "flex",
    gap: "10px",
    padding: "10px 24px",
    background: "rgba(15, 23, 42, 0.4)",
    borderBottom: "1px solid rgba(255, 255, 255, 0.05)",
  },
  pillBtn: {
    background: "rgba(30, 41, 59, 0.6)",
    border: "1px solid rgba(168, 85, 247, 0.3)",
    color: "#e2e8f0",
    borderRadius: "20px",
    padding: "5px 12px",
    fontSize: "0.76rem",
    cursor: "pointer",
  },
  chatArea: {
    flex: 1,
    overflowY: "auto",
    padding: "20px 24px",
    display: "flex",
    flexDirection: "column",
    gap: "18px",
  },
  messageRow: { display: "flex", width: "100%" },
  messageBubble: {
    maxWidth: "85%",
    borderRadius: "14px",
    padding: "16px 20px",
    lineHeight: 1.55,
  },
  userBubble: {
    background: "linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%)",
    color: "#ffffff",
    boxShadow: "0 4px 15px rgba(124, 58, 237, 0.3)",
  },
  assistantBubble: {
    background: "rgba(15, 23, 42, 0.75)",
    border: "1px solid rgba(168, 85, 247, 0.25)",
    color: "#f1f5f9",
    boxShadow: "0 4px 20px rgba(0, 0, 0, 0.4)",
  },
  msgHeader: {
    display: "flex",
    justifyContent: "space-between",
    marginBottom: "6px",
    fontSize: "0.82rem",
  },
  timeTag: { fontSize: "0.72rem", opacity: 0.6 },
  msgContent: { fontSize: "0.94rem" },
  citationsSection: { marginTop: "14px", borderTop: "1px solid rgba(255, 255, 255, 0.1)", paddingTop: "10px" },
  sectionTitle: { fontSize: "0.75rem", color: "#c084fc", fontWeight: 700, marginBottom: "6px" },
  citationBadgeList: { display: "flex", flexWrap: "wrap", gap: "6px" },
  citationPill: {
    fontSize: "0.75rem",
    background: "rgba(168, 85, 247, 0.15)",
    border: "1px solid rgba(168, 85, 247, 0.4)",
    color: "#f3e8ff",
    borderRadius: "6px",
    padding: "3px 8px",
    fontFamily: "monospace",
  },
  cropSection: { marginTop: "14px" },
  cropGrid: { display: "flex", flexWrap: "wrap", gap: "10px", marginTop: "6px" },
  cropCard: {
    background: "#090d16",
    border: "1px solid rgba(168, 85, 247, 0.35)",
    borderRadius: "8px",
    overflow: "hidden",
    maxWidth: "400px",
  },
  cropImg: { width: "100%", height: "auto", display: "block" },
  cropCaption: {
    fontSize: "0.7rem",
    padding: "6px 8px",
    background: "#0f172a",
    color: "#94a3b8",
    fontFamily: "monospace",
  },
  agentThinkingCard: {
    background: "rgba(15, 23, 42, 0.9)",
    border: "1px solid #a855f7",
    borderRadius: "12px",
    padding: "14px 18px",
    boxShadow: "0 0 24px rgba(168, 85, 247, 0.35)",
    maxWidth: "85%",
  },
  agentThinkingHeader: { display: "flex", alignItems: "center", gap: "8px", fontSize: "0.85rem", color: "#d8b4fe" },
  pulseDot: {
    width: "8px",
    height: "8px",
    borderRadius: "50%",
    background: "#a855f7",
    boxShadow: "0 0 10px #a855f7",
  },
  stepsList: { marginTop: "12px", display: "flex", flexDirection: "column", gap: "8px" },
  stepItem: { display: "flex", alignItems: "flex-start", gap: "10px" },
  stepStatusBadge: {
    width: "16px",
    height: "16px",
    borderRadius: "50%",
    color: "white",
    fontSize: "0.62rem",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
    marginTop: "2px",
  },
  stepTitle: { fontSize: "0.82rem", fontWeight: 600, color: "#f1f5f9" },
  stepDetail: { fontSize: "0.72rem", color: "#94a3b8", marginTop: "2px" },
  footer: {
    display: "flex",
    gap: "12px",
    padding: "16px 24px",
    background: "rgba(15, 23, 42, 0.8)",
    borderTop: "1px solid rgba(255, 255, 255, 0.1)",
  },
  input: {
    flex: 1,
    background: "rgba(30, 41, 59, 0.7)",
    border: "1px solid rgba(168, 85, 247, 0.3)",
    borderRadius: "10px",
    padding: "12px 16px",
    color: "#ffffff",
    fontSize: "0.92rem",
    outline: "none",
  },
  sendBtn: {
    background: "linear-gradient(135deg, #a855f7 0%, #7c3aed 100%)",
    color: "#ffffff",
    border: "none",
    borderRadius: "10px",
    padding: "12px 24px",
    fontWeight: 700,
    cursor: "pointer",
    boxShadow: "0 0 20px rgba(168, 85, 247, 0.5)",
  },
};

export default CogentSearchChat;
