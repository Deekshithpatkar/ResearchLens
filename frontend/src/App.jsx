import React, { useState, useEffect, useRef } from "react";
import { 
  authAPI, 
  papersAPI, 
  analyticsAPI,
  chatsAPI 
} from "./api";
import { 
  BookOpen, 
  MessageSquare, 
  BarChart2, 
  GitCommit, 
  Share2, 
  LogOut, 
  Upload, 
  Trash2, 
  Loader2, 
  AlertCircle, 
  Plus, 
  FileText, 
  ChevronRight,
  Database,
  Search,
  User,
  KeyRound,
  FileCheck,
  Eye,
  EyeOff
} from "lucide-react";

// Self-contained Markdown and Table Parser
function renderMarkdown(text) {
  if (!text) return "";
  const lines = text.split("\n");
  const elements = [];
  let listItems = [];
  let tableRows = [];

  const flushList = (key) => {
    if (listItems.length > 0) {
      elements.push(
        <ul key={`list-${key}`} className="list-disc pl-5 my-3 space-y-1.5 text-gray-300">
          {listItems}
        </ul>
      );
      listItems = [];
    }
  };

  const flushTable = (key) => {
    if (tableRows.length > 0) {
      const headers = tableRows[0];
      const dataRows = tableRows.slice(1);
      elements.push(
        <div key={`table-container-${key}`} className="overflow-x-auto my-4 rounded-xl border border-white/10 shadow-lg">
          <table className="min-w-full divide-y divide-white/10 text-xs">
            <thead className="bg-gray-950/40">
              <tr>
                {headers.map((h, i) => (
                  <th key={i} className="px-4 py-3 text-left font-bold text-gray-200 border-r border-white/5 last:border-r-0">
                    {parseInline(h)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 bg-gray-900/10">
              {dataRows.map((row, ri) => (
                <tr key={ri} className="hover:bg-white/5 transition-all">
                  {row.map((cell, ci) => (
                    <td key={ci} className="px-4 py-3 text-gray-300 border-r border-white/5 last:border-r-0 leading-relaxed">
                      {parseInline(cell)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
      tableRows = [];
    }
  };

  const parseInline = (str) => {
    let parts = str.split(/(\*\*.*?\*\*|`.*?`)/g);
    return parts.map((part, i) => {
      if (part.startsWith("**") && part.endsWith("**")) {
        return <strong key={i} className="font-bold text-white">{part.slice(2, -2)}</strong>;
      }
      if (part.startsWith("`") && part.endsWith("`")) {
        return <code key={i} className="bg-gray-950/60 px-1.5 py-0.5 rounded text-red-300 text-xs font-mono">{part.slice(1, -1)}</code>;
      }
      return part;
    });
  };

  for (let idx = 0; idx < lines.length; idx++) {
    const line = lines[idx].trim();

    const isTable = line.startsWith("|") || (line.includes("|") && (tableRows.length > 0 || line.match(/^[|\s:-]+$/) || (line.match(/\|/g) || []).length >= 2));

    if (isTable) {
      flushList(idx);
      if (line.match(/^[|\s:-]+$/)) {
        continue;
      }
      let cells = line.split("|").map(c => c.trim());
      if (cells[0] === "") cells.shift();
      if (cells[cells.length - 1] === "") cells.pop();
      tableRows.push(cells);
      continue;
    } else {
      flushTable(idx);
    }

    if (line.startsWith("###")) {
      flushList(idx);
      elements.push(<h3 key={idx} className="text-sm font-bold text-blue-400 mt-5 mb-2">{parseInline(line.substring(3).trim())}</h3>);
      continue;
    }
    if (line.startsWith("##")) {
      flushList(idx);
      elements.push(<h2 key={idx} className="text-md font-bold text-white mt-6 mb-3 border-b border-white/5 pb-1">{parseInline(line.substring(2).trim())}</h2>);
      continue;
    }
    if (line.startsWith("#")) {
      flushList(idx);
      elements.push(<h1 key={idx} className="text-lg font-bold text-white mt-6 mb-4">{parseInline(line.substring(1).trim())}</h1>);
      continue;
    }

    if (line.startsWith("* ") || line.startsWith("- ")) {
      listItems.push(<li key={idx}>{parseInline(line.substring(2))}</li>);
      continue;
    } else {
      flushList(idx);
    }

    if (line) {
      elements.push(<p key={idx} className="my-2.5 text-gray-300 leading-relaxed">{parseInline(line)}</p>);
    }
  }

  flushList(lines.length);
  flushTable(lines.length);

  return elements;
}

export default function App() {
  const [token, setToken] = useState(localStorage.getItem("authToken"));
  const [activeTab, setActiveTab] = useState("library");
  
  // Auth Form State
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [authError, setAuthError] = useState("");
  const [authLoading, setAuthLoading] = useState(false);

  // App State
  const [papers, setPapers] = useState({});
  const [loadingPapers, setLoadingPapers] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(null);
  const [uploadError, setUploadError] = useState("");
  const [uploadSuccess, setUploadSuccess] = useState("");

  // RAG Chat State
  const [selectedPaperForChat, setSelectedPaperForChat] = useState("");
  const [chatQuery, setChatQuery] = useState("");
  const [chatLog, setChatLog] = useState([]);
  const [chatLoading, setChatLoading] = useState(false);
  const [ragSessions, setRagSessions] = useState([]);
  const [selectedRagSessionId, setSelectedRagSessionId] = useState("");

  // Analytics State
  const [analyticsQuery, setAnalyticsQuery] = useState("");
  const [selectedPapersForAnalytics, setSelectedPapersForAnalytics] = useState([]);
  const [analyticsChatLog, setAnalyticsChatLog] = useState([]);
  const [analyticsLoading, setAnalyticsLoading] = useState(false);
  const [analyticsSessions, setAnalyticsSessions] = useState([]);
  const [selectedAnalyticsSessionId, setSelectedAnalyticsSessionId] = useState("");

  // Timeline State
  const [timelineData, setTimelineData] = useState(null);
  const [timelineLoading, setTimelineLoading] = useState(false);

  // Clusters State
  const [clusterData, setClusterData] = useState(null);
  const [clusterLoading, setClusterLoading] = useState(false);
  const [clusterType, setClusterType] = useState("hierarchical"); // "cosine" or "hierarchical"

  const chatEndRef = useRef(null);
  const analyticsEndRef = useRef(null);

  // Load papers and sessions on authentication
  useEffect(() => {
    if (token) {
      fetchPapers();
      fetchSessions("rag");
      fetchSessions("analytics");
    }
  }, [token]);

  // Auto-scroll chat logs to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatLog]);

  useEffect(() => {
    analyticsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [analyticsChatLog]);

  const fetchSessions = async (type) => {
    try {
      const data = await chatsAPI.list(type);
      if (type === "rag") {
        setRagSessions(data);
      } else if (type === "analytics") {
        setAnalyticsSessions(data);
      }
    } catch (err) {
      console.error(`Failed to fetch ${type} sessions:`, err);
    }
  };

  const loadSessionMessages = async (sessionId, type) => {
    try {
      const messages = await chatsAPI.getMessages(sessionId);
      const formatted = messages.map(msg => ({
        role: msg.role,
        text: msg.content,
        chunks: msg.chunks || []
      }));
      if (type === "rag") {
        setChatLog(formatted);
        setSelectedRagSessionId(sessionId);
      } else if (type === "analytics") {
        setAnalyticsChatLog(formatted);
        setSelectedAnalyticsSessionId(sessionId);
      }
    } catch (err) {
      console.error("Failed to load messages:", err);
    }
  };

  const handleDeleteSession = async (sessionId, type, e) => {
    if (e) e.stopPropagation();
    if (!confirm("Are you sure you want to delete this chat session?")) return;
    try {
      await chatsAPI.delete(sessionId);
      fetchSessions(type);
      if (type === "rag" && selectedRagSessionId === sessionId) {
        setSelectedRagSessionId("");
        setChatLog([]);
      } else if (type === "analytics" && selectedAnalyticsSessionId === sessionId) {
        setSelectedAnalyticsSessionId("");
        setAnalyticsChatLog([]);
      }
    } catch (err) {
      alert("Failed to delete chat session");
    }
  };

  const handleNewChat = (type) => {
    if (type === "rag") {
      setSelectedRagSessionId("");
      setChatLog([]);
      setChatQuery("");
    } else if (type === "analytics") {
      setSelectedAnalyticsSessionId("");
      setAnalyticsChatLog([]);
      setAnalyticsQuery("");
    }
  };

  const fetchPapers = async () => {
    setLoadingPapers(true);
    try {
      const data = await papersAPI.list();
      setPapers(data.papers || {});
    } catch (err) {
      console.error("Failed to load papers:", err);
    } finally {
      setLoadingPapers(false);
    }
  };

  // Auth Handlers
  const handleAuth = async (e) => {
    e.preventDefault();
    setAuthError("");
    setAuthLoading(true);
    try {
      if (isLogin) {
        const data = await authAPI.login(email, password);
        localStorage.setItem("authToken", data.access_token);
        localStorage.setItem("userEmail", email);
        setToken(data.access_token);
      } else {
        await authAPI.register(email, password);
        // Automatically log the user in after registration
        const data = await authAPI.login(email, password);
        localStorage.setItem("authToken", data.access_token);
        localStorage.setItem("userEmail", email);
        setToken(data.access_token);
      }
    } catch (err) {
      setAuthError(err.response?.data?.detail || "Authentication failed. Please try again.");
    } finally {
      setAuthLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("authToken");
    localStorage.removeItem("userEmail");
    setToken(null);
    setChatLog([]);
    setPapers({});
    setAnalyticsResult(null);
    setTimelineData(null);
    setClusterData(null);
  };

  // File Upload Handler
  const handleFileUpload = async (e) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setUploadError("");
    setUploadSuccess("");
    setUploadProgress(0);

    try {
      const data = await papersAPI.upload(files, (progressEvent) => {
        const percentCompleted = Math.round(
          (progressEvent.loaded * 100) / progressEvent.total
        );
        setUploadProgress(percentCompleted);
      });
      setUploadSuccess(`Successfully uploaded ${files.length} paper(s)!`);
      fetchPapers();
    } catch (err) {
      setUploadError(err.response?.data?.detail || "Failed to upload file(s).");
    } finally {
      setUploadProgress(null);
    }
  };

  // Delete Paper Handler
  const handleDeletePaper = async (paperId) => {
    if (!confirm(`Are you sure you want to delete ${paperId}?`)) return;
    try {
      await papersAPI.delete(paperId);
      fetchPapers();
      if (selectedPaperForChat === paperId) {
        setSelectedPaperForChat("");
      }
    } catch (err) {
      alert("Failed to delete paper");
    }
  };

  // RAG Chat Handler
  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!chatQuery.trim()) return;

    const userMessage = { role: "user", text: chatQuery };
    setChatLog((prev) => [...prev, userMessage]);
    const currentQuery = chatQuery;
    setChatQuery("");
    setChatLoading(true);

    try {
      const data = await papersAPI.query(
        currentQuery,
        selectedPaperForChat || null,
        8,
        selectedRagSessionId || null
      );
      const assistantMessage = {
        role: "assistant",
        text: data.answer,
        chunks: data.chunks || [],
      };
      setChatLog((prev) => [...prev, assistantMessage]);
      if (data.session_id && data.session_id !== selectedRagSessionId) {
        setSelectedRagSessionId(data.session_id);
        fetchSessions("rag");
      }
    } catch (err) {
      const errorMessage = {
        role: "error",
        text: err.response?.data?.detail || "RAG search failed. Try again.",
      };
      setChatLog((prev) => [...prev, errorMessage]);
    } finally {
      setChatLoading(false);
    }
  };

  // Analytics Handler
  const runAnalytics = async (e) => {
    if (e) e.preventDefault();
    if (!analyticsQuery.trim()) return;

    const currentQuery = analyticsQuery;
    const userMessage = { role: "user", text: currentQuery };
    setAnalyticsChatLog((prev) => [...prev, userMessage]);
    setAnalyticsQuery("");
    setAnalyticsLoading(true);

    // Build conversation history prompt
    let finalQuery = "";
    if (analyticsChatLog.length > 0) {
      finalQuery = "We are having an interactive conversation comparing research papers. Follow up on our previous discussion:\n\n";
      analyticsChatLog.forEach(msg => {
        if (msg.role === "user") {
          finalQuery += `Question: ${msg.text}\n`;
        } else if (msg.role === "assistant") {
          finalQuery += `Answer: ${msg.text}\n\n`;
        }
      });
      finalQuery += `New Follow-up Question: ${currentQuery}`;
    } else {
      finalQuery = currentQuery;
    }

    try {
      const data = await analyticsAPI.global(
        finalQuery,
        selectedPapersForAnalytics,
        currentQuery,
        selectedAnalyticsSessionId || null
      );
      const assistantMessage = {
        role: "assistant",
        text: data.answer,
        fields: data.fields_used || [],
        warnings: data.warnings || [],
      };
      setAnalyticsChatLog((prev) => [...prev, assistantMessage]);
      if (data.session_id && data.session_id !== selectedAnalyticsSessionId) {
        setSelectedAnalyticsSessionId(data.session_id);
        fetchSessions("analytics");
      }
    } catch (err) {
      const errorMessage = {
        role: "error",
        text: err.response?.data?.detail || "Analytics query failed. Try again.",
      };
      setAnalyticsChatLog((prev) => [...prev, errorMessage]);
    } finally {
      setAnalyticsLoading(false);
    }
  };

  // Timeline Handler
  const loadTimeline = async () => {
    setTimelineLoading(true);
    try {
      const data = await analyticsAPI.getTimeline();
      setTimelineData(data);
    } catch (err) {
      alert("Failed to load timeline");
    } finally {
      setTimelineLoading(false);
    }
  };

  // Cluster Handler
  const loadClusters = async (type = clusterType) => {
    setClusterLoading(true);
    try {
      const data = type === "cosine" 
        ? await analyticsAPI.getCosineClusters() 
        : await analyticsAPI.getHierarchicalClusters();
      setClusterData(data);
    } catch (err) {
      alert("Failed to load clusters");
    } finally {
      setClusterLoading(false);
    }
  };

  // Switch tab handlers to auto-load if needed
  const handleTabChange = (tab) => {
    setActiveTab(tab);
    if (tab === "timeline") {
      loadTimeline();
    } else if (tab === "clusters") {
      loadClusters(clusterType);
    }
  };

  const handleClusterTypeChange = (type) => {
    setClusterType(type);
    loadClusters(type);
  };

  // Auth Screen
  if (!token) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-darkBg px-4 relative overflow-hidden">
        {/* Decorative background glows */}
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-600/10 rounded-full blur-3xl"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-indigo-600/10 rounded-full blur-3xl"></div>
        
        <div className="glass-panel w-full max-w-md p-8 rounded-2xl shadow-2xl relative z-10">
          <div className="flex flex-col items-center mb-8">
            <div className="h-12 w-12 bg-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/30 mb-3">
              <BookOpen className="h-6 w-6 text-white" />
            </div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
              ResearchLens
            </h1>
            <p className="text-gray-400 text-sm mt-1">Multi-Tenant AI Literature Synthesis</p>
          </div>

          <div className="flex bg-gray-900/50 p-1 rounded-lg mb-6 border border-white/5">
            <button
              onClick={() => { setIsLogin(true); setAuthError(""); }}
              className={`flex-1 py-2 text-sm font-semibold rounded-md transition-all ${
                isLogin ? "bg-blue-600 text-white shadow" : "text-gray-400 hover:text-white"
              }`}
            >
              Log In
            </button>
            <button
              onClick={() => { setIsLogin(false); setAuthError(""); }}
              className={`flex-1 py-2 text-sm font-semibold rounded-md transition-all ${
                !isLogin ? "bg-blue-600 text-white shadow" : "text-gray-400 hover:text-white"
              }`}
            >
              Register
            </button>
          </div>

          <form onSubmit={handleAuth} className="space-y-4">
            <div>
              <label className="block text-xs font-semibold text-gray-400 mb-1.5">EMAIL ADDRESS</label>
              <div className="relative">
                <span className="absolute inset-y-0 left-0 pl-3.5 flex items-center text-gray-400">
                  <User className="h-4 w-4" />
                </span>
                <input
                  type="email"
                  required
                  placeholder="name@university.edu"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="glass-input glass-input-icon w-full"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-semibold text-gray-400 mb-1.5">PASSWORD</label>
              <div className="relative">
                <span className="absolute inset-y-0 left-0 pl-3.5 flex items-center text-gray-400">
                  <KeyRound className="h-4 w-4" />
                </span>
                <input
                  type={showPassword ? "text" : "password"}
                  required
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="glass-input glass-input-icon glass-input-password w-full"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute inset-y-0 right-0 pr-3.5 flex items-center text-gray-400 hover:text-white transition-all cursor-pointer"
                >
                  {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </div>

            {authError && (
              <div className={`p-3 rounded-lg flex items-start gap-2.5 text-xs ${
                authError.includes("successful") 
                  ? "bg-emerald-950/40 border border-emerald-800/40 text-emerald-300"
                  : "bg-red-950/40 border border-red-800/40 text-red-300"
              }`}>
                <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
                <span>{authError}</span>
              </div>
            )}

            <button
              type="submit"
              disabled={authLoading}
              className="w-full py-3 bg-blue-600 hover:bg-blue-500 disabled:bg-blue-800 transition-all font-semibold rounded-lg shadow-lg shadow-blue-500/20 text-sm flex items-center justify-center gap-2 mt-6"
            >
              {authLoading && <Loader2 className="h-4 w-4 animate-spin" />}
              {isLogin ? "Sign In" : "Create Account"}
            </button>
          </form>
        </div>
      </div>
    );
  }

  // Dashboard Main Screen
  return (
    <div className="min-h-screen flex bg-darkBg text-gray-100 font-sans">
      
      {/* SIDEBAR */}
      <aside className="w-72 border-r border-white/5 bg-gray-950/30 flex flex-col shrink-0">
        
        {/* Header Logo */}
        <div className="p-6 border-b border-white/5 flex items-center gap-3">
          <div className="h-9 w-9 bg-blue-600 rounded-lg flex items-center justify-center shadow-lg shadow-blue-500/30">
            <BookOpen className="h-5 w-5 text-white" />
          </div>
          <div>
            <h1 className="font-bold text-md leading-none">ResearchLens</h1>
            <span className="text-[10px] text-gray-400 font-semibold tracking-wider">AI WORKSPACE</span>
          </div>
        </div>

        {/* Upload Widget */}
        <div className="p-5 border-b border-white/5">
          <div className="relative border border-dashed border-white/10 hover:border-blue-500/50 rounded-xl p-4 transition-all text-center cursor-pointer group bg-gray-900/10">
            <input
              type="file"
              multiple
              accept=".pdf"
              onChange={handleFileUpload}
              className="absolute inset-0 opacity-0 cursor-pointer"
              disabled={uploadProgress !== null}
            />
            <div className="flex flex-col items-center gap-2">
              <div className="h-9 w-9 bg-gray-800 rounded-full flex items-center justify-center group-hover:bg-blue-600/10 group-hover:text-blue-400 transition-all">
                {uploadProgress !== null ? (
                  <Loader2 className="h-4.5 w-4.5 animate-spin text-blue-400" />
                ) : (
                  <Upload className="h-4.5 w-4.5 text-gray-400" />
                )}
              </div>
              <div className="text-xs font-semibold">
                {uploadProgress !== null ? `Uploading (${uploadProgress}%)` : "Upload PDF Papers"}
              </div>
              <div className="text-[10px] text-gray-500">Supports PDF up to 45MB</div>
            </div>
          </div>
          {uploadError && <div className="text-[10px] text-red-400 mt-2 text-center">{uploadError}</div>}
          {uploadSuccess && <div className="text-[10px] text-emerald-400 mt-2 text-center">{uploadSuccess}</div>}
        </div>

        {/* Sidebar Nav */}
        <nav className="flex-1 p-4 space-y-1.5">
          <button
            onClick={() => handleTabChange("library")}
            className={`w-full py-2.5 px-4 rounded-lg flex items-center gap-3 text-sm font-semibold transition-all ${
              activeTab === "library" 
                ? "bg-blue-600/10 text-blue-400 border border-blue-500/20" 
                : "text-gray-400 hover:text-white hover:bg-white/5"
            }`}
          >
            <Database className="h-4 w-4" />
            Workspace Library
          </button>
          
          <button
            onClick={() => handleTabChange("chat")}
            className={`w-full py-2.5 px-4 rounded-lg flex items-center gap-3 text-sm font-semibold transition-all ${
              activeTab === "chat" 
                ? "bg-blue-600/10 text-blue-400 border border-blue-500/20" 
                : "text-gray-400 hover:text-white hover:bg-white/5"
            }`}
          >
            <MessageSquare className="h-4 w-4" />
            Semantic RAG Chat
          </button>

          <button
            onClick={() => handleTabChange("analytics")}
            className={`w-full py-2.5 px-4 rounded-lg flex items-center gap-3 text-sm font-semibold transition-all ${
              activeTab === "analytics" 
                ? "bg-blue-600/10 text-blue-400 border border-blue-500/20" 
                : "text-gray-400 hover:text-white hover:bg-white/5"
            }`}
          >
            <BarChart2 className="h-4 w-4" />
            Compare Analytics
          </button>

          <button
            onClick={() => handleTabChange("timeline")}
            className={`w-full py-2.5 px-4 rounded-lg flex items-center gap-3 text-sm font-semibold transition-all ${
              activeTab === "timeline" 
                ? "bg-blue-600/10 text-blue-400 border border-blue-500/20" 
                : "text-gray-400 hover:text-white hover:bg-white/5"
            }`}
          >
            <GitCommit className="h-4 w-4" />
            Research Timeline
          </button>

          <button
            onClick={() => handleTabChange("clusters")}
            className={`w-full py-2.5 px-4 rounded-lg flex items-center gap-3 text-sm font-semibold transition-all ${
              activeTab === "clusters" 
                ? "bg-blue-600/10 text-blue-400 border border-blue-500/20" 
                : "text-gray-400 hover:text-white hover:bg-white/5"
            }`}
          >
            <Share2 className="h-4 w-4" />
            Topic Clustering
          </button>
        </nav>

        {/* Footer User Info */}
        <div className="p-4 border-t border-white/5 flex items-center justify-between bg-gray-950/20">
          <div className="truncate pr-2">
            <p className="text-[10px] font-bold text-gray-500 leading-none">SIGNED IN AS</p>
            <span className="text-xs font-semibold text-gray-300 truncate block mt-1">
              {localStorage.getItem("userEmail") || "User"}
            </span>
          </div>
          <button
            onClick={handleLogout}
            className="h-8 w-8 rounded-lg flex items-center justify-center hover:bg-red-500/10 hover:text-red-400 text-gray-400 transition-all shrink-0"
            title="Log Out"
          >
            <LogOut className="h-4 w-4" />
          </button>
        </div>

      </aside>

      {/* MAIN VIEWPORT */}
      <main className="flex-1 flex flex-col min-w-0 overflow-y-auto">
        
        {/* Workspace Library Tab */}
        {activeTab === "library" && (
          <div className="p-8 max-w-6xl w-full mx-auto space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-xl font-bold">Workspace Library</h2>
                <p className="text-xs text-gray-400 mt-1">Manage and inspect your uploaded PDF research papers.</p>
              </div>
              <button 
                onClick={fetchPapers}
                className="p-2 border border-white/5 rounded-lg bg-gray-900/30 hover:bg-gray-800 transition-all"
                title="Refresh Library"
              >
                <Loader2 className={`h-4 w-4 ${loadingPapers ? "animate-spin" : ""}`} />
              </button>
            </div>

            {loadingPapers ? (
              <div className="flex flex-col items-center justify-center py-20 text-gray-400 gap-3">
                <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
                <span className="text-sm">Fetching files...</span>
              </div>
            ) : Object.keys(papers).length === 0 ? (
              <div className="glass-panel text-center py-24 rounded-2xl flex flex-col items-center gap-3">
                <FileText className="h-10 w-10 text-gray-500" />
                <h3 className="font-semibold text-gray-300">Your library is empty</h3>
                <p className="text-xs text-gray-500 max-w-sm">Upload research papers using the sidebar widget to start comparing and RAG querying.</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
                {Object.entries(papers).map(([paperId, meta]) => (
                  <div key={paperId} className="glass-panel p-5 rounded-xl hover:border-blue-500/30 transition-all flex flex-col justify-between h-44 relative group">
                    <div>
                      <div className="flex items-start justify-between gap-3 mb-2">
                        <h4 className="font-bold text-sm text-gray-200 line-clamp-2 leading-snug">{paperId.replace(/_/g, " ")}</h4>
                        <button
                          onClick={() => handleDeletePaper(paperId)}
                          className="p-1.5 rounded-lg hover:bg-red-500/10 hover:text-red-400 text-gray-500 transition-all opacity-0 group-hover:opacity-100"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                      <p className="text-[10px] text-gray-400 font-semibold tracking-wide truncate mb-1">FILE: {meta.filename}</p>
                    </div>

                    <div className="border-t border-white/5 pt-3 flex items-center justify-between text-[11px] text-gray-400 font-medium">
                      <span>{meta.num_chunks} chunks</span>
                      <span>Uploaded {new Date(meta.uploaded_at).toLocaleDateString()}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Semantic RAG Chat Tab */}
        {activeTab === "chat" && (
          <div className="flex-1 flex gap-6 p-8 w-full max-w-7xl mx-auto h-[calc(100vh-80px)] overflow-hidden">
            {/* RAG Chat Sessions Sidebar */}
            <div className="w-64 shrink-0 flex flex-col glass-panel rounded-2xl p-4 overflow-hidden border border-white/5 bg-gray-900/10">
              <button
                onClick={() => handleNewChat("rag")}
                className="w-full py-2.5 px-4 mb-4 bg-blue-600 hover:bg-blue-500 font-semibold rounded-lg shadow text-xs flex items-center justify-center gap-2 cursor-pointer transition-all shrink-0 text-white"
              >
                <Plus className="h-4 w-4" />
                New Chat
              </button>
              <div className="flex-1 overflow-y-auto space-y-1.5 pr-1">
                <span className="text-[10px] font-bold text-gray-400 block px-2 mb-2 tracking-wider">PREVIOUS CONVERSATIONS</span>
                {ragSessions.length === 0 ? (
                  <div className="text-center text-xs text-gray-500 py-8 italic">No chats saved</div>
                ) : (
                  ragSessions.map((session) => (
                    <div
                      key={session.id}
                      onClick={() => loadSessionMessages(session.id, "rag")}
                      className={`flex items-center justify-between group p-2.5 rounded-lg text-xs cursor-pointer transition-all ${
                        selectedRagSessionId === session.id
                          ? "bg-blue-600/20 text-blue-200 border border-blue-500/30"
                          : "hover:bg-white/5 text-gray-300 border border-transparent"
                      }`}
                    >
                      <span className="truncate pr-2">{session.title}</span>
                      <button
                        onClick={(e) => handleDeleteSession(session.id, "rag", e)}
                        className="opacity-0 group-hover:opacity-100 hover:text-red-400 p-0.5 rounded transition-all cursor-pointer"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Main Chat Thread Area */}
            <div className="flex-1 flex flex-col min-w-0 h-full">
              {/* Header Selector */}
              <div className="flex items-center justify-between mb-6 shrink-0">
                <div>
                  <h2 className="text-xl font-bold">Semantic RAG Chat</h2>
                  <p className="text-xs text-gray-400 mt-1">Converse with your papers using secure vector retrieval.</p>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-400 font-semibold">SEARCH SCOPE:</span>
                  <select
                    value={selectedPaperForChat}
                    onChange={(e) => setSelectedPaperForChat(e.target.value)}
                    className="glass-input text-xs py-1.5"
                  >
                    <option value="">Global Workspace (All Papers)</option>
                    {Object.keys(papers).map((pid) => (
                      <option key={pid} value={pid}>{pid.replace(/_/g, " ")}</option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Chat Thread */}
              <div className="flex-1 glass-panel rounded-2xl p-6 overflow-y-auto mb-4 space-y-5 flex flex-col min-h-0">
                {chatLog.length === 0 ? (
                  <div className="flex-1 flex flex-col items-center justify-center text-gray-500 text-center gap-3">
                    <MessageSquare className="h-10 w-10 text-gray-600" />
                    <h3 className="font-semibold text-gray-400">Ask a Question</h3>
                    <p className="text-xs text-gray-500 max-w-sm">Compare architectures, request summaries, or verify claims across your files.</p>
                  </div>
                ) : (
                  <>
                    {chatLog.map((msg, idx) => (
                      <div key={idx} className={`flex gap-4 ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                        <div className={`max-w-[85%] rounded-xl p-4 text-sm ${
                          msg.role === "user" 
                            ? "bg-blue-600 text-white rounded-br-none" 
                            : msg.role === "error"
                            ? "bg-red-950/40 border border-red-800/40 text-red-200"
                            : "bg-gray-900/60 border border-white/5 text-gray-100 rounded-bl-none"
                        }`}>
                          <p className="leading-relaxed">{msg.text}</p>
                          
                          {/* Chunks/Sources Quotes */}
                          {msg.chunks && msg.chunks.length > 0 && (
                            <div className="mt-4 pt-3 border-t border-white/5 space-y-2">
                              <span className="text-[10px] font-bold text-blue-400 tracking-wider">RETRIEVED CITATIONS:</span>
                              <div className="grid grid-cols-1 gap-2 max-h-32 overflow-y-auto pr-1">
                                {msg.chunks.map((chk, cidx) => (
                                  <div key={cidx} className="bg-black/20 p-2.5 rounded-lg text-xs border border-white/5">
                                    <div className="flex items-center justify-between font-bold text-[10px] text-gray-400 mb-1">
                                      <span>PAPER: {chk.paper_id.replace(/_/g, " ")}</span>
                                      <span>SCORE: {(chk.similarity_score ?? 0).toFixed(3)}</span>
                                    </div>
                                    <p className="text-gray-300 italic text-[11px]">"...{(chk.content ?? "").substring(0, 150)}..."</p>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                    {chatLoading && (
                      <div className="flex gap-4 justify-start">
                        <div className="bg-gray-900/60 border border-white/5 rounded-xl rounded-bl-none p-4 flex items-center gap-2.5 text-xs text-gray-400">
                          <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
                          <span>Generating Grounded Response...</span>
                        </div>
                      </div>
                    )}
                    <div ref={chatEndRef} />
                  </>
                )}
              </div>

              {/* Chat Input */}
              <form onSubmit={handleChatSubmit} className="flex gap-3 shrink-0">
                <input
                  type="text"
                  required
                  placeholder="Ask a question about your papers..."
                  value={chatQuery}
                  onChange={(e) => setChatQuery(e.target.value)}
                  className="glass-input flex-1 py-3"
                  disabled={chatLoading}
                />
                <button
                  type="submit"
                  disabled={chatLoading}
                  className="px-6 bg-blue-600 hover:bg-blue-500 transition-all font-semibold rounded-lg flex items-center justify-center gap-2 text-white cursor-pointer"
                >
                  <Search className="h-4.5 w-4.5" />
                </button>
              </form>
            </div>
          </div>
        )}

        {/* Compare Analytics Tab */}
        {activeTab === "analytics" && (
          <div className="p-8 w-full max-w-7xl mx-auto h-[calc(100vh-80px)] overflow-hidden flex gap-6">
            {/* Analytics Chat Sessions Sidebar */}
            <div className="w-64 shrink-0 flex flex-col glass-panel rounded-2xl p-4 overflow-hidden border border-white/5 bg-gray-900/10">
              <button
                onClick={() => handleNewChat("analytics")}
                className="w-full py-2.5 px-4 mb-4 bg-blue-600 hover:bg-blue-500 font-semibold rounded-lg shadow text-xs flex items-center justify-center gap-2 cursor-pointer transition-all shrink-0 text-white"
              >
                <Plus className="h-4 w-4" />
                New Comparison
              </button>
              <div className="flex-1 overflow-y-auto space-y-1.5 pr-1">
                <span className="text-[10px] font-bold text-gray-400 block px-2 mb-2 tracking-wider">PREVIOUS COMPARISONS</span>
                {analyticsSessions.length === 0 ? (
                  <div className="text-center text-xs text-gray-500 py-8 italic">No comparisons saved</div>
                ) : (
                  analyticsSessions.map((session) => (
                    <div
                      key={session.id}
                      onClick={() => loadSessionMessages(session.id, "analytics")}
                      className={`flex items-center justify-between group p-2.5 rounded-lg text-xs cursor-pointer transition-all ${
                        selectedAnalyticsSessionId === session.id
                          ? "bg-blue-600/20 text-blue-200 border border-blue-500/30"
                          : "hover:bg-white/5 text-gray-300 border border-transparent"
                      }`}
                    >
                      <span className="truncate pr-2">{session.title}</span>
                      <button
                        onClick={(e) => handleDeleteSession(session.id, "analytics", e)}
                        className="opacity-0 group-hover:opacity-100 hover:text-red-400 p-0.5 rounded transition-all cursor-pointer"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Main Content Area */}
            <div className="flex-1 flex flex-col min-w-0 h-full">
              {/* Header Info */}
              <div className="mb-4 shrink-0">
                <h2 className="text-xl font-bold">Comparative Analytics</h2>
                <p className="text-xs text-gray-400 mt-1">Perform Map-Reduce RAG comparison across all structured paper profiles.</p>
              </div>

              {/* Grid of Controls & Chat */}
              <div className="flex-1 grid grid-cols-1 lg:grid-cols-4 gap-6 items-stretch min-h-0">
                
                {/* Controls Column */}
                <div className="lg:col-span-1 glass-panel p-5 rounded-xl space-y-4 flex flex-col overflow-hidden bg-gray-900/5 select-none border border-white/5">
                  <h3 className="text-xs font-bold text-gray-400 tracking-wider shrink-0">SELECT PAPERS:</h3>
                  <div className="flex-1 overflow-y-auto space-y-2 pr-1">
                    {Object.keys(papers).map((pid) => (
                      <label key={pid} className="flex items-center gap-2.5 text-xs text-gray-300 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={selectedPapersForAnalytics.includes(pid)}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setSelectedPapersForAnalytics((prev) => [...prev, pid]);
                            } else {
                              setSelectedPapersForAnalytics((prev) => prev.filter((x) => x !== pid));
                            }
                          }}
                          className="rounded border-white/15 bg-gray-900 text-blue-600 focus:ring-0"
                        />
                        <span className="truncate">{pid.replace(/_/g, " ")}</span>
                      </label>
                    ))}
                  </div>
                  <div className="pt-2 border-t border-white/5 flex gap-2 shrink-0">
                    <button 
                      onClick={() => setSelectedPapersForAnalytics(Object.keys(papers))}
                      className="text-[10px] text-blue-400 hover:underline font-bold"
                    >
                      Select All
                    </button>
                    <span className="text-[10px] text-gray-600">|</span>
                    <button 
                      onClick={() => setSelectedPapersForAnalytics([])}
                      className="text-[10px] text-gray-400 hover:underline font-bold"
                    >
                      Clear
                    </button>
                  </div>
                </div>

                {/* Main Compare Column */}
                <div className="lg:col-span-3 flex flex-col h-full min-h-0">
                  
                  {/* Chat Log View */}
                  <div className="flex-1 glass-panel rounded-xl p-6 overflow-y-auto mb-4 space-y-5 min-h-0">
                    {analyticsChatLog.length === 0 ? (
                      <div className="h-full flex flex-col items-center justify-center text-gray-500 text-center gap-3">
                        <BarChart2 className="h-10 w-10 text-gray-600" />
                        <h3 className="font-semibold text-gray-400">Compare Papers Side-by-Side</h3>
                        <p className="text-xs text-gray-500 max-w-sm">Enter a comparison query to trigger the AI Map-Reduce compiler (e.g. "compare objectives", "limitations").</p>
                      </div>
                    ) : (
                      <div className="space-y-6">
                        {analyticsChatLog.map((msg, idx) => (
                          <div key={idx} className={`flex gap-4 ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                            <div className={`max-w-[90%] rounded-xl p-5 text-sm ${
                              msg.role === "user" 
                                ? "bg-blue-600 text-white rounded-br-none px-4 py-2.5" 
                                : msg.role === "error"
                                ? "bg-red-950/40 border border-red-800/40 text-red-200"
                                : "bg-gray-900/60 border border-white/5 text-gray-100 rounded-bl-none w-full"
                            }`}>
                              {msg.role === "user" ? (
                                <p className="leading-relaxed font-semibold">{msg.text}</p>
                              ) : (
                                <div className="space-y-4">
                                  <div className="flex items-center justify-between border-b border-white/5 pb-2">
                                    <span className="text-[10px] font-bold text-blue-400 tracking-wider">MAP-REDUCE SYNTHESIS:</span>
                                    {msg.fields && msg.fields.length > 0 && (
                                      <span className="text-[9px] text-gray-500">FIELDS ROUTED: {msg.fields.join(", ")}</span>
                                    )}
                                  </div>
                                  <div className="prose prose-invert max-w-none text-gray-200">
                                    {renderMarkdown(msg.text)}
                                  </div>
                                  {msg.warnings && msg.warnings.length > 0 && (
                                    <div className="pt-2 border-t border-white/5 text-[10px] text-gray-400 space-y-1">
                                      <span className="font-bold text-yellow-400/80">WARNINGS:</span>
                                      {msg.warnings.map((w, wIdx) => (
                                        <div key={wIdx}>• {w}</div>
                                      ))}
                                    </div>
                                  )}
                                </div>
                              )}
                            </div>
                          </div>
                        ))}
                        {analyticsLoading && (
                          <div className="flex gap-4 justify-start">
                            <div className="bg-gray-900/60 border border-white/5 rounded-xl rounded-bl-none p-4 flex items-center gap-2.5 text-xs text-gray-400">
                              <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
                              <span>Mapping profiles and reducing answers...</span>
                            </div>
                          </div>
                        )}
                        <div ref={analyticsEndRef} />
                      </div>
                    )}
                  </div>

                  {/* Input Form */}
                  <form onSubmit={runAnalytics} className="flex gap-3 shrink-0">
                    <input
                      type="text"
                      required
                      placeholder="Ask a follow-up comparison (e.g. 'Compare their limitations', 'Which is faster?')..."
                      value={analyticsQuery}
                      onChange={(e) => setAnalyticsQuery(e.target.value)}
                      className="glass-input flex-1 py-3"
                      disabled={analyticsLoading || Object.keys(papers).length === 0}
                    />
                    <button
                      type="submit"
                      disabled={analyticsLoading || Object.keys(papers).length === 0}
                      className="px-6 bg-blue-600 hover:bg-blue-500 disabled:bg-blue-800 transition-all font-semibold rounded-lg flex items-center justify-center gap-2 text-sm text-white cursor-pointer"
                    >
                      Compare
                    </button>
                  </form>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Chronological Timeline Tab */}
        {activeTab === "timeline" && (
          <div className="p-8 max-w-4xl w-full mx-auto space-y-8">
            <div>
              <h2 className="text-xl font-bold">Research Timeline</h2>
              <p className="text-xs text-gray-400 mt-1">Chronological evolution of research papers uploaded to your workspace.</p>
            </div>

            {timelineLoading ? (
              <div className="flex flex-col items-center justify-center py-20 text-gray-400 gap-3">
                <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
                <span className="text-sm">Sorting timelines and compiling overview...</span>
              </div>
            ) : timelineData ? (
              <div className="space-y-6">
                
                {/* Overview Card */}
                <div className="glass-panel p-6 rounded-xl border-l-4 border-blue-500">
                  <span className="text-[10px] font-bold text-blue-400 tracking-wider block mb-1.5">DOMAIN EVOLUTION SUMMARY</span>
                  <p className="text-sm text-gray-300 leading-relaxed">{timelineData.overview}</p>
                </div>

                {/* Timeline Pipeline */}
                <div className="relative border-l border-white/10 pl-6 ml-4 space-y-8">
                  {timelineData.timeline.map((item, idx) => (
                    <div key={idx} className="relative">
                      {/* Node circle */}
                      <span className="absolute -left-[31px] top-1.5 h-4 w-4 rounded-full bg-blue-600 border-4 border-darkBg shadow shadow-blue-500/50"></span>
                      
                      <div className="glass-panel p-5 rounded-xl space-y-2">
                        <div className="flex items-center justify-between">
                          <h4 className="font-bold text-sm text-gray-200">{item.title}</h4>
                          <span className="px-2 py-0.5 rounded bg-blue-950 text-blue-300 border border-blue-800/40 text-[10px] font-bold">
                            {item.publication_year || "Unknown Year"}
                          </span>
                        </div>
                        <p className="text-[11px] text-gray-400">AUTHORS: {item.authors?.join(", ") || "Not specified"}</p>
                        <p className="text-xs text-gray-300 leading-relaxed pt-1">{item.objective}</p>
                      </div>
                    </div>
                  ))}
                </div>

              </div>
            ) : (
              <div className="glass-panel p-20 rounded-xl text-center text-gray-500">
                Timeline could not be loaded. Please ensure you have completed papers in your library.
              </div>
            )}
          </div>
        )}

        {/* Topic Clustering Tab */}
        {activeTab === "clusters" && (
          <div className="p-8 max-w-4xl w-full mx-auto space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-xl font-bold">Topic Clustering</h2>
                <p className="text-xs text-gray-400 mt-1">Mathematical paper groupings labeled dynamically by Gemini.</p>
              </div>
              <div className="flex bg-gray-900/50 p-1 rounded-lg border border-white/5 text-xs font-semibold">
                <button
                  onClick={() => handleClusterTypeChange("hierarchical")}
                  className={`px-3 py-1.5 rounded-md transition-all ${
                    clusterType === "hierarchical" ? "bg-blue-600 text-white" : "text-gray-400 hover:text-white"
                  }`}
                >
                  Hierarchical
                </button>
                <button
                  onClick={() => handleClusterTypeChange("cosine")}
                  className={`px-3 py-1.5 rounded-md transition-all ${
                    clusterType === "cosine" ? "bg-blue-600 text-white" : "text-gray-400 hover:text-white"
                  }`}
                >
                  Cosine
                </button>
              </div>
            </div>

            {clusterLoading ? (
              <div className="flex flex-col items-center justify-center py-20 text-gray-400 gap-3">
                <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
                <span className="text-sm">Calculating clustering coefficients...</span>
              </div>
            ) : clusterData && clusterData.clusters ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                {clusterData.clusters.map((cluster) => (
                  <div key={cluster.cluster_id} className="glass-panel p-6 rounded-xl flex flex-col justify-between gap-4 border-t-2 border-indigo-500">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <h4 className="font-bold text-sm text-gray-100">{cluster.name}</h4>
                        <span className="text-[10px] font-bold text-indigo-400 bg-indigo-950/40 border border-indigo-800/40 px-2 py-0.5 rounded">
                          CLUSTER {cluster.cluster_id}
                        </span>
                      </div>
                      <p className="text-xs text-gray-300 leading-relaxed">{cluster.description}</p>
                    </div>

                    <div className="space-y-2 pt-3 border-t border-white/5">
                      <span className="text-[9px] font-bold text-gray-500 tracking-wider block">MEMBERS:</span>
                      <div className="flex flex-wrap gap-1.5">
                        {cluster.papers.map((paper) => (
                          <span 
                            key={paper} 
                            className="text-[10px] font-semibold bg-gray-900 border border-white/5 rounded px-2.5 py-1 text-gray-300"
                          >
                            {paper.replace(/_/g, " ")}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="glass-panel p-20 rounded-xl text-center text-gray-500">
                Clustering could not be loaded. Please upload papers first.
              </div>
            )}
          </div>
        )}

      </main>
    </div>
  );
}
