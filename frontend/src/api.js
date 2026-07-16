import axios from "axios";

const API_BASE_URL = "http://127.0.0.1:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
});

// Automatically inject JWT Bearer Token if it exists in localStorage
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem("authToken");
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Global response error handler (handling unauthorized sessions)
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      localStorage.removeItem("authToken");
      // Optional: reload page to redirect to login view
      window.location.reload();
    }
    return Promise.reject(error);
  }
);

export const authAPI = {
  register: async (email, password) => {
    const response = await api.post("/auth/register", { email, password });
    return response.data;
  },
  login: async (email, password) => {
    const formData = new FormData();
    formData.append("username", email);
    formData.append("password", password);
    const response = await api.post("/auth/login", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
    return response.data;
  },
};

export const papersAPI = {
  upload: async (files, onUploadProgress) => {
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append("file", files[i]);
    }
    const response = await api.post("/upload-pdf/", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
      onUploadProgress,
    });
    return response.data;
  },
  list: async () => {
    const response = await api.get("/papers/");
    return response.data;
  },
  delete: async (paperId) => {
    const response = await api.delete(`/papers/${paperId}/`);
    return response.data;
  },
  query: async (query, paperId = null, topK = 8) => {
    let url = `/query/?query=${encodeURIComponent(query)}&top_k=${topK}`;
    if (paperId) {
      url += `&paper_id=${encodeURIComponent(paperId)}`;
    }
    const response = await api.post(url);
    return response.data;
  },
};

export const analyticsAPI = {
  global: async (query, paperIds = null) => {
    let url = `/analytics/?query=${encodeURIComponent(query)}`;
    if (paperIds && paperIds.length > 0) {
      url += `&paper_ids=${encodeURIComponent(paperIds.join(","))}`;
    }
    const response = await api.post(url);
    return response.data;
  },
  reprocess: async (paperIds = null) => {
    let url = "/analytics/reprocess/";
    if (paperIds && paperIds.length > 0) {
      url += `?paper_ids=${encodeURIComponent(paperIds.join(","))}`;
    }
    const response = await api.post(url);
    return response.data;
  },
  getCosineClusters: async () => {
    const response = await api.get("/analytics/clusters/cosine/");
    return response.data;
  },
  getHierarchicalClusters: async () => {
    const response = await api.get("/analytics/clusters/hierarchical/");
    return response.data;
  },
  getTimeline: async () => {
    const response = await api.get("/analytics/timeline/");
    return response.data;
  },
};

export default api;
