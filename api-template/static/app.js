"use strict";

// --------------------------------------------------------------------------
// State + DOM
// --------------------------------------------------------------------------
const state = { token: null, username: null, chats: [], activeChatId: null, pendingFile: null };

const $ = (id) => document.getElementById(id);
const els = {
  loginView: $("login-view"), appView: $("app-view"),
  loginForm: $("login-form"), keyInput: $("key-input"),
  loginBtn: $("login-btn"), loginError: $("login-error"),
  newChatBtn: $("new-chat-btn"), chatList: $("chat-list"),
  userLabel: $("user-label"), logoutBtn: $("logout-btn"),
  chatTitle: $("chat-title"), datasetChip: $("dataset-chip"),
  messages: $("messages"), emptyState: $("empty-state"),
  composer: $("composer"), composerError: $("composer-error"),
  fileInput: $("file-input"), attachedFile: $("attached-file"),
  attachedName: $("attached-name"), clearFile: $("clear-file"),
  msgInput: $("message-input"), sendBtn: $("send-btn"),
};

const TOKEN_KEY = "featurify_token";

// --------------------------------------------------------------------------
// API helper
// --------------------------------------------------------------------------
async function api(path, { method = "GET", body, isForm = false } = {}) {
  const headers = {};
  if (state.token) headers["Authorization"] = "Bearer " + state.token;
  if (!isForm && body !== undefined) headers["Content-Type"] = "application/json";

  const resp = await fetch(path, {
    method,
    headers,
    body: isForm ? body : body !== undefined ? JSON.stringify(body) : undefined,
  });

  if (resp.status === 401) { logout(); throw new Error("Session expired. Please log in again."); }

  let data = null;
  try { data = await resp.json(); } catch (_) {}
  if (!resp.ok) throw new Error((data && data.detail) || "Request failed (" + resp.status + ")");
  return data;
}

// --------------------------------------------------------------------------
// Auth
// --------------------------------------------------------------------------
els.loginForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  els.loginError.hidden = true;
  const key = els.keyInput.value.trim();
  if (!key) return;
  els.loginBtn.disabled = true;
  els.loginBtn.textContent = "Checking…";
  try {
    const res = await api("/auth/login", { method: "POST", body: { key } });
    state.token = res.token;
    state.username = res.username;
    localStorage.setItem(TOKEN_KEY, res.token);
    enterApp();
  } catch (err) {
    els.loginError.textContent = err.message;
    els.loginError.hidden = false;
  } finally {
    els.loginBtn.disabled = false;
    els.loginBtn.textContent = "Enter";
  }
});

function logout() {
  localStorage.removeItem(TOKEN_KEY);
  state.token = null; state.username = null;
  state.chats = []; state.activeChatId = null;
  els.appView.hidden = true;
  els.loginView.hidden = false;
  els.keyInput.value = "";
}
els.logoutBtn.addEventListener("click", logout);

async function enterApp() {
  els.loginView.hidden = true;
  els.appView.hidden = false;
  els.userLabel.textContent = state.username;
  await loadChats();
}

// --------------------------------------------------------------------------
// Chats
// --------------------------------------------------------------------------
async function loadChats() {
  state.chats = await api("/api/chats");
  renderChatList();
  if (state.chats.length && !state.activeChatId) {
    selectChat(state.chats[0].id);
  } else if (!state.chats.length) {
    state.activeChatId = null;
    els.chatTitle.textContent = "—";
    els.datasetChip.hidden = true;
    clearMessages(true);
  }
}

function renderChatList() {
  els.chatList.innerHTML = "";
  for (const c of state.chats) {
    const item = document.createElement("div");
    item.className = "chat-item" + (c.id === state.activeChatId ? " active" : "");
    item.onclick = () => selectChat(c.id);

    const title = document.createElement("span");
    title.className = "ci-title";
    title.textContent = c.title;
    item.appendChild(title);

    if (c.dataset_filename) {
      const d = document.createElement("span");
      d.className = "ci-data";
      d.textContent = "csv";
      item.appendChild(d);
    }

    const del = document.createElement("button");
    del.className = "ci-del";
    del.textContent = "×";
    del.title = "Delete chat";
    del.onclick = (e) => { e.stopPropagation(); deleteChat(c.id); };
    item.appendChild(del);

    els.chatList.appendChild(item);
  }
}

els.newChatBtn.addEventListener("click", async () => {
  const chat = await api("/api/chats", { method: "POST", body: {} });
  state.chats.unshift(chat);
  renderChatList();
  selectChat(chat.id);
  els.msgInput.focus();
});

async function deleteChat(id) {
  if (!confirm("Delete this chat?")) return;
  await api("/api/chats/" + id, { method: "DELETE" });
  state.chats = state.chats.filter((c) => c.id !== id);
  if (state.activeChatId === id) state.activeChatId = null;
  renderChatList();
  if (!state.activeChatId && state.chats.length) selectChat(state.chats[0].id);
  else if (!state.chats.length) {
    els.chatTitle.textContent = "—";
    els.datasetChip.hidden = true;
    clearMessages(true);
  }
}

async function selectChat(id) {
  state.activeChatId = id;
  clearPendingFile();
  renderChatList();
  const chat = state.chats.find((c) => c.id === id);
  els.chatTitle.textContent = chat ? chat.title : "—";
  updateDatasetChip(chat);
  const msgs = await api("/api/chats/" + id + "/messages");
  clearMessages(msgs.length === 0);
  for (const m of msgs) appendMessage(m);
  scrollToBottom();
}

function updateDatasetChip(chat) {
  if (chat && chat.dataset_filename) {
    els.datasetChip.textContent = "◆ " + chat.dataset_filename;
    els.datasetChip.hidden = false;
  } else {
    els.datasetChip.hidden = true;
  }
}

// --------------------------------------------------------------------------
// Messages
// --------------------------------------------------------------------------
function clearMessages(showEmpty) {
  els.messages.innerHTML = "";
  if (showEmpty) {
    const e = els.emptyState.cloneNode(true);
    e.hidden = false;
    els.messages.appendChild(e);
  }
}

function scrollToBottom() {
  els.messages.scrollTop = els.messages.scrollHeight;
}

function appendMessage(m, fileName) {
  const empty = els.messages.querySelector(".empty-state");
  if (empty) empty.remove();

  const wrap = document.createElement("div");
  wrap.className = "msg " + m.role;

  const role = document.createElement("div");
  role.className = "msg-role";
  role.textContent = m.role === "user" ? "You" : "Featurify";
  wrap.appendChild(role);

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  if (m.role === "user") {
    bubble.textContent = m.content;
    if (fileName) {
      const tag = document.createElement("div");
      tag.className = "file-tag";
      tag.textContent = "◆ " + fileName;
      bubble.appendChild(tag);
    }
  } else {
    const analysis = document.createElement("div");
    analysis.className = "analysis";
    analysis.textContent = m.content || "(no analysis returned)";
    bubble.appendChild(analysis);
    renderStructured(bubble, m.data);
  }

  wrap.appendChild(bubble);
  els.messages.appendChild(wrap);
  return wrap;
}

function renderStructured(bubble, data) {
  if (!data) return;
  const groups = [
    ["remove", "Remove", data.remove_features],
    ["transform", "Transform", data.transform_features],
    ["create", "Create", data.create_features],
    ["models", "Recommended models", data.recommended_models],
  ];
  const hasAny = groups.some(([, , arr]) => Array.isArray(arr) && arr.length);
  if (hasAny) {
    const container = document.createElement("div");
    container.className = "feat-groups";
    for (const [cls, label, arr] of groups) {
      if (!Array.isArray(arr) || !arr.length) continue;
      const g = document.createElement("div");
      g.className = "feat-group";
      const l = document.createElement("div");
      l.className = "feat-label " + cls;
      l.textContent = label;
      const chips = document.createElement("div");
      chips.className = "chips";
      for (const v of arr) {
        const chip = document.createElement("span");
        chip.className = "chip";
        chip.textContent = v;
        chips.appendChild(chip);
      }
      g.appendChild(l); g.appendChild(chips);
      container.appendChild(g);
    }
    bubble.appendChild(container);
  }
}

// --------------------------------------------------------------------------
// Composer
// --------------------------------------------------------------------------
els.fileInput.addEventListener("change", () => {
  const f = els.fileInput.files[0];
  if (!f) return;
  if (!f.name.toLowerCase().endsWith(".csv")) {
    showComposerError("Only CSV files are allowed.");
    els.fileInput.value = "";
    return;
  }
  state.pendingFile = f;
  els.attachedName.textContent = "◆ " + f.name;
  els.attachedFile.hidden = false;
});

els.clearFile.addEventListener("click", clearPendingFile);
function clearPendingFile() {
  state.pendingFile = null;
  els.fileInput.value = "";
  els.attachedFile.hidden = true;
}

function showComposerError(msg) {
  els.composerError.textContent = msg;
  els.composerError.hidden = false;
  setTimeout(() => { els.composerError.hidden = true; }, 5000);
}

// auto-grow textarea
els.msgInput.addEventListener("input", () => {
  els.msgInput.style.height = "auto";
  els.msgInput.style.height = Math.min(els.msgInput.scrollHeight, 180) + "px";
});

// Enter to send, Shift+Enter for newline
els.msgInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    els.composer.requestSubmit();
  }
});

els.composer.addEventListener("submit", async (e) => {
  e.preventDefault();
  els.composerError.hidden = true;

  if (!state.activeChatId) {
    // No chat yet — create one on the fly.
    const chat = await api("/api/chats", { method: "POST", body: {} });
    state.chats.unshift(chat);
    state.activeChatId = chat.id;
    renderChatList();
    clearMessages(false);
  }

  const text = els.msgInput.value.trim();
  if (!text) return;

  const file = state.pendingFile;
  const fileName = file ? file.name : null;

  // optimistic user bubble
  appendMessage({ role: "user", content: text }, fileName);
  scrollToBottom();

  // reset composer
  els.msgInput.value = "";
  els.msgInput.style.height = "auto";
  clearPendingFile();
  setSending(true);

  // thinking indicator
  const thinking = appendThinking();
  scrollToBottom();

  try {
    const form = new FormData();
    form.append("message", text);
    if (file) form.append("file", file);

    const assistant = await api("/api/chats/" + state.activeChatId + "/messages", {
      method: "POST", body: form, isForm: true,
    });

    thinking.remove();
    appendMessage(assistant);
    scrollToBottom();

    // Refresh chat list (title + dataset chip may have changed).
    await refreshChatMeta();
  } catch (err) {
    thinking.remove();
    showComposerError(err.message);
  } finally {
    setSending(false);
    els.msgInput.focus();
  }
});

function appendThinking() {
  const empty = els.messages.querySelector(".empty-state");
  if (empty) empty.remove();
  const wrap = document.createElement("div");
  wrap.className = "msg assistant";
  wrap.innerHTML =
    '<div class="msg-role">Featurify</div>' +
    '<div class="bubble"><span class="thinking">analyzing</span></div>';
  els.messages.appendChild(wrap);
  return wrap;
}

function setSending(on) {
  els.sendBtn.disabled = on;
  els.sendBtn.textContent = on ? "…" : "Send";
}

async function refreshChatMeta() {
  state.chats = await api("/api/chats");
  renderChatList();
  const chat = state.chats.find((c) => c.id === state.activeChatId);
  if (chat) {
    els.chatTitle.textContent = chat.title;
    updateDatasetChip(chat);
  }
}

// --------------------------------------------------------------------------
// Boot
// --------------------------------------------------------------------------
(async function boot() {
  const stored = localStorage.getItem(TOKEN_KEY);
  if (!stored) return; // show login
  state.token = stored;
  try {
    const me = await api("/auth/me");
    state.username = me.username;
    await enterApp();
  } catch (_) {
    logout();
  }
})();
