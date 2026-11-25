(() => {
  "use strict";
  const byId = (id) => document.getElementById(id);
  const form = byId("connect-form");
  const apiInput = byId("api-url");
  const tokenInput = byId("api-token");
  const errorBox = byId("connection-error");
  const dashboard = byId("dashboard");
  const statePill = byId("connection-state");
  const refreshButton = byId("refresh-button");
  const simulationDay = byId("simulation-day");
  const simulationDayValue = byId("simulation-day-value");
  const providerState = byId("provider-state");
  const providerNote = byId("chat-provider-note");
  const chatLog = byId("chat-log");
  const chatForm = byId("chat-form");
  const chatInput = byId("chat-input");
  const sendChat = byId("send-chat");
  const clearChat = byId("clear-chat");
  const chatError = byId("chat-error");
  let apiBase = "";
  let bearerToken = "";
  let pollHandle = null;
  let providerReady = false;
  let sessionId = "control-room-" + (window.crypto?.randomUUID?.() || (Date.now() + "-" + Math.random().toString(16).slice(2)));

  const pretty = (value) => String(value ?? "unknown").replaceAll("_", " ");
  const safeNode = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const setConnection = (state, label) => {
    statePill.className = "state-pill " + state;
    statePill.lastChild.textContent = label;
  };
  const setError = (message) => {
    errorBox.hidden = !message;
    errorBox.textContent = message || "";
  };
  const appendBar = (parent, label, value, meta, tone) => {
    const row = safeNode("div", tone === "blue" ? "proj-row" : "pop-row");
    row.append(safeNode("span", tone === "blue" ? "proj-name" : "pop-name", label));
    const track = safeNode("div", "bar-track");
    const fill = safeNode("div", "bar-fill");
    fill.style.width = Math.max(0, Math.min(100, value * 100)) + "%";
    track.append(fill);
    row.append(track);
    row.append(safeNode("span", tone === "blue" ? "proj-value" : "bar-value", (value * 100).toFixed(1) + "%"));
    if (meta) row.append(safeNode("span", "pop-meta", meta));
    parent.append(row);
  };
  const render = (payload) => {
    const snap = payload?.development?.prenatal;
    if (!snap || !Array.isArray(snap.populations) || !Array.isArray(snap.projections)) {
      throw new Error("The API responded, but its monitoring snapshot has an unexpected shape.");
    }
    byId("phase-value").textContent = pretty(snap.phase);
    byId("day-value").textContent = "DAY " + snap.age_days + " · WEEK " + snap.gestational_week + " / 40";
    byId("population-value").textContent = snap.population_count;
    byId("projection-value").textContent = snap.projection_count;
    byId("updates-value").textContent = Number(snap.total_plasticity_updates || 0).toLocaleString();
    byId("exposures-value").textContent = Number(snap.accepted_exposures || 0) + " accepted simulated cues";
    byId("mean-strength").textContent = "MEAN " + (Number(snap.mean_projection_strength || 0) * 100).toFixed(2) + "%";

    const populations = byId("population-list");
    populations.replaceChildren();
    snap.populations.forEach((item) => {
      appendBar(populations, pretty(item.identifier), Number(item.maturity || 0),
        "activity " + (Number(item.activity || 0) * 100).toFixed(1) + "% · " + item.capacity_units + " abstract units");
    });

    const projections = byId("projection-list");
    projections.replaceChildren();
    snap.projections.forEach((item) => {
      appendBar(projections, pretty(item.source) + " → " + pretty(item.target), Number(item.strength || 0),
        Number(item.update_count || 0).toLocaleString() + " updates", "blue");
    });

    const systems = byId("systems-grid");
    systems.replaceChildren();
    Object.entries(payload.cognitive_systems || {}).forEach(([name, state]) => {
      const tile = safeNode("article", "system-tile");
      tile.append(safeNode("div", "system-name", pretty(name)));
      tile.append(safeNode("div", "system-state", pretty(state)));
      systems.append(tile);
    });
    const runtime = payload.organism?.runtime || {};
    [
      ["identity source", runtime.identity_source || "unknown"],
      ["storage mode", runtime.storage_mode || "unknown"],
      ["external database", runtime.external_database_connected === false ? "not connected" : "status unavailable"],
      ["restart retention", runtime.personal_experience_persists_restart === false ? "volatile · resets on restart" : "status unavailable"]
    ].forEach(([name, state]) => {
      const tile = safeNode("article", "system-tile");
      tile.append(safeNode("div", "system-name", name));
      tile.append(safeNode("div", "system-state", pretty(state)));
      systems.append(tile);
    });
    byId("last-updated").textContent = "UPDATED " + new Date(payload.observed_at || Date.now()).toLocaleString();
    dashboard.hidden = false;
    setConnection("connected", " CONNECTED");
    refreshButton.disabled = false;
  };

  const authFetch = (path, options = {}) => fetch(apiBase + path, {
    ...options,
    headers: {
      "Authorization": "Bearer " + bearerToken,
      "Accept": "application/json",
      ...(options.headers || {})
    },
    cache: "no-store",
    credentials: "omit",
    redirect: "error"
  });
  const setProviderState = (status) => {
    providerReady = Boolean(status?.configured);
    const missingRemoteKey = !providerReady
      && Boolean(status?.model)
      && /^https:\/\//.test(String(status?.base_url || ""))
      && status?.authentication_configured === false;
    providerState.textContent = providerReady
      ? "MODEL READY · " + String(status.model || "CONFIGURED")
      : missingRemoteKey ? "PROVIDER KEY REQUIRED" : "MODEL NOT CONFIGURED";
    providerNote.textContent = providerReady
      ? "The cognitive cycle is ready to receive messages. Replies are generated by the configured language model and coordinated by Artificial Human's software cognitive systems."
      : missingRemoteKey
        ? "The model endpoint is selected, but its private API key is missing. Add ARTIFICIAL_HUMAN_LLM_API_KEY in the Render service's Environment settings. Never enter provider keys in this dashboard."
        : "The API is live, but no reasoning model is configured yet. Add an OpenAI-compatible provider on the private Render service to enable conversation.";
    chatInput.disabled = !providerReady;
    sendChat.disabled = !providerReady;
    clearChat.disabled = !providerReady;
  };
  const appendMessage = (role, message) => {
    const bubble = safeNode("article", "chat-message " + role);
    bubble.append(safeNode("span", "chat-role", role === "user" ? "YOU" : "ARTIFICIAL HUMAN"));
    bubble.append(safeNode("p", "chat-content", message));
    chatLog.append(bubble);
    chatLog.scrollTop = chatLog.scrollHeight;
  };
  const setChatError = (message) => {
    chatError.hidden = !message;
    chatError.textContent = message || "";
  };
  const loadProviderStatus = async () => {
    try {
      const response = await authFetch("/brain/status");
      if (!response.ok) throw new Error("Could not inspect reasoning-provider status (HTTP " + response.status + ").");
      setProviderState(await response.json());
    } catch (error) {
      providerReady = false;
      setProviderState({configured:false});
      providerState.textContent = "CHAT STATUS UNAVAILABLE";
      setChatError(error instanceof Error ? error.message : "Unable to check model status.");
    }
  };
  chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = chatInput.value.trim();
    if (!message || !providerReady || !apiBase || !bearerToken) return;
    chatInput.value = "";
    setChatError("");
    appendMessage("user", message);
    sendChat.disabled = true;
    try {
      const response = await authFetch("/brain/respond", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({message, mode:"assistant", session_id:sessionId})
      });
      const payload = await response.json();
      if (!response.ok) {
        if (response.status === 503) {
          providerReady = false;
          setProviderState({configured:false});
        }
        throw new Error(payload.detail || "Conversation request failed (HTTP " + response.status + ").");
      }
      appendMessage("assistant", payload.answer || "The model returned an empty response.");
      providerState.textContent = "CONNECTED · " + String(payload.model || "MODEL");
    } catch (error) {
      setChatError(error instanceof Error ? error.message : "Unable to send this message.");
    } finally {
      sendChat.disabled = !providerReady;
      chatInput.focus();
    }
  });
  clearChat.addEventListener("click", async () => {
    if (!apiBase || !bearerToken || !providerReady) return;
    try {
      const response = await authFetch("/brain/sessions/" + encodeURIComponent(sessionId), {method:"DELETE"});
      if (!response.ok) throw new Error("Could not clear the working-memory session.");
      sessionId = "control-room-" + (window.crypto?.randomUUID?.() || (Date.now() + "-" + Math.random().toString(16).slice(2)));
      chatLog.replaceChildren(safeNode("div","chat-empty","Session cleared. Start a new conversation."));
      setChatError("");
    } catch (error) {
      setChatError(error instanceof Error ? error.message : "Unable to clear the session.");
    }
  });
  const refresh = async () => {
    if (!apiBase || !bearerToken) return;
    refreshButton.disabled = true;
    setError("");
    try {
      const response = await fetch(apiBase + "/monitor/overview?days=" + encodeURIComponent(simulationDay.value) + "&seed=7", {
        method: "GET",
        headers: { "Authorization": "Bearer " + bearerToken, "Accept": "application/json" },
        cache: "no-store",
        credentials: "omit",
        redirect: "error"
      });
      if (!response.ok) {
        if (response.status === 401) throw new Error("Authentication failed. Check the bearer token.");
        if (response.status === 503) throw new Error("The public API is not configured with an authentication token.");
        throw new Error("Monitor API returned HTTP " + response.status + ".");
      }
      render(await response.json());
      await loadProviderStatus();
    } catch (error) {
      setConnection("failed", " CONNECTION ISSUE");
      setError(error instanceof Error ? error.message : "Unable to reach the monitor API.");
      if (!dashboard.hidden) refreshButton.disabled = false;
    } finally {
      refreshButton.disabled = false;
    }
  };

  simulationDay.addEventListener("input", () => {
    simulationDayValue.textContent = simulationDay.value + " / 280 days";
  });
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    apiBase = apiInput.value.trim().replace(/\/+$/, "");
    bearerToken = tokenInput.value;
    tokenInput.value = "";
    let target;
    try { target = new URL(apiBase); } catch { setError("Enter a valid API URL."); return; }
    const localHost = ["localhost", "127.0.0.1"].includes(target.hostname);
    if ((target.protocol !== "https:" && !(target.protocol === "http:" && localHost))
      || target.username || target.password || target.search || target.hash || !bearerToken) {
      bearerToken = "";
      setError("Use your own HTTPS API URL (HTTP is allowed only for localhost) and enter its bearer token.");
      return;
    }
    apiBase = target.origin + target.pathname.replace(/\/+$/, "");
    if (pollHandle) window.clearInterval(pollHandle);
    await refresh();
    pollHandle = window.setInterval(refresh, 30000);
  });
  refreshButton.addEventListener("click", refresh);
  window.addEventListener("pagehide", () => {
    if (pollHandle) window.clearInterval(pollHandle);
    bearerToken = "";
    apiBase = "";
    providerReady = false;
    chatInput.disabled = true;
    sendChat.disabled = true;
    clearChat.disabled = true;
  });
})();
