(() => {
  "use strict";

  const byId = (id) => document.getElementById(id);
  const loginPanel = byId("login-panel");
  const loginForm = byId("login-form");
  const usernameInput = byId("login-username");
  const passwordInput = byId("login-password");
  const passkeyLogin = byId("passkey-login");
  const passkeyRegister = byId("passkey-register");
  const passkeyState = byId("passkey-state");
  const passkeyError = byId("passkey-error");
  const errorBox = byId("connection-error");
  const dashboard = byId("dashboard");
  const statePill = byId("connection-state");
  const refreshButton = byId("refresh-button");
  const logoutButton = byId("logout-button");
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
  const cameraStart = byId("camera-start");
  const cameraCaptureButton = byId("camera-capture-button");
  const cameraStop = byId("camera-stop");
  const cameraVideo = byId("camera-preview");
  const cameraCapture = byId("camera-capture");
  const cameraPlaceholder = byId("camera-placeholder");
  const cameraState = byId("camera-state");
  const cameraError = byId("camera-error");

  const apiBase = "/api";
  let providerReady = false;
  let activeSession = false;
  let cameraStream = null;
  let capturedImageUrl = null;
  let passkeyIsRegistered = false;
  let sessionId = "control-room-" + (window.crypto?.randomUUID?.()
    || (Date.now() + "-" + Math.random().toString(16).slice(2)));

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
  const setPasskeyError = (message) => {
    passkeyError.hidden = !message;
    passkeyError.textContent = message || "";
  };
  const setCameraError = (message) => {
    cameraError.hidden = !message;
    cameraError.textContent = message || "";
  };
  const apiFetch = (path, options = {}) => fetch(apiBase + path, {
    ...options,
    headers: {
      "Accept": "application/json",
      ...(options.headers || {})
    },
    cache: "no-store",
    credentials: "same-origin",
    redirect: "error"
  });
  const responsePayload = async (response) => {
    try { return await response.json(); } catch { return {}; }
  };

  const appendBar = (parent, label, value, meta, tone) => {
    const row = safeNode("div", tone === "blue" ? "proj-row" : "pop-row");
    row.append(safeNode("span", tone === "blue" ? "proj-name" : "pop-name", label));
    const track = safeNode("div", "bar-track");
    const fill = safeNode("div", "bar-fill");
    fill.style.width = Math.max(0, Math.min(100, value * 100)) + "%";
    track.append(fill);
    row.append(track);
    row.append(safeNode(
      "span",
      tone === "blue" ? "proj-value" : "bar-value",
      (value * 100).toFixed(1) + "%"
    ));
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
    const dynamics = payload.organism?.neural_dynamics;
    if (dynamics && Array.isArray(dynamics.populations)) {
      const summary = safeNode("article", "system-tile");
      summary.append(safeNode("div", "system-name", "recurrent activity"));
      summary.append(safeNode(
        "div",
        "system-state",
        Number(dynamics.inputs_processed || 0).toLocaleString() + " inputs · "
          + Number(dynamics.integration_steps || 0).toLocaleString() + " updates · "
          + dynamics.active_population_count + "/" + dynamics.population_count + " active"
      ));
      systems.append(summary);
      dynamics.populations.forEach((population) => {
        const tile = safeNode("article", "system-tile");
        tile.append(safeNode("div", "system-name", "activity · " + pretty(population.identifier)));
        const track = safeNode("div", "bar-track");
        const fill = safeNode("div", "bar-fill");
        fill.style.width = Math.max(0, Math.min(100, Number(population.activity || 0) * 100)) + "%";
        track.append(fill);
        tile.append(track);
        tile.append(safeNode("div", "system-state", (Number(population.activity || 0) * 100).toFixed(1) + "%"));
        systems.append(tile);
      });
    }
    const associations = payload.organism?.associative_memory;
    if (associations) {
      const tile = safeNode("article", "system-tile");
      tile.append(safeNode("div", "system-name", "associative memory"));
      tile.append(safeNode(
        "div",
        "system-state",
        Number(associations.association_count || 0).toLocaleString() + " links · "
          + Number(associations.coactivations || 0).toLocaleString() + " co-activations"
      ));
      systems.append(tile);
    }
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
        ? "The model endpoint is selected, but its private API key is missing. Add ARTIFICIAL_HUMAN_LLM_API_KEY in Render's Environment settings. Never enter provider keys here."
        : "The API is connected, but no reasoning model is configured yet. Add an OpenAI-compatible provider on the private Render service to enable conversation.";
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
      const response = await apiFetch("/brain/status");
      if (!response.ok) throw new Error("Could not inspect reasoning-provider status (HTTP " + response.status + ").");
      setProviderState(await responsePayload(response));
    } catch (error) {
      providerReady = false;
      setProviderState({configured:false});
      providerState.textContent = "CHAT STATUS UNAVAILABLE";
      setChatError(error instanceof Error ? error.message : "Unable to check model status.");
    }
  };

  const startAuthenticatedSession = async () => {
    activeSession = true;
    loginPanel.hidden = true;
    dashboard.hidden = false;
    logoutButton.hidden = false;
    setError("");
    setConnection("connected", " CONNECTED");
    await refresh();
  };
  const showSignedOut = (message = "") => {
    activeSession = false;
    dashboard.hidden = true;
    loginPanel.hidden = false;
    logoutButton.hidden = true;
    providerReady = false;
    chatInput.disabled = true;
    sendChat.disabled = true;
    clearChat.disabled = true;
    stopCamera();
    setConnection("idle", " SIGN IN");
    setError(message);
  };
  const inspectSession = async () => {
    try {
      const response = await apiFetch("/auth/session");
      const payload = await responsePayload(response);
      if (!response.ok) {
        showSignedOut(payload.detail || "The sign-in service is not configured yet.");
        return;
      }
      passkeyIsRegistered = Boolean(payload.passkey_registered);
      passkeyLogin.hidden = !passkeyIsRegistered;
      passkeyState.textContent = passkeyIsRegistered ? "PASSKEY READY" : "NOT REGISTERED";
      if (payload.authenticated) {
        usernameInput.value = payload.username || "";
        await startAuthenticatedSession();
      } else {
        showSignedOut("");
      }
    } catch (error) {
      showSignedOut(error instanceof Error ? error.message : "Cannot reach the sign-in service.");
      setConnection("failed", " OFFLINE");
    }
  };

  loginForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const username = usernameInput.value.trim();
    const password = passwordInput.value;
    if (!username || !password) return;
    setError("");
    const submit = loginForm.querySelector("button[type=submit]");
    submit.disabled = true;
    try {
      const response = await apiFetch("/auth/password", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({username, password})
      });
      const payload = await responsePayload(response);
      if (!response.ok) throw new Error(payload.detail || "Sign-in failed (HTTP " + response.status + ").");
      passwordInput.value = "";
      passkeyIsRegistered = Boolean(payload.passkey_registered);
      passkeyLogin.hidden = !passkeyIsRegistered;
      passkeyState.textContent = passkeyIsRegistered ? "PASSKEY READY" : "NOT REGISTERED";
      await startAuthenticatedSession();
    } catch (error) {
      setError(error instanceof Error ? error.message : "Unable to sign in.");
    } finally {
      submit.disabled = false;
    }
  });

  const base64UrlToBytes = (value) => {
    const padded = value.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat((4 - value.length % 4) % 4);
    const binary = atob(padded);
    return Uint8Array.from(binary, (character) => character.charCodeAt(0));
  };
  const bytesToBase64Url = (value) => {
    const bytes = new Uint8Array(value);
    let binary = "";
    bytes.forEach((byte) => { binary += String.fromCharCode(byte); });
    return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
  };
  const toBrowserCreationOptions = (options) => ({
    ...options,
    challenge: base64UrlToBytes(options.challenge),
    user: {...options.user, id: base64UrlToBytes(options.user.id)},
    excludeCredentials: (options.excludeCredentials || []).map((item) => ({
      ...item, id: base64UrlToBytes(item.id)
    }))
  });
  const toBrowserRequestOptions = (options) => ({
    ...options,
    challenge: base64UrlToBytes(options.challenge),
    allowCredentials: (options.allowCredentials || []).map((item) => ({
      ...item, id: base64UrlToBytes(item.id)
    }))
  });
  const credentialToJson = (credential) => {
    const response = {};
    [
      "attestationObject", "authenticatorData", "clientDataJSON",
      "signature", "userHandle"
    ].forEach((field) => {
      const value = credential.response[field];
      if (value instanceof ArrayBuffer) response[field] = bytesToBase64Url(value);
      else if (value !== undefined && value !== null) response[field] = value;
    });
    if (typeof credential.response.getTransports === "function") {
      response.transports = credential.response.getTransports();
    }
    return {
      id: credential.id,
      rawId: bytesToBase64Url(credential.rawId),
      type: credential.type,
      authenticatorAttachment: credential.authenticatorAttachment,
      response,
      clientExtensionResults: credential.getClientExtensionResults?.() || {}
    };
  };

  passkeyLogin.addEventListener("click", async () => {
    setError("");
    passkeyLogin.disabled = true;
    try {
      if (!window.PublicKeyCredential || !navigator.credentials) {
        throw new Error("This browser does not support passkeys. Use your password instead.");
      }
      const optionsResponse = await apiFetch("/auth/passkey/options", {method:"POST"});
      const options = await responsePayload(optionsResponse);
      if (!optionsResponse.ok) throw new Error(options.detail || "Passkey sign-in is unavailable.");
      const credential = await navigator.credentials.get({publicKey:toBrowserRequestOptions(options)});
      if (!credential) throw new Error("No passkey response was returned.");
      const verifyResponse = await apiFetch("/auth/passkey/verify", {
        method:"POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({credential:credentialToJson(credential)})
      });
      const payload = await responsePayload(verifyResponse);
      if (!verifyResponse.ok) throw new Error(payload.detail || "Passkey verification failed.");
      await startAuthenticatedSession();
    } catch (error) {
      setError(error instanceof Error ? error.message : "Passkey sign-in failed.");
    } finally {
      passkeyLogin.disabled = false;
    }
  });

  passkeyRegister.addEventListener("click", async () => {
    setPasskeyError("");
    passkeyRegister.disabled = true;
    try {
      if (!window.PublicKeyCredential || !navigator.credentials) {
        throw new Error("This browser does not support passkeys. Password sign-in remains available.");
      }
      const optionsResponse = await apiFetch("/auth/passkey/register/options", {method:"POST"});
      const options = await responsePayload(optionsResponse);
      if (!optionsResponse.ok) throw new Error(options.detail || "Could not prepare passkey registration.");
      const credential = await navigator.credentials.create({publicKey:toBrowserCreationOptions(options)});
      if (!credential) throw new Error("No passkey was created.");
      const verifyResponse = await apiFetch("/auth/passkey/register/verify", {
        method:"POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({credential:credentialToJson(credential)})
      });
      const payload = await responsePayload(verifyResponse);
      if (!verifyResponse.ok) throw new Error(payload.detail || "Passkey verification failed.");
      passkeyIsRegistered = true;
      passkeyLogin.hidden = false;
      passkeyState.textContent = "PASSKEY READY";
      passkeyRegister.textContent = "Passkey added on this device";
      setPasskeyError("Passkey registered. Future sign-ins on this browser can use your device authenticator.");
    } catch (error) {
      setPasskeyError(error instanceof Error ? error.message : "Passkey registration failed.");
    } finally {
      passkeyRegister.disabled = false;
    }
  });

  logoutButton.addEventListener("click", async () => {
    try { await apiFetch("/auth/logout", {method:"POST"}); } catch { /* sign out locally even if offline */ }
    passwordInput.value = "";
    chatLog.replaceChildren(safeNode("div","chat-empty","Sign in to start a private conversation."));
    setChatError("");
    showSignedOut("You have signed out.");
  });

  chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = chatInput.value.trim();
    if (!message || !providerReady || !activeSession) return;
    chatInput.value = "";
    setChatError("");
    appendMessage("user", message);
    sendChat.disabled = true;
    try {
      const response = await apiFetch("/brain/respond", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({message, mode:"assistant", session_id:sessionId})
      });
      const payload = await responsePayload(response);
      if (!response.ok) {
        if (response.status === 401) showSignedOut("Your session expired. Sign in again.");
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
    if (!activeSession || !providerReady) return;
    try {
      const response = await apiFetch("/brain/sessions/" + encodeURIComponent(sessionId), {method:"DELETE"});
      if (!response.ok) throw new Error("Could not clear the working-memory session.");
      sessionId = "control-room-" + (window.crypto?.randomUUID?.()
        || (Date.now() + "-" + Math.random().toString(16).slice(2)));
      chatLog.replaceChildren(safeNode("div","chat-empty","Session cleared. Start a new conversation."));
      setChatError("");
    } catch (error) {
      setChatError(error instanceof Error ? error.message : "Unable to clear the session.");
    }
  });

  const refresh = async () => {
    if (!activeSession) return;
    refreshButton.disabled = true;
    setError("");
    try {
      const response = await apiFetch("/monitor/overview?days=" + encodeURIComponent(simulationDay.value) + "&seed=7");
      const payload = await responsePayload(response);
      if (!response.ok) {
        if (response.status === 401) {
          showSignedOut("Your session expired. Sign in again.");
          return;
        }
        throw new Error(payload.detail || "Monitor API returned HTTP " + response.status + ".");
      }
      render(payload);
      await loadProviderStatus();
    } catch (error) {
      setConnection("failed", " CONNECTION ISSUE");
      setError(error instanceof Error ? error.message : "Unable to reach the monitor API.");
      if (!dashboard.hidden) refreshButton.disabled = false;
    } finally {
      refreshButton.disabled = false;
    }
  };

  const stopCamera = () => {
    if (cameraStream) {
      cameraStream.getTracks().forEach((track) => track.stop());
      cameraStream = null;
    }
    cameraVideo.pause();
    cameraVideo.srcObject = null;
    cameraVideo.hidden = true;
    cameraPlaceholder.hidden = false;
    cameraPlaceholder.textContent = "Camera is off";
    cameraCaptureButton.disabled = true;
    cameraStop.disabled = true;
    cameraStart.disabled = false;
    cameraState.textContent = "CAMERA OFF";
    if (capturedImageUrl) {
      URL.revokeObjectURL(capturedImageUrl);
      capturedImageUrl = null;
    }
    cameraCapture.removeAttribute("src");
    cameraCapture.hidden = true;
    setCameraError("");
  };
  cameraStart.addEventListener("click", async () => {
    setCameraError("");
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError("Camera access requires a supported browser and a secure HTTPS page.");
      return;
    }
    cameraStart.disabled = true;
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {facingMode:"user"}
      });
      cameraVideo.srcObject = cameraStream;
      cameraVideo.hidden = false;
      cameraPlaceholder.hidden = true;
      await cameraVideo.play();
      cameraCaptureButton.disabled = false;
      cameraStop.disabled = false;
      cameraState.textContent = "CAMERA ACTIVE · LOCAL ONLY";
    } catch (error) {
      stopCamera();
      setCameraError(error instanceof Error ? error.message : "Camera permission was denied or unavailable.");
    } finally {
      cameraStart.disabled = Boolean(cameraStream);
    }
  });
  cameraCaptureButton.addEventListener("click", () => {
    if (!cameraStream || !cameraVideo.videoWidth || !cameraVideo.videoHeight) return;
    const canvas = document.createElement("canvas");
    canvas.width = cameraVideo.videoWidth;
    canvas.height = cameraVideo.videoHeight;
    const context = canvas.getContext("2d");
    if (!context) {
      setCameraError("This browser cannot create a local camera snapshot.");
      return;
    }
    context.drawImage(cameraVideo, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (!blob) {
        setCameraError("The local camera snapshot could not be created.");
        return;
      }
      if (capturedImageUrl) URL.revokeObjectURL(capturedImageUrl);
      capturedImageUrl = URL.createObjectURL(blob);
      cameraCapture.src = capturedImageUrl;
      cameraCapture.hidden = false;
      setCameraError("Snapshot is only in this tab's memory. It has not been uploaded or stored.");
    }, "image/jpeg", 0.88);
    canvas.width = 0;
    canvas.height = 0;
  });
  cameraStop.addEventListener("click", stopCamera);

  simulationDay.addEventListener("input", () => {
    simulationDayValue.textContent = simulationDay.value + " / 280 days";
    if (activeSession) refresh();
  });
  refreshButton.addEventListener("click", refresh);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && activeSession) refresh();
  });
  window.addEventListener("pagehide", () => {
    stopCamera();
    providerReady = false;
    chatInput.disabled = true;
    sendChat.disabled = true;
    clearChat.disabled = true;
  });

  inspectSession();
})();