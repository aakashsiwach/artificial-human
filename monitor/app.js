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
  let apiBase = "";
  let bearerToken = "";
  let pollHandle = null;

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
    byId("last-updated").textContent = "UPDATED " + new Date(payload.observed_at || Date.now()).toLocaleString();
    dashboard.hidden = false;
    setConnection("connected", " CONNECTED");
    refreshButton.disabled = false;
  };

  const refresh = async () => {
    if (!apiBase || !bearerToken) return;
    refreshButton.disabled = true;
    setError("");
    try {
      const response = await fetch(apiBase + "/monitor/overview?days=280&seed=7", {
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
    } catch (error) {
      setConnection("failed", " CONNECTION ISSUE");
      setError(error instanceof Error ? error.message : "Unable to reach the monitor API.");
      if (!dashboard.hidden) refreshButton.disabled = false;
    } finally {
      refreshButton.disabled = false;
    }
  };

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
  });
})();
