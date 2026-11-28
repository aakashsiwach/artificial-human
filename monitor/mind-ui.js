(() => {
  "use strict";
  const byId = (id) => document.getElementById(id);
  const dashboard = byId("dashboard");
  const publicHome = byId("public-home");
  const brainVisual = byId("brain-visual");
  const publicHost = byId("public-brain-host");
  const dashboardHost = byId("brain-view-host");
  const loginPanel = byId("login-panel");
  const loginScrim = byId("login-scrim");
  const loginToggle = byId("owner-login-toggle");
  const refreshButton = byId("refresh-button");

  const moduleSets = {
    active: [
      ["Continuous in-memory runtime","One composition-owned loop advances without HTTP requests or language-model calls.","runtime.py · bounded RAM"],
      ["12-neuron developmental LIF substrate","A deterministic spiking substrate with inspectable membrane and spike state.","development/program.py"],
      ["Excitatory and inhibitory synapses","Delayed signed transmission with bounded event queues and currents.","neural/simulation.py"],
      ["STDP-inspired synaptic plasticity","Local timing-dependent parameter updates are bounded and source-defined.","development/program.py"],
      ["Homeostatic regulation","Activity regulation adapts internal gain within defined limits.","homeostatic controller"],
      ["Connectivity growth and pruning","Stage-gated candidate synapses and functional-use pruning are implemented.","development/program.py"],
      ["Synthetic sensory event path","Explicit synthetic modalities enter the organism loop; real sensors are not connected.","organism_loop.py"],
      ["Attention and working memory","Bounded selection and a seven-item volatile workspace guide the loop.","cognition · organism loop"],
      ["Episodic engram circuit","Sparse cue binding, partial completion and interference controls are available.","neural/episodic.py"],
      ["Prediction and feedback","The simulated world model compares expected and observed outcomes.","organism_loop.py"],
      ["Learned action selection","A fixed allowlist selects actions for the simulated body.","action selection · simulation"],
      ["Internal scheduling and resource limits","Event backpressure, tick budgets, health and bounded traces are enforced.","runtime.py · metabolism"],
      ["Simulated body and environment","A bounded scalar environment closes the prediction/action/feedback loop.","organism_loop.py"]
    ],
    partial: [
      ["Owner-scoped long-term memory","Consent-aware volatile records and hybrid retrieval exist, but do not reproduce a complete human memory system.","LIMITED MECHANISM"],
      ["Sleep-inspired consolidation","Replay-derived proposals require owner review; this is not an autonomous sleep cycle.","OWNER REVIEW"],
      ["Structured schema induction","Repeated owner-annotated event patterns can produce reviewable schema proposals.","OWNER REVIEW"],
      ["Text language interface","An optional replaceable model adapter handles request text; it is not the mind's foundation.","OPTIONAL PROVIDER"],
      ["Camera preview","The browser can preview locally, but frames do not enter the organism's neural processing.","LOCAL PREVIEW ONLY"],
      ["Safe self-development","Operator-injected Linux Bubblewrap runs allowlisted checks in a networkless namespace; independent security review remains outstanding.","OPT-IN · REVIEW REQUIRED"]
    ],
    dormant: [
      ["Live visual perception","No camera frames are encoded into the organism's neural state.","NOT CONNECTED"],
      ["Auditory perception","No microphone input or audio encoder is integrated.","NOT BUILT"],
      ["Physical tactile and vestibular sensing","Current signals are synthetic fixtures, not physical sensor streams.","NOT CONNECTED"],
      ["Affective and emotional appraisal","No integrated emotion or affective-state model is implemented.","NOT BUILT"],
      ["Intrinsic needs and motivation","No autonomous biological-style drive system is implemented.","NOT BUILT"],
      ["Human autobiographical memory","Current episodic mechanisms are bounded software models, not a human life-history system.","NOT ESTABLISHED"],
      ["Physical embodiment and motor control","The body is a one-dimensional simulator; no physical actuator is connected.","SIMULATION ONLY"],
      ["Speech and voice generation","There is no speech synthesis or voice interface in the active brain.","NOT BUILT"],
      ["Autonomous source-code rewriting","The organism cannot rewrite, adopt or deploy its own executable source.","DISABLED BY DESIGN"],
      ["Whole-brain emulation or consciousness","No complete human-brain model or evidence of subjective experience exists.","NOT IMPLEMENTED"]
    ]
  };

  Object.entries(moduleSets).forEach(([kind, items]) => {
    const root = byId(kind + "-module-grid");
    if (!root) return;
    items.forEach((item) => {
      const card = document.createElement("article");
      card.className = "module-card " + (kind === "active" ? "active" : kind === "partial" ? "partial" : "dormant");
      const dot = document.createElement("span");
      dot.className = "module-indicator";
      dot.setAttribute("aria-hidden","true");
      const copy = document.createElement("div");
      copy.className = "module-copy";
      const title = document.createElement("b");
      title.textContent = item[0];
      const description = document.createElement("p");
      description.textContent = item[1];
      const meta = document.createElement("div");
      meta.className = "module-meta";
      const status = document.createElement("span");
      status.textContent = kind === "active" ? "ACTIVE" : kind === "partial" ? "PARTIAL" : "DORMANT";
      const evidence = document.createElement("span");
      evidence.textContent = item[2];
      meta.append(status,evidence);
      copy.append(title,description,meta);
      card.append(dot,copy);
      root.append(card);
    });
    byId(kind + "-count").textContent = items.length;
  });

  function selectView(name) {
    document.querySelectorAll(".view-tab").forEach((tab) => {
      const selected = tab.dataset.view === name;
      tab.classList.toggle("is-active",selected);
      tab.setAttribute("aria-selected",String(selected));
      tab.tabIndex = selected ? 0 : -1;
    });
    document.querySelectorAll(".view-panel").forEach((panel) => {
      panel.hidden = panel.id !== name + "-view";
    });
    if (refreshButton) refreshButton.hidden = name !== "developing";
  }

  document.querySelectorAll(".view-tab").forEach((tab) => {
    tab.addEventListener("click",() => selectView(tab.dataset.view));
    tab.addEventListener("keydown",(event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      const tabs = Array.from(document.querySelectorAll(".view-tab"));
      const index = tabs.indexOf(tab);
      const next = tabs[(index + (event.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
      next.focus();
      selectView(next.dataset.view);
    });
  });

  function syncSurface() {
    const signedIn = dashboard && !dashboard.hidden;
    publicHome.hidden = signedIn;
    loginToggle.hidden = signedIn;
    loginPanel.hidden = true;
    loginScrim.hidden = true;
    if (refreshButton) refreshButton.hidden = true;
    if (brainVisual) {
      const desiredHost = signedIn ? dashboardHost : publicHost;
      if (brainVisual.parentElement !== desiredHost) desiredHost.appendChild(brainVisual);
    }
    if (signedIn) selectView("brain");
  }
  const observer = new MutationObserver(syncSurface);
  observer.observe(dashboard,{attributes:true,attributeFilter:["hidden"]});
  syncSurface();

  function openLogin(open) {
    loginPanel.hidden = !open;
    loginScrim.hidden = !open;
    if (open) {
      const input = byId("login-username");
      if (input) window.setTimeout(() => input.focus(),60);
    }
  }
  loginToggle.addEventListener("click",() => openLogin(loginPanel.hidden));
  loginScrim.addEventListener("click",() => openLogin(false));
  document.addEventListener("keydown",(event) => {
    if (event.key === "Escape" && !loginPanel.hidden) openLogin(false);
  });

  let traceItems = [];
  let traceCursor = 0;
  function addTrace(kind,text,meta) {
    const row = document.createElement("div");
    row.className = "trace-line";
    const label = document.createElement("span");
    label.className = "trace-kind";
    label.textContent = kind;
    const value = document.createElement("span");
    value.className = "trace-text";
    value.textContent = text;
    const tail = document.createElement("span");
    tail.className = "trace-meta";
    tail.textContent = meta || "";
    row.append(label,value,tail);
    return row;
  }
  function showTraceItem() {
    const stream = byId("code-stream");
    if (!stream || traceItems.length === 0) return;
    if (stream.querySelector(".trace-empty")) stream.replaceChildren();
    const item = traceItems[traceCursor % traceItems.length];
    stream.append(addTrace(item[0],item[1],item[2]));
    traceCursor += 1;
    while (stream.children.length > 9) stream.removeChild(stream.firstElementChild);
    stream.scrollTop = stream.scrollHeight;
  }
  function updateSnapshot(payload) {
    const organism = payload && payload.organism ? payload.organism : {};
    const runtime = organism.continuous_runtime || {};
    const loop = runtime.organism || {};
    const neural = loop.neural || {};
    const working = loop.working_memory || {};
    const action = loop.action_selection || {};
    const associations = organism.associative_memory || {};
    traceItems = [
      ["LOOP","ContinuousOrganismRuntime · " + (runtime.state || "snapshot"),"tick " + (runtime.tick_count ?? "—")],
      ["NEURAL","DevelopmentalProgram · " + (neural.neuron_count ?? "—") + " neurons / " + (neural.synapse_count ?? "—") + " synapses",(neural.total_spikes ?? 0) + " spikes"],
      ["LEARN","Experience events · " + (neural.experience_count ?? 0),"bounded plasticity"],
      ["SIGNAL","Processed events · " + (runtime.processed_events ?? 0),"pending " + (runtime.pending_events ?? 0)],
      ["MEMORY","Synthetic episodes · " + (loop.episodes ?? 0),"process RAM"],
      ["WORKING","Workspace items · " + (working.item_count ?? "bounded"),"volatile state"],
      ["ACTION","Selection updates · " + (action.updates ?? 0),"fixed allowlist"],
      ["ASSOC","Associative links · " + (associations.association_count ?? 0),"co-activations " + (associations.coactivations ?? 0)],
      ["HEALTH","Runtime health · " + (runtime.health || "unknown"),"dropped " + (runtime.dropped_events ?? 0)]
    ];
    byId("brain-runtime-state").textContent = "RUNTIME " + String(runtime.state || "SNAPSHOT").toUpperCase() + " · " + String(runtime.health || "STATUS").toUpperCase();
    byId("brain-tick").textContent = "TICK " + (runtime.tick_count ?? "—");
    const stream = byId("code-stream");
    stream.replaceChildren();
    traceCursor = 0;
    for (let i=0;i<Math.min(6,traceItems.length);i++) showTraceItem();
  }
  window.addEventListener("organism:snapshot",(event) => updateSnapshot(event.detail || {}));
  window.setInterval(showTraceItem,1600);

  document.querySelectorAll(".copy-command").forEach((button) => {
    button.addEventListener("click",async () => {
      const command = button.dataset.copy || "";
      const feedback = byId("copy-feedback");
      try {
        if (!navigator.clipboard || !window.isSecureContext) throw new Error("clipboard unavailable");
        await navigator.clipboard.writeText(command);
        feedback.textContent = "Copied: " + command;
      } catch {
        feedback.textContent = "Clipboard unavailable. Select and copy manually: " + command;
      }
    });
  });
})();