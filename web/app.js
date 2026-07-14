// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

// tt-kernel community catalog — a pure client-side index over the Hugging Face Hub.
// Nothing is stored or proxied by the host: the visitor's browser talks to the HF public
// API directly. Bundles remain under their authors' governance; we only show pointers.

(() => {
  "use strict";
  const CFG = window.TTK_CONFIG;
  const HF = CFG.HF_ORIGIN.replace(/\/+$/, "");

  // ---- DOM refs ----
  const $grid = document.getElementById("grid");
  const $status = document.getElementById("status");
  const $search = document.getElementById("search");
  const $sort = document.getElementById("sort");
  const $archFilters = document.getElementById("arch-filters");
  const $capFilters = document.getElementById("cap-filters");
  const $filters = document.getElementById("filters");
  const $drawer = document.getElementById("drawer");
  const $drawerBody = document.getElementById("drawer-body");

  document.getElementById("brand-name").textContent = CFG.BRAND;
  document.getElementById("brand-tagline").textContent = CFG.TAGLINE;
  document.title = CFG.BRAND;

  // ---- state ----
  let models = []; // {id, owner, name, downloads, likes, modified, tags, arch, manifest, enriched}
  const state = {
    q: "",
    sort: "downloads",
    arches: new Set(), // active arch filters
    caps: new Set(), // active capability filters (display labels, e.g. "MoE")
    features: new Set(), // active feature filters: runner
  };

  const esc = (s) =>
    String(s ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
    );

  const fmtNum = (n) => {
    if (n == null) return "—";
    if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(1) + "k";
    return String(n);
  };

  const fmtDate = (s) => {
    if (!s) return "";
    const d = new Date(s);
    if (isNaN(d)) return "";
    const days = Math.floor((Date.now() - d.getTime()) / 86400000);
    if (days <= 0) return "today";
    if (days === 1) return "yesterday";
    if (days < 30) return days + "d ago";
    if (days < 365) return Math.floor(days / 30) + "mo ago";
    return Math.floor(days / 365) + "y ago";
  };

  const archOf = (tags) =>
    (tags || []).find((t) => CFG.KNOWN_ARCHES.includes(t)) || null;

  // Distinct capability display labels a repo's tags map to (deduped across aliases).
  const capsOf = (tags) => {
    const labels = [];
    for (const t of tags || []) {
      const label = CFG.CAPABILITIES[String(t).toLowerCase()];
      if (label && !labels.includes(label)) labels.push(label);
    }
    return labels;
  };

  // ---- load the catalog list ----
  async function loadList() {
    const url =
      `${HF}/api/models?filter=${encodeURIComponent(CFG.CATALOG_TAG)}` +
      `&sort=downloads&direction=-1&limit=${CFG.LIST_LIMIT}&full=true`;
    let data;
    try {
      const res = await fetch(url, { headers: { Accept: "application/json" } });
      if (!res.ok) throw new Error(`HF API returned ${res.status}`);
      data = await res.json();
    } catch (err) {
      $status.className = "status error";
      $status.innerHTML =
        `Could not reach the Hugging Face API (${esc(err.message)}). ` +
        `This site indexes HF live from your browser; check your connection or CORS. ` +
        `The <code>${esc(CFG.CATALOG_TAG)}</code> tag is what bundles opt into via ` +
        `<code>tt-kernel push --publish</code>.`;
      return;
    }

    models = data.map((m) => {
      const id = m.id || m.modelId;
      const [owner, ...rest] = id.split("/");
      return {
        id,
        owner,
        name: rest.join("/") || id,
        downloads: m.downloads ?? null,
        likes: m.likes ?? null,
        modified: m.lastModified || m.createdAt || null,
        tags: m.tags || [],
        arch: archOf(m.tags),
        caps: capsOf(m.tags),
        manifest: null,
        enriched: false,
      };
    });

    if (!models.length) {
      $status.textContent =
        "No bundles are listed yet. Publish one with `tt-kernel push … --publish`.";
      return;
    }
    buildArchFilters();
    buildCapFilters();
    render();
    enrichAll();
  }

  function buildArchFilters() {
    const arches = [...new Set(models.map((m) => m.arch).filter(Boolean))].sort();
    $archFilters.innerHTML = arches
      .map((a) => `<button class="chip" data-arch="${esc(a)}">${esc(a)}</button>`)
      .join("");
  }

  function buildCapFilters() {
    const caps = [...new Set(models.flatMap((m) => m.caps))].sort();
    $capFilters.innerHTML = caps
      .map((c) => `<button class="chip" data-cap="${esc(c)}">${esc(c)}</button>`)
      .join("");
  }

  // ---- filtering / sorting ----
  function visible() {
    let out = models.filter((m) => {
      if (state.q) {
        const hay = (m.id + " " + (m.manifest?.weights?.repo_id || "")).toLowerCase();
        if (!hay.includes(state.q)) return false;
      }
      if (state.arches.size && !state.arches.has(m.arch)) return false;
      for (const c of state.caps) {
        if (!m.caps.includes(c)) return false;
      }
      for (const f of state.features) {
        if (!m.enriched) return false; // can't confirm the feature yet
        const man = m.manifest;
        if (f === "runner" && !man?.runner) return false;
      }
      return true;
    });
    out.sort((a, b) => {
      if (state.sort === "name") return a.id.localeCompare(b.id);
      if (state.sort === "modified")
        return new Date(b.modified || 0) - new Date(a.modified || 0);
      return (b.downloads || 0) - (a.downloads || 0);
    });
    return out;
  }

  function render() {
    const list = visible();
    $status.className = "status";
    const total = models.length;
    $status.textContent =
      list.length === total
        ? `${total} bundle${total === 1 ? "" : "s"}`
        : `${list.length} of ${total} bundles`;

    $grid.innerHTML = list.map(cardHTML).join("");
  }

  function cardHTML(m) {
    const man = m.manifest;
    const badges = [];
    if (m.arch) badges.push(`<span class="badge arch">${esc(m.arch)}</span>`);
    for (const c of m.caps) badges.push(`<span class="badge cap">${esc(c)}</span>`);
    if (m.enriched && man) {
      if (man.runner) badges.push(`<span class="badge green">runner</span>`);
      if (man.tt_metal_version)
        badges.push(`<span class="badge">tt-metal ${esc(man.tt_metal_version)}</span>`);
    } else if (!m.enriched) {
      badges.push(`<span class="badge pending">…</span>`);
    }
    const kernels =
      m.enriched && man?.kernel_count != null
        ? `${man.kernel_count} kernel group${man.kernel_count === 1 ? "" : "s"} · `
        : "";
    return `
      <article class="card" data-id="${esc(m.id)}">
        <div class="card-top">
          <div class="card-name"><span class="card-owner">${esc(m.owner)}/</span>${esc(m.name)}</div>
          <div class="card-dl">↓ ${fmtNum(m.downloads)}</div>
        </div>
        <div class="badges">${badges.join("")}</div>
        ${cmd(`tt-kernel pull ${m.id}`, "cmd-primary")}
        <div class="card-meta">${kernels}updated ${esc(fmtDate(m.modified))}</div>
      </article>`;
  }

  // ---- background enrichment: fetch each manifest with limited concurrency ----
  async function fetchManifest(id) {
    const url = `${HF}/${id}/resolve/main/${CFG.MANIFEST_NAME}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(String(res.status));
    return res.json();
  }

  async function enrichAll() {
    const queue = models.slice();
    let active = 0;
    let dirty = false;
    const scheduleRender = (() => {
      let t = null;
      return () => {
        dirty = true;
        if (t) return;
        t = setTimeout(() => {
          t = null;
          if (dirty) {
            dirty = false;
            render();
          }
        }, 250);
      };
    })();

    return new Promise((resolve) => {
      const pump = () => {
        if (!queue.length && active === 0) return resolve();
        while (active < CFG.ENRICH_CONCURRENCY && queue.length) {
          const m = queue.shift();
          active++;
          fetchManifest(m.id)
            .then((man) => { m.manifest = man; if (!m.arch) m.arch = man.arch || null; })
            .catch(() => { m.manifest = null; })
            .finally(() => {
              m.enriched = true;
              active--;
              scheduleRender();
              pump();
            });
        }
      };
      pump();
    });
  }

  // ---- detail drawer ----
  async function openDrawer(id) {
    const m = models.find((x) => x.id === id);
    if (!m) return;
    $drawer.hidden = false;
    document.body.style.overflow = "hidden";
    $drawerBody.innerHTML =
      `<h2 class="d-title" id="d-title">${esc(m.name)}</h2>` +
      `<p class="d-sub">${esc(m.owner)}</p>` +
      `<p class="d-loading">Loading manifest…</p>`;

    let man = m.manifest;
    if (!m.enriched) {
      try { man = await fetchManifest(id); m.manifest = man; m.enriched = true; }
      catch { man = null; m.enriched = true; }
    }
    $drawerBody.innerHTML = drawerHTML(m, man);
  }

  function cmd(text, extra = "") {
    return (
      `<div class="cmd ${extra}">${esc(text)}` +
      `<button class="copy-btn" data-copy="${esc(text)}">copy</button></div>`
    );
  }

  function drawerHTML(m, man) {
    const repoUrl = `${HF}/${m.id}`;
    const kv = [];
    const row = (k, v) => kv.push(`<dt>${esc(k)}</dt><dd>${v}</dd>`);
    row("Owner", esc(m.owner));
    row("Downloads", fmtNum(m.downloads));
    row("Updated", esc(fmtDate(m.modified)) || "—");
    if (man) {
      row("Arch", esc(man.arch || m.arch || "—"));
      row("tt-metal", esc(man.tt_metal_version || "—"));
      row("Kernel groups", man.kernel_count ?? "—");
      row("Devices", man.device_count ?? "—");
      row("build_key", man.build_key != null ? `<code>${esc(man.build_key)}</code>` : "—");
      row("Runner", man.runner
        ? `<code>${esc(man.runner.spec)}</code>${man.runner.wheels?.length ? " (packaged)" : " (reference)"}`
        : "none");
      if (man.weights) row("Target model", `<a href="${HF}/${esc(man.weights.repo_id)}" target="_blank" rel="noopener">${esc(man.weights.repo_id)}</a>`);
    }
    if (m.caps.length) row("Capabilities", m.caps.map((c) => esc(c)).join(", "));

    const notManifest = !man
      ? `<p class="d-loading">No <code>${esc(CFG.MANIFEST_NAME)}</code> found in this repo — ` +
        `it may not be a tt-kernel bundle, or the repo went private.</p>`
      : "";

    const runnable = man && (man.runner || man.weights);

    return `
      <h2 class="d-title" id="d-title">${esc(m.name)}</h2>
      <p class="d-sub">${esc(m.owner)}</p>
      ${notManifest}
      <div class="d-hero">
        <h3>Pull it</h3>
        ${cmd(`tt-kernel pull ${m.id}`, "cmd-primary")}
        <p class="hero-note">Installs the precompiled kernels${runnable ? " + runner + weights" : ""}.
        <code>pull</code> checks compatibility and refuses a mismatched tt-metal build.</p>
      </div>
      <div class="d-section">
        <h3>Details</h3>
        <dl class="kv">${kv.join("")}</dl>
      </div>
      <div class="d-section">
        <h3>Other commands</h3>
        ${cmd(`tt-kernel info ${m.id}`)}
        ${runnable ? cmd(`tt-kernel run ${m.id}`) : ""}
      </div>
      <p class="d-source">Source: <a href="${repoUrl}" target="_blank" rel="noopener">${esc(m.id)} on Hugging Face ↗</a> · governed solely by its author.</p>`;
  }

  async function doCopy(btn) {
    try {
      await navigator.clipboard.writeText(btn.dataset.copy);
    } catch {
      // Fallback for non-secure contexts (plain http:// on a LAN box).
      const ta = document.createElement("textarea");
      ta.value = btn.dataset.copy;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand("copy"); } catch { /* give up */ }
      ta.remove();
    }
    const old = btn.textContent;
    btn.textContent = "copied ✓";
    setTimeout(() => (btn.textContent = old), 1200);
  }

  function closeDrawer() {
    $drawer.hidden = true;
    document.body.style.overflow = "";
  }

  // ---- events ----
  let searchTimer = null;
  $search.addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      state.q = $search.value.trim().toLowerCase();
      render();
    }, 120);
  });

  $sort.addEventListener("change", () => { state.sort = $sort.value; render(); });

  $filters.addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (!chip) return;
    if (chip.dataset.arch) toggleSet(state.arches, chip.dataset.arch, chip);
    else if (chip.dataset.cap) toggleSet(state.caps, chip.dataset.cap, chip);
    else if (chip.dataset.feature) toggleSet(state.features, chip.dataset.feature, chip);
    render();
  });

  function toggleSet(set, key, chip) {
    if (set.has(key)) { set.delete(key); chip.setAttribute("aria-pressed", "false"); }
    else { set.add(key); chip.setAttribute("aria-pressed", "true"); }
  }

  // Single delegated copy handler for cards + drawer (a copy click never opens the drawer).
  document.addEventListener("click", (e) => {
    const btn = e.target.closest(".copy-btn");
    if (!btn) return;
    e.stopPropagation();
    e.preventDefault();
    doCopy(btn);
  });

  $grid.addEventListener("click", (e) => {
    if (e.target.closest(".copy-btn")) return; // handled above; don't open the drawer
    const card = e.target.closest(".card");
    if (card) openDrawer(card.dataset.id);
  });

  $drawer.addEventListener("click", (e) => {
    if (e.target.hasAttribute("data-close")) closeDrawer();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !$drawer.hidden) closeDrawer();
  });

  loadList();
})();
