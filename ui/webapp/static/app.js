/* Browser annotation UI: renders both PDFs as images with clickable diff overlays. */

const GAP = 12;
const RENDER_SCALES = [1, 1.5, 2, 3, 4];
const MIN_ZOOM = 0.25;
const MAX_ZOOM = 4;

const S = {
  state: null,
  diff: null,
  zoom: 1,
  autoFit: true,
  sync: true,
  selected: new Set(),
  changes: [],
  changeIdx: -1,
  searchHits: [],
  searchIdx: -1,
  panes: {},
};

const $ = (sel) => document.querySelector(sel);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function renderScaleFor(zoom) {
  const wanted = zoom * (window.devicePixelRatio || 1);
  return RENDER_SCALES.find((scale) => scale >= wanted) || RENDER_SCALES[RENDER_SCALES.length - 1];
}

/* ------------------------------------------------------------------ panes */

function initPanes() {
  for (const side of ["left", "right"]) {
    const root = $(`#pane-${side}`);
    const scroll = root.querySelector(".pane-scroll");
    const pane = {
      side,
      root,
      scroll,
      sizer: root.querySelector(".pane-sizer"),
      inner: root.querySelector(".pane-inner"),
      marquee: root.querySelector(".marquee"),
      fileName: root.querySelector(".file-name"),
      pages: [],
      docWidth: 0,
      docHeight: 0,
      observer: null,
    };
    S.panes[side] = pane;
    scroll.addEventListener("scroll", () => onPaneScroll(pane));
    attachMarquee(pane);
  }
}

function renderPane(side, data) {
  const pane = S.panes[side];
  pane.fileName.textContent = data.file_name;
  pane.inner.innerHTML = "";
  pane.pages = [];
  if (pane.observer) pane.observer.disconnect();

  const maxWidth = Math.max(...data.pages.map((p) => p.width));
  let y = GAP;

  data.pages.forEach((page, index) => {
    const el = document.createElement("div");
    el.className = "page pending";
    el.style.width = `${page.width}px`;
    el.style.height = `${page.height}px`;
    el.style.position = "absolute";
    el.style.left = `${(maxWidth - page.width) / 2}px`;
    el.style.top = `${y}px`;
    el.dataset.page = index;
    pane.inner.appendChild(el);
    pane.pages.push({ top: y, left: (maxWidth - page.width) / 2, width: page.width, height: page.height, el });
    y += page.height + GAP;
  });

  pane.docWidth = maxWidth;
  pane.docHeight = y;

  for (const group of data.groups) {
    for (const rect of group.rects) {
      const page = pane.pages[rect.page];
      if (!page) continue;
      const hl = document.createElement("div");
      hl.className = `hl hl-${group.diff_type}`;
      hl.dataset.gid = group.id;
      hl.style.left = `${rect.x0}px`;
      hl.style.top = `${rect.y0}px`;
      hl.style.width = `${Math.max(rect.x1 - rect.x0, 2)}px`;
      hl.style.height = `${Math.max(rect.y1 - rect.y0, 2)}px`;
      hl.addEventListener("click", (event) => {
        event.stopPropagation();
        toggleSelection(group.id);
      });
      page.el.appendChild(hl);
    }
  }

  pane.observer = new IntersectionObserver(
    (entries) => entries.forEach((entry) => entry.isIntersecting && loadPageImage(pane, Number(entry.target.dataset.page))),
    { root: pane.scroll, rootMargin: "400px 0px" }
  );
  pane.pages.forEach((page) => pane.observer.observe(page.el));

  pane.scroll.scrollTop = 0;
}

function loadPageImage(pane, index) {
  const page = pane.pages[index];
  if (!page) return;
  const scale = renderScaleFor(S.zoom);
  let img = page.el.querySelector("img");
  if (img && Number(img.dataset.scale) === scale) return;
  if (!img) {
    img = document.createElement("img");
    img.alt = `page ${index + 1}`;
    page.el.insertBefore(img, page.el.firstChild);
  }
  img.dataset.scale = scale;
  img.src = `/api/page/${pane.side}/${index}?scale=${scale}`;
  page.el.classList.remove("pending");
}

function applyZoom() {
  $("#zoom-label").textContent = `${Math.round(S.zoom * 100)}%`;
  for (const pane of Object.values(S.panes)) {
    pane.inner.style.transform = `scale(${S.zoom})`;
    pane.sizer.style.width = `${pane.docWidth * S.zoom}px`;
    pane.sizer.style.height = `${pane.docHeight * S.zoom}px`;
    pane.pages.forEach((page, index) => {
      if (page.el.querySelector("img")) loadPageImage(pane, index);
    });
  }
}

function setZoom(value) {
  S.zoom = Math.min(Math.max(value, MIN_ZOOM), MAX_ZOOM);
  applyZoom();
}

function fitToWidth() {
  const widest = Math.max(...Object.values(S.panes).map((pane) => pane.docWidth || 0));
  const narrowest = Math.min(...Object.values(S.panes).map((pane) => pane.scroll.clientWidth));
  if (widest > 0 && narrowest > 0) setZoom((narrowest - 18) / widest);
  S.autoFit = true;
}

function zoomBy(factor) {
  S.autoFit = false;
  setZoom(S.zoom * factor);
}

/* -------------------------------------------------------------- selection */

function allGroups() {
  if (!S.diff) return [];
  return [...S.diff.left.groups, ...S.diff.right.groups];
}

function groupById(id) {
  return allGroups().find((g) => g.id === id);
}

function toggleSelection(id) {
  if (S.selected.has(id)) S.selected.delete(id);
  else S.selected.add(id);
  refreshSelectionUI();
}

function refreshSelectionUI() {
  document.querySelectorAll(".hl").forEach((el) => el.classList.toggle("selected", S.selected.has(el.dataset.gid)));
  $("#selection-count").textContent = String(S.selected.size);

  const list = $("#selection-list");
  list.innerHTML = "";
  if (S.selected.size === 0) {
    list.innerHTML = '<li class="empty">Click a highlighted change, or drag a box over several of them.</li>';
    return;
  }

  for (const id of S.selected) {
    const group = groupById(id);
    if (!group) continue;
    const li = document.createElement("li");
    li.innerHTML = `
      <div class="meta">
        <span class="tag ${group.side === "left" ? "tag-del" : "tag-ins"}">${group.diff_type}</span>
        <span>${group.side === "left" ? "Original" : "Revised"} · p.${group.page_num + 1} · ${group.word_count}w</span>
        <button class="remove" title="Remove">&times;</button>
      </div>
      <div class="snippet"></div>`;
    li.querySelector(".snippet").textContent = group.text;
    li.querySelector(".remove").addEventListener("click", (event) => {
      event.stopPropagation();
      toggleSelection(id);
    });
    li.addEventListener("click", () => scrollToGroup(group));
    list.appendChild(li);
  }
}

function attachMarquee(pane) {
  let startX = 0;
  let startY = 0;
  let pressed = false;
  let dragging = false;

  const contentPos = (event) => {
    const rect = pane.scroll.getBoundingClientRect();
    return [event.clientX - rect.left + pane.scroll.scrollLeft, event.clientY - rect.top + pane.scroll.scrollTop];
  };

  // Without preventDefault the browser starts a native image drag and swallows mouseup.
  pane.scroll.addEventListener("dragstart", (event) => event.preventDefault());

  pane.scroll.addEventListener("mousedown", (event) => {
    if (event.button !== 0 || event.target.classList.contains("hl")) return;
    event.preventDefault();
    [startX, startY] = contentPos(event);
    pressed = true;
    dragging = false;
  });

  window.addEventListener("mousemove", (event) => {
    if (!pressed) return;
    const [x, y] = contentPos(event);
    if (!dragging && Math.abs(x - startX) < 5 && Math.abs(y - startY) < 5) return;
    dragging = true;
    pane.marquee.classList.remove("hidden");
    pane.marquee.style.left = `${Math.min(x, startX)}px`;
    pane.marquee.style.top = `${Math.min(y, startY)}px`;
    pane.marquee.style.width = `${Math.abs(x - startX)}px`;
    pane.marquee.style.height = `${Math.abs(y - startY)}px`;
  });

  window.addEventListener("mouseup", (event) => {
    if (!pressed) return;
    pressed = false;
    pane.marquee.classList.add("hidden");
    if (!dragging) return;
    dragging = false;

    const [endX, endY] = contentPos(event);
    selectInBox(
      pane,
      Math.min(startX, endX) / S.zoom,
      Math.min(startY, endY) / S.zoom,
      Math.max(startX, endX) / S.zoom,
      Math.max(startY, endY) / S.zoom
    );
  });
}

function selectInBox(pane, x0, y0, x1, y1) {
  const groups = S.diff ? S.diff[pane.side].groups : [];
  let added = 0;
  for (const group of groups) {
    const hit = group.rects.some((rect) => {
      const page = pane.pages[rect.page];
      if (!page) return false;
      const left = page.left + rect.x0;
      const top = page.top + rect.y0;
      const right = page.left + rect.x1;
      const bottom = page.top + rect.y1;
      return right >= x0 && left <= x1 && bottom >= y0 && top <= y1;
    });
    if (hit && !S.selected.has(group.id)) {
      S.selected.add(group.id);
      added += 1;
    }
  }
  if (added) refreshSelectionUI();
  else showBanner("No changes found in the selected area.", false, 2500);
}

/* ------------------------------------------------------------- navigation */

function buildChangeList() {
  S.changes = allGroups()
    .slice()
    .sort((a, b) => a.page_num - b.page_num || a.rects[0].y0 - b.rects[0].y0);
  S.changeIdx = -1;
  updateChangeCounter();
}

function updateChangeCounter() {
  const total = S.changes.length;
  const position = S.changeIdx >= 0 ? S.changeIdx + 1 : 0;
  $("#change-counter").textContent = total ? `${position}/${total} changes` : "no changes";
}

function stepChange(direction) {
  if (!S.changes.length) return;
  S.changeIdx = (S.changeIdx + direction + S.changes.length) % S.changes.length;
  scrollToGroup(S.changes[S.changeIdx], false);
  updateChangeCounter();
}

function scrollToGroup(group, updateIndex = true) {
  document.querySelectorAll(".hl.current").forEach((el) => el.classList.remove("current"));
  const pane = S.panes[group.side];
  pane.inner.querySelectorAll(`.hl[data-gid="${group.id}"]`).forEach((el) => el.classList.add("current"));
  scrollRectIntoView(group.side, group.rects[0]);

  if (updateIndex) {
    const index = S.changes.indexOf(group);
    if (index !== -1) S.changeIdx = index;
    updateChangeCounter();
  }
}

function scrollRectIntoView(side, rect) {
  const pane = S.panes[side];
  const page = pane.pages[rect.page];
  if (!page) return;
  const target = (page.top + rect.y0) * S.zoom - pane.scroll.clientHeight / 3;
  pane.scroll.scrollTo({ top: Math.max(target, 0), behavior: "smooth" });
}

/* ----------------------------------------------------------------- search */

function clearSearch(resetInput = false) {
  if (resetInput) $("#search-input").value = "";
  document.querySelectorAll(".search-hit").forEach((el) => el.remove());
  S.searchHits = [];
  S.searchIdx = -1;
  $("#search-count").textContent = "";
}

async function runSearch() {
  const query = $("#search-input").value.trim();
  clearSearch();
  if (!query) return;

  const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
  if (!response.ok) {
    showBanner("Search failed.", true, 3000);
    return;
  }
  const data = await response.json();

  for (const side of ["left", "right"]) {
    for (const match of data[side] || []) S.searchHits.push({ side, ...match });
  }
  S.searchHits.sort((a, b) => a.page - b.page || a.rects[0].y0 - b.rects[0].y0);

  for (const [index, hit] of S.searchHits.entries()) {
    const pane = S.panes[hit.side];
    for (const rect of hit.rects) {
      const page = pane.pages[rect.page];
      if (!page) continue;
      const el = document.createElement("div");
      el.className = "search-hit";
      el.dataset.hit = index;
      el.style.left = `${rect.x0}px`;
      el.style.top = `${rect.y0}px`;
      el.style.width = `${Math.max(rect.x1 - rect.x0, 2)}px`;
      el.style.height = `${Math.max(rect.y1 - rect.y0, 2)}px`;
      page.el.appendChild(el);
    }
  }

  updateSearchCount();
  if (S.searchHits.length) stepSearch(1);
  else $("#search-count").textContent = "0";
}

function updateSearchCount() {
  const total = S.searchHits.length;
  $("#search-count").textContent = total ? `${S.searchIdx + 1}/${total}` : "";
}

function stepSearch(direction) {
  if (!S.searchHits.length) return;
  S.searchIdx = (S.searchIdx + direction + S.searchHits.length) % S.searchHits.length;
  const hit = S.searchHits[S.searchIdx];
  document.querySelectorAll(".search-hit.current").forEach((el) => el.classList.remove("current"));
  document.querySelectorAll(`.search-hit[data-hit="${S.searchIdx}"]`).forEach((el) => el.classList.add("current"));
  scrollRectIntoView(hit.side, hit.rects[0]);
  updateSearchCount();
}

let syncing = false;
function onPaneScroll(pane) {
  if (!S.sync || syncing) return;
  const other = S.panes[pane.side === "left" ? "right" : "left"];
  const span = pane.scroll.scrollHeight - pane.scroll.clientHeight;
  const otherSpan = other.scroll.scrollHeight - other.scroll.clientHeight;
  if (span <= 0 || otherSpan <= 0) return;
  syncing = true;
  other.scroll.scrollTop = (pane.scroll.scrollTop / span) * otherSpan;
  requestAnimationFrame(() => {
    syncing = false;
  });
}

/* ------------------------------------------------------------------ state */

function showOverlay(text) {
  $("#overlay-text").textContent = text;
  $("#overlay").classList.remove("hidden");
}

function hideOverlay() {
  $("#overlay").classList.add("hidden");
}

let bannerTimer = null;
function showBanner(message, isError = false, timeout = 0) {
  const banner = $("#banner");
  banner.textContent = message;
  banner.classList.toggle("error", isError);
  banner.classList.remove("hidden");
  clearTimeout(bannerTimer);
  if (timeout) bannerTimer = setTimeout(() => banner.classList.add("hidden"), timeout);
}

function renderEntry(state) {
  $("#entry-counter").textContent = `Entry ${state.index + 1} of ${state.total}`;
  $("#progress-bar").style.width = `${(state.index / Math.max(state.total, 1)) * 100}%`;
  $("#reviewer-comment").textContent = state.entry.filtered_comment || "(empty)";
  $("#author-response").textContent = state.entry.filtered_response || "(empty)";
  $("#output-path").textContent = `Output: ${state.output_path}`;
  const link = $("#openreview-link");
  link.href = state.openreview_url;
  link.textContent = `OpenReview: ${state.paper_id}`;
}

async function refreshState() {
  S.state = await (await fetch("/api/state")).json();

  if (S.state.done) {
    $("#entry-counter").textContent = `${S.state.total} of ${S.state.total} entries`;
    $("#progress-bar").style.width = "100%";
    ["btn-save", "btn-skip", "btn-next", "btn-prev"].forEach((id) => ($(`#${id}`).disabled = true));
    showOverlay(`All ${S.state.total} entries processed. Results written to ${S.state.output_path}`);
    $("#overlay").querySelector(".spinner").classList.add("hidden");
    return;
  }

  renderEntry(S.state);
  await loadDiff();
}

async function loadDiff() {
  showOverlay("Computing diff for this paper…");
  while (true) {
    const response = await fetch("/api/diff");
    if (response.status === 200) {
      S.diff = (await response.json()).diff;
      break;
    }
    if (response.status >= 400 && response.status !== 409) {
      const body = await response.json().catch(() => ({}));
      hideOverlay();
      showBanner(`Diff failed: ${body.error || response.statusText}`, true);
      return;
    }
    await sleep(500);
  }

  hideOverlay();
  renderPane("left", S.diff.left);
  renderPane("right", S.diff.right);
  fitToWidth();
  clearSearch(true);
  S.selected.clear();
  refreshSelectionUI();
  buildChangeList();
  $("#banner").classList.add("hidden");
}

async function saveEntry() {
  if (S.selected.size === 0 && !confirm("No changes selected. Record this action item with an empty diff list?")) return;
  $("#btn-save").disabled = true;
  try {
    await fetch("/api/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ group_ids: [...S.selected] }),
    });
    await refreshState();
  } finally {
    $("#btn-save").disabled = false;
  }
}

async function skipEntry() {
  await fetch("/api/skip", { method: "POST" });
  await refreshState();
}

/* --------------------------------------------------------------- bindings */

function bindControls() {
  $("#btn-next").addEventListener("click", () => stepChange(1));
  $("#btn-prev").addEventListener("click", () => stepChange(-1));
  $("#btn-zoom-in").addEventListener("click", () => zoomBy(1.2));
  $("#btn-zoom-out").addEventListener("click", () => zoomBy(1 / 1.2));
  $("#btn-zoom-fit").addEventListener("click", fitToWidth);
  $("#btn-save").addEventListener("click", saveEntry);
  $("#btn-skip").addEventListener("click", skipEntry);
  $("#btn-clear").addEventListener("click", () => {
    S.selected.clear();
    refreshSelectionUI();
  });
  $("#chk-sync").addEventListener("change", (event) => (S.sync = event.target.checked));

  $("#btn-search-next").addEventListener("click", () => stepSearch(1));
  $("#btn-search-prev").addEventListener("click", () => stepSearch(-1));
  $("#search-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      if (S.searchHits.length) stepSearch(event.shiftKey ? -1 : 1);
      else runSearch();
    } else if (event.key === "Escape") {
      clearSearch(true);
      event.target.blur();
    }
  });
  $("#search-input").addEventListener("change", runSearch);

  document.addEventListener("keydown", (event) => {
    if (event.key === "F3") {
      event.preventDefault();
      stepSearch(event.shiftKey ? -1 : 1);
      return;
    }
    if (event.target.matches("input, textarea") || event.ctrlKey || event.metaKey) return;
    if (event.key === "/") {
      event.preventDefault();
      $("#search-input").focus();
      return;
    }
    const key = event.key.toLowerCase();
    if (key === "n") stepChange(1);
    else if (key === "p") stepChange(-1);
    else if (key === "s") saveEntry();
    else if (key === "escape") {
      S.selected.clear();
      refreshSelectionUI();
      clearSearch(true);
    } else if (key === "+" || key === "=") zoomBy(1.2);
    else if (key === "-") zoomBy(1 / 1.2);
    else if (key === "f") fitToWidth();
    else return;
    event.preventDefault();
  });

  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => S.autoFit && fitToWidth(), 150);
  });
}

initPanes();
bindControls();
refreshState();
