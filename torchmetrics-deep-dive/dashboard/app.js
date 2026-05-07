/* ============================================================
 *  TorchMetrics Deep Dive — Mastery Dashboard logic
 *
 *  Modes:
 *    read    — topic summary
 *    quiz    — sequential drill-down questions for current topic
 *    flash   — flashcards for current topic, with star confidence
 *  Global modes (sidebar):
 *    random         — random question from anywhere
 *    flashcards-all — flashcards across all topics, weak-first
 *    mastery        — heat-map of confidence across all questions
 *
 *  Persistence (localStorage key "tm-confidence"):
 *    {
 *      "topic-id/q-idx": { stars: 0-5, lastSeen: ISO date, count: int }
 *    }
 * ============================================================ */
(function () {
  const DATA = window.TM_DATA;
  const STORAGE_KEY = "tm-confidence";
  let currentTopicId = null;
  let currentMode = "read";          // read | quiz | flash
  let currentGlobalMode = null;       // random | flashcards-all | mastery | null
  let revealState = {};               // topicId -> { qIdx -> { revealed, followups: { key: { revealed, shown } } } }
  let quizQuestionIndex = 0;
  let flashIndex = 0;
  let flashAnswerShown = false;
  let randomQueue = [];               // shuffled list of {topicId, qIdx}
  let allFlashQueue = [];             // weak-first list of {topicId, qIdx}

  /* ---------- Persistence ---------- */
  function loadConfidence() {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}; }
    catch (e) { return {}; }
  }
  function saveConfidence(c) {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(c)); } catch (e) {}
  }
  function getStars(topicId, qIdx) {
    const c = loadConfidence();
    return c[`${topicId}/${qIdx}`]?.stars ?? 0;
  }
  function setStars(topicId, qIdx, stars) {
    const c = loadConfidence();
    c[`${topicId}/${qIdx}`] = {
      stars,
      lastSeen: new Date().toISOString(),
      count: (c[`${topicId}/${qIdx}`]?.count || 0) + 1,
    };
    saveConfidence(c);
  }
  function topicAvgStars(topicId) {
    const t = DATA.topics.find((x) => x.id === topicId);
    if (!t || !t.questions.length) return 0;
    const c = loadConfidence();
    let sum = 0, n = 0;
    t.questions.forEach((_, i) => {
      const r = c[`${topicId}/${i}`]?.stars;
      if (r) { sum += r; n += 1; }
    });
    return n === 0 ? 0 : sum / n;
  }
  function masteryStats() {
    const total = DATA.topics.reduce((acc, t) => acc + t.questions.length, 0);
    const c = loadConfidence();
    const counts = [0, 0, 0, 0, 0, 0]; // index = stars (0..5)
    DATA.topics.forEach((t) => {
      t.questions.forEach((_, i) => {
        const s = c[`${t.id}/${i}`]?.stars ?? 0;
        counts[s] += 1;
      });
    });
    return { total, counts };
  }

  /* ---------- Sidebar ---------- */
  function renderSidebar() {
    const nav = document.getElementById("nav");
    nav.innerHTML = "";
    DATA.categories.forEach((cat) => {
      const topics = DATA.topics.filter((t) => t.category === cat);
      if (topics.length === 0) return;
      const catEl = document.createElement("div");
      catEl.className = "nav-cat";
      catEl.textContent = cat;
      nav.appendChild(catEl);
      topics.forEach((topic) => {
        const item = document.createElement("div");
        item.className = "nav-item";
        item.dataset.topicId = topic.id;
        const avg = topicAvgStars(topic.id);
        const lvl = avg ? Math.round(avg) : 0;
        item.innerHTML = `
          <span>${topic.title}${avg ? `<span class="conf-pip lvl-${lvl}" title="avg ${avg.toFixed(1)}★"></span>` : ""}</span>
          <span class="nav-q-count">${topic.questions.length}Q</span>
        `;
        item.addEventListener("click", () => selectTopic(topic.id));
        nav.appendChild(item);
      });
    });
  }
  function highlightActiveNav() {
    document.querySelectorAll(".nav-item").forEach((el) => {
      el.classList.toggle("active", el.dataset.topicId === currentTopicId && !currentGlobalMode);
    });
    document.querySelectorAll(".gm-btn").forEach((el) => {
      el.classList.toggle("active", el.dataset.gmode === currentGlobalMode);
    });
  }

  /* ---------- Topic selection & mode ---------- */
  function selectTopic(topicId) {
    currentGlobalMode = null;
    currentTopicId = topicId;
    quizQuestionIndex = 0;
    flashIndex = 0;
    flashAnswerShown = false;
    if (!revealState[topicId]) revealState[topicId] = {};
    document.getElementById("welcome").classList.add("hidden");
    document.getElementById("topic-modes").classList.remove("hidden");
    highlightActiveNav();
    updateBreadcrumbs();
    renderCurrentView();
  }
  function updateBreadcrumbs() {
    const topic = DATA.topics.find((t) => t.id === currentTopicId);
    if (currentGlobalMode === "random") {
      document.getElementById("crumb-cat").textContent = "🎲 Global";
      document.getElementById("crumb-topic").textContent = "Random Mix";
      return;
    }
    if (currentGlobalMode === "flashcards-all") {
      document.getElementById("crumb-cat").textContent = "🃏 Global";
      document.getElementById("crumb-topic").textContent = "Flashcards (all topics)";
      return;
    }
    if (currentGlobalMode === "mastery") {
      document.getElementById("crumb-cat").textContent = "📊 Overview";
      document.getElementById("crumb-topic").textContent = "Mastery Map";
      return;
    }
    if (!topic) {
      document.getElementById("crumb-cat").textContent = "—";
      document.getElementById("crumb-topic").textContent = "Pick a topic";
      return;
    }
    document.getElementById("crumb-cat").textContent = topic.category;
    document.getElementById("crumb-topic").textContent = topic.title;
  }
  function setMode(mode) {
    currentMode = mode;
    document.querySelectorAll(".mode-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.mode === mode);
    });
    if (currentTopicId && !currentGlobalMode) renderCurrentView();
  }

  function hideAllViews() {
    ["welcome", "read-view", "quiz-view", "flash-view", "mastery-view"].forEach((id) => {
      const el = document.getElementById(id);
      if (el) el.classList.add("hidden");
    });
  }

  function renderCurrentView() {
    if (currentGlobalMode === "random") return renderRandom();
    if (currentGlobalMode === "flashcards-all") return renderFlashAll();
    if (currentGlobalMode === "mastery") return renderMastery();

    if (!currentTopicId) {
      hideAllViews();
      document.getElementById("welcome").classList.remove("hidden");
      return;
    }
    hideAllViews();
    if (currentMode === "read") {
      document.getElementById("read-view").classList.remove("hidden");
      renderRead();
    } else if (currentMode === "quiz") {
      document.getElementById("quiz-view").classList.remove("hidden");
      renderQuiz();
    } else if (currentMode === "flash") {
      document.getElementById("flash-view").classList.remove("hidden");
      renderFlash();
    }
  }

  /* ---------- Read mode ---------- */
  function renderRead() {
    const topic = DATA.topics.find((t) => t.id === currentTopicId);
    if (!topic) return;
    const view = document.getElementById("read-view");
    const sections = (topic.summary || []).map(renderSummaryBlock).join("");
    view.innerHTML = `
      <h1>${escapeHtml(topic.title)}</h1>
      <p class="read-sub">${escapeHtml(topic.category)} · ${topic.questions.length} interview question${topic.questions.length === 1 ? "" : "s"} (with multi-level follow-ups)</p>
      <div class="read-section">${sections}</div>
      <div class="read-jump">
        <button class="jump-btn secondary" id="go-flash">🃏 Flashcards</button>
        <button class="jump-btn" id="go-quiz">🎯 Test yourself in Quiz mode →</button>
      </div>
    `;
    document.getElementById("go-quiz").addEventListener("click", () => setMode("quiz"));
    document.getElementById("go-flash").addEventListener("click", () => setMode("flash"));
    document.querySelector(".content").scrollTo({ top: 0 });
  }

  function renderSummaryBlock(b) {
    if (b.h2) return `<h2>${b.h2}</h2>`;
    if (b.p) return `<p>${b.p}</p>`;
    if (b.ul) return `<ul>${b.ul.map((x) => `<li>${x}</li>`).join("")}</ul>`;
    if (b.ol) return `<ol>${b.ol.map((x) => `<li>${x}</li>`).join("")}</ol>`;
    if (b.pre) return `<pre><code>${escapeHtml(b.pre)}</code></pre>`;
    if (b.table) {
      const [head, ...rows] = b.table;
      return `<table>
        <thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead>
        <tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody>
      </table>`;
    }
    if (b.callout) return `<div class="callout">${b.callout}</div>`;
    if (b.warn) return `<div class="callout callout-warn">${b.warn}</div>`;
    return "";
  }

  /* ---------- Quiz mode ---------- */
  function renderQuiz() {
    const topic = DATA.topics.find((t) => t.id === currentTopicId);
    if (!topic || !topic.questions || topic.questions.length === 0) {
      document.getElementById("quiz-view").innerHTML = `
        <div class="empty-state"><h3>No questions</h3>
        <p>This topic doesn't have quiz questions configured.</p></div>`;
      return;
    }
    const total = topic.questions.length;
    if (quizQuestionIndex >= total) quizQuestionIndex = 0;
    const idx = quizQuestionIndex;
    const question = topic.questions[idx];
    const tState = (revealState[currentTopicId][idx] = revealState[currentTopicId][idx] || { revealed: false, followups: {} });

    const view = document.getElementById("quiz-view");
    const progressPct = ((idx + 1) / total) * 100;
    view.innerHTML = `
      <div class="quiz-header">
        <div class="quiz-progress">
          Question ${idx + 1} of ${total}
          <span class="progress-bar"><span class="progress-fill" style="width: ${progressPct}%"></span></span>
        </div>
        <div class="quiz-nav">
          <button id="prev-q" ${idx === 0 ? "disabled" : ""}>← Previous</button>
          <button id="reset-q">↻ Hide answers</button>
          <button id="next-q" ${idx === total - 1 ? "disabled" : ""}>Next →</button>
        </div>
      </div>
      <div class="quiz-card">
        <div class="quiz-q"><span class="q-tag">Q${idx + 1}</span>${question.q}</div>
        ${renderAnswerBlock(question, tState, [idx], 0)}
        ${renderStarsRow(currentTopicId, idx, "quiz")}
      </div>
    `;
    bindQuizControls(topic, idx);
    bindStarRow();
    document.querySelector(".content").scrollTo({ top: 0 });
  }

  function renderAnswerBlock(question, tState, path, level) {
    const pathKey = path.join("/");
    const isRevealed = tState.followups[pathKey]?.revealed ?? (level === 0 ? tState.revealed : false);

    if (!isRevealed) {
      return `<button class="reveal-btn" data-action="reveal" data-path="${pathKey}">Show answer</button>`;
    }
    const followBtns = (question.followups || []).map((fu, fIdx) => {
      const subPath = [...path, fIdx];
      const subKey = subPath.join("/");
      const subLabel = "F" + subPath.slice(1).map((p) => p + 1).join(".");
      const fState = tState.followups[subKey];
      const fRevealed = fState?.shown;
      if (fRevealed) {
        return renderFollowup(fu, tState, subPath, level + 1);
      }
      return `
        <div class="followup followup-${Math.min(level, 3)}">
          <button class="reveal-btn secondary" data-action="show-followup" data-path="${subKey}">
            ↳ Show follow-up ${subLabel}: <em>${truncate(fu.q, 90)}</em>
          </button>
        </div>`;
    }).join("");
    return `<div class="quiz-a">${question.a}</div>${followBtns}`;
  }

  function renderFollowup(fu, tState, path, level) {
    const pathKey = path.join("/");
    const isAnswerShown = tState.followups[pathKey]?.revealed;
    const tagBg = `followup-${Math.min(level - 1, 3)}`;
    const labelLevel = "F" + path.slice(1).map((p) => p + 1).join(".");
    if (!isAnswerShown) {
      return `
        <div class="followup ${tagBg}">
          <div class="quiz-q"><span class="q-tag">${labelLevel}</span>${fu.q}</div>
          <button class="reveal-btn" data-action="reveal" data-path="${pathKey}">Show answer</button>
        </div>`;
    }
    const childBtns = (fu.followups || []).map((sub, sIdx) => {
      const subPath = [...path, sIdx];
      const subKey = subPath.join("/");
      const subLabel = "F" + subPath.slice(1).map((p) => p + 1).join(".");
      const sState = tState.followups[subKey];
      if (sState?.shown) {
        return renderFollowup(sub, tState, subPath, level + 1);
      }
      return `
        <div class="followup followup-${Math.min(level, 3)}">
          <button class="reveal-btn secondary" data-action="show-followup" data-path="${subKey}">
            ↳ Show follow-up ${subLabel}: <em>${truncate(sub.q, 90)}</em>
          </button>
        </div>`;
    }).join("");
    return `
      <div class="followup ${tagBg}">
        <div class="quiz-q"><span class="q-tag">${labelLevel}</span>${fu.q}</div>
        <div class="quiz-a">${fu.a}</div>
        ${childBtns}
      </div>`;
  }

  function bindQuizControls(topic, idx) {
    document.getElementById("prev-q")?.addEventListener("click", () => {
      if (quizQuestionIndex > 0) { quizQuestionIndex -= 1; renderQuiz(); }
    });
    document.getElementById("next-q")?.addEventListener("click", () => {
      if (quizQuestionIndex < topic.questions.length - 1) { quizQuestionIndex += 1; renderQuiz(); }
    });
    document.getElementById("reset-q")?.addEventListener("click", () => {
      revealState[currentTopicId][quizQuestionIndex] = { revealed: false, followups: {} };
      renderQuiz();
    });
    document.querySelectorAll('[data-action="reveal"]').forEach((btn) => {
      btn.addEventListener("click", () => {
        const path = btn.dataset.path.split("/").map(Number);
        const tState = revealState[currentTopicId][path[0]];
        if (path.length === 1) tState.revealed = true;
        else tState.followups[btn.dataset.path] = { revealed: true, shown: true };
        renderQuiz();
      });
    });
    document.querySelectorAll('[data-action="show-followup"]').forEach((btn) => {
      btn.addEventListener("click", () => {
        const path = btn.dataset.path.split("/").map(Number);
        const tState = revealState[currentTopicId][path[0]];
        tState.followups[btn.dataset.path] = { ...(tState.followups[btn.dataset.path] || {}), shown: true };
        renderQuiz();
      });
    });
  }

  function renderStarsRow(topicId, qIdx, contextLabel) {
    const stars = getStars(topicId, qIdx);
    const tip = stars
      ? `Confidence: ${stars}★`
      : `Rate confidence (1=★ very weak, 5=★★★★★ mastered)`;
    return `
      <div class="stars-row" data-topic="${topicId}" data-qidx="${qIdx}">
        <span class="stars-label">${tip}</span>
        ${[1,2,3,4,5].map((n) =>
          `<span class="star ${n <= stars ? 'active committed' : ''}" data-stars="${n}">★</span>`
        ).join("")}
      </div>`;
  }

  function bindStarRow() {
    document.querySelectorAll(".stars-row").forEach((row) => {
      const topicId = row.dataset.topic;
      const qIdx = Number(row.dataset.qidx);
      row.querySelectorAll(".star").forEach((star) => {
        star.addEventListener("mouseenter", () => {
          const n = Number(star.dataset.stars);
          row.querySelectorAll(".star").forEach((s) => {
            s.classList.toggle("active", Number(s.dataset.stars) <= n);
          });
        });
        star.addEventListener("mouseleave", () => {
          const stars = getStars(topicId, qIdx);
          row.querySelectorAll(".star").forEach((s) => {
            s.classList.toggle("active", Number(s.dataset.stars) <= stars);
          });
        });
        star.addEventListener("click", () => {
          const n = Number(star.dataset.stars);
          const current = getStars(topicId, qIdx);
          // toggle: clicking the same level twice clears
          setStars(topicId, qIdx, n === current ? 0 : n);
          renderSidebar(); // update pip color
          highlightActiveNav();
          // re-render the stars row in place
          const newRow = document.createElement("div");
          newRow.innerHTML = renderStarsRow(topicId, qIdx, "");
          row.replaceWith(newRow.firstElementChild);
          bindStarRow();
        });
      });
    });
  }

  /* ---------- Flashcard mode (per-topic) ---------- */
  function renderFlash() {
    const topic = DATA.topics.find((t) => t.id === currentTopicId);
    if (!topic || !topic.questions || topic.questions.length === 0) {
      document.getElementById("flash-view").innerHTML = `
        <div class="empty-state"><h3>No flashcards</h3></div>`;
      return;
    }
    const total = topic.questions.length;
    if (flashIndex >= total) flashIndex = 0;
    const idx = flashIndex;
    const q = topic.questions[idx];
    const view = document.getElementById("flash-view");
    view.innerHTML = `
      <div class="flash-header">
        <div>
          <div class="flash-title">${escapeHtml(topic.title)}</div>
          <div class="flash-progress">Card ${idx + 1} of ${total}</div>
        </div>
        <div class="quiz-nav">
          <button id="flash-prev" ${idx === 0 ? "disabled" : ""}>← Previous</button>
          <button id="flash-shuffle">🎲 Shuffle</button>
          <button id="flash-next" ${idx === total - 1 ? "disabled" : ""}>Next →</button>
        </div>
      </div>
      ${renderFlashcard(topic, idx, q)}
    `;
    bindFlashControls(topic);
  }

  function renderFlashcard(topic, idx, q) {
    return `
      <div class="flashcard">
        <div>
          <div class="flash-cat">${escapeHtml(topic.category)}</div>
          <div class="flash-topic">${escapeHtml(topic.title)}</div>
          <div class="flash-q">${q.q}</div>
          ${flashAnswerShown ? `<div class="flash-a">${q.a}</div>` : ""}
        </div>
        <div class="flash-actions">
          ${flashAnswerShown
            ? renderStarsRow(topic.id, idx, "flash")
            : `<button class="flash-control" id="flip-card">Flip — show answer</button>`}
          ${flashAnswerShown
            ? `<button class="flash-control" id="flash-advance">Next card →</button>`
            : ""}
        </div>
      </div>`;
  }

  function bindFlashControls(topic) {
    document.getElementById("flash-prev")?.addEventListener("click", () => {
      if (flashIndex > 0) { flashIndex -= 1; flashAnswerShown = false; renderFlash(); }
    });
    document.getElementById("flash-next")?.addEventListener("click", () => {
      if (flashIndex < topic.questions.length - 1) { flashIndex += 1; flashAnswerShown = false; renderFlash(); }
    });
    document.getElementById("flash-shuffle")?.addEventListener("click", () => {
      flashIndex = Math.floor(Math.random() * topic.questions.length);
      flashAnswerShown = false;
      renderFlash();
    });
    document.getElementById("flip-card")?.addEventListener("click", () => {
      flashAnswerShown = true; renderFlash();
    });
    document.getElementById("flash-advance")?.addEventListener("click", () => {
      flashIndex = (flashIndex + 1) % topic.questions.length;
      flashAnswerShown = false; renderFlash();
    });
    bindStarRow();
  }

  /* ---------- Random Mix ---------- */
  function buildAllQuestions() {
    const all = [];
    DATA.topics.forEach((t) => {
      t.questions.forEach((_, i) => all.push({ topicId: t.id, qIdx: i }));
    });
    return all;
  }
  function shuffle(a) {
    const out = a.slice();
    for (let i = out.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [out[i], out[j]] = [out[j], out[i]];
    }
    return out;
  }
  function renderRandom() {
    hideAllViews();
    document.getElementById("flash-view").classList.remove("hidden");
    document.getElementById("topic-modes").classList.add("hidden");
    if (randomQueue.length === 0) randomQueue = shuffle(buildAllQuestions());
    const head = randomQueue[0];
    const topic = DATA.topics.find((t) => t.id === head.topicId);
    const q = topic.questions[head.qIdx];
    const view = document.getElementById("flash-view");
    view.innerHTML = `
      <div class="flash-header">
        <div>
          <div class="flash-title">🎲 Random Mix</div>
          <div class="flash-progress">${randomQueue.length} cards remaining</div>
        </div>
        <div class="quiz-nav">
          <button id="flash-shuffle">🎲 Reshuffle</button>
        </div>
      </div>
      ${renderFlashcard(topic, head.qIdx, q)}
    `;
    document.getElementById("flash-shuffle")?.addEventListener("click", () => {
      randomQueue = shuffle(buildAllQuestions());
      flashAnswerShown = false; renderRandom();
    });
    document.getElementById("flip-card")?.addEventListener("click", () => {
      flashAnswerShown = true; renderRandom();
    });
    document.getElementById("flash-advance")?.addEventListener("click", () => {
      randomQueue.shift();
      if (randomQueue.length === 0) randomQueue = shuffle(buildAllQuestions());
      flashAnswerShown = false; renderRandom();
    });
    bindStarRow();
  }

  /* ---------- Flashcards (all) — weak-first ---------- */
  function buildWeakFirstQueue() {
    const c = loadConfidence();
    const all = buildAllQuestions().map((x) => {
      const stars = c[`${x.topicId}/${x.qIdx}`]?.stars ?? 0;
      const lastSeen = c[`${x.topicId}/${x.qIdx}`]?.lastSeen ?? "1970";
      return { ...x, stars, lastSeen };
    });
    // sort: lower stars first; among same stars, oldest first
    all.sort((a, b) => (a.stars - b.stars) || (a.lastSeen < b.lastSeen ? -1 : 1));
    return all;
  }
  function renderFlashAll() {
    hideAllViews();
    document.getElementById("flash-view").classList.remove("hidden");
    document.getElementById("topic-modes").classList.add("hidden");
    if (allFlashQueue.length === 0) allFlashQueue = buildWeakFirstQueue();
    const head = allFlashQueue[0];
    if (!head) {
      document.getElementById("flash-view").innerHTML = `
        <div class="empty-state"><h3>All cards mastered ⭐</h3>
        <p>Reset progress in the sidebar to start over.</p></div>`;
      return;
    }
    const topic = DATA.topics.find((t) => t.id === head.topicId);
    const q = topic.questions[head.qIdx];
    const view = document.getElementById("flash-view");
    view.innerHTML = `
      <div class="flash-header">
        <div>
          <div class="flash-title">🃏 Flashcards — weak first</div>
          <div class="flash-progress">${allFlashQueue.length} cards remaining · current ${head.stars}★</div>
        </div>
        <div class="quiz-nav">
          <button id="flash-rebuild">↻ Rebuild queue</button>
        </div>
      </div>
      ${renderFlashcard(topic, head.qIdx, q)}
    `;
    document.getElementById("flash-rebuild")?.addEventListener("click", () => {
      allFlashQueue = buildWeakFirstQueue();
      flashAnswerShown = false; renderFlashAll();
    });
    document.getElementById("flip-card")?.addEventListener("click", () => {
      flashAnswerShown = true; renderFlashAll();
    });
    document.getElementById("flash-advance")?.addEventListener("click", () => {
      allFlashQueue.shift();
      flashAnswerShown = false; renderFlashAll();
    });
    bindStarRow();
  }

  /* ---------- Mastery Map ---------- */
  function renderMastery() {
    hideAllViews();
    document.getElementById("mastery-view").classList.remove("hidden");
    document.getElementById("topic-modes").classList.add("hidden");
    const view = document.getElementById("mastery-view");
    const stats = masteryStats();
    const seenCount = stats.counts.slice(1).reduce((a, b) => a + b, 0);
    const masteredCount = stats.counts[5];
    const weakCount = stats.counts[1] + stats.counts[2];
    const avgStars = (() => {
      let sum = 0; let n = 0;
      stats.counts.forEach((cnt, lvl) => { if (lvl > 0) { sum += cnt * lvl; n += cnt; } });
      return n === 0 ? 0 : (sum / n).toFixed(2);
    })();

    let html = `
      <h1>📊 Mastery Map</h1>
      <p class="mastery-sub">A heat-map of your confidence on every question. Click any cell to jump to that question. Use this to find weak spots and drill them.</p>
      <div class="mastery-stats">
        <div class="mastery-stat"><div class="num">${stats.total}</div><div class="lbl">Total questions</div></div>
        <div class="mastery-stat"><div class="num">${seenCount}</div><div class="lbl">Rated</div></div>
        <div class="mastery-stat"><div class="num">${avgStars}</div><div class="lbl">Avg ★</div></div>
        <div class="mastery-stat"><div class="num">${masteredCount}</div><div class="lbl">Mastered (5★)</div></div>
        <div class="mastery-stat"><div class="num">${weakCount}</div><div class="lbl">Weak (1–2★)</div></div>
      </div>
      <div class="mastery-legend">
        <div class="legend-item"><span class="legend-cell l0"></span> Unrated</div>
        <div class="legend-item"><span class="legend-cell l1"></span> 1★ Very weak</div>
        <div class="legend-item"><span class="legend-cell l2"></span> 2★ Weak</div>
        <div class="legend-item"><span class="legend-cell l3"></span> 3★ Fair</div>
        <div class="legend-item"><span class="legend-cell l4"></span> 4★ Strong</div>
        <div class="legend-item"><span class="legend-cell l5"></span> 5★ Mastered</div>
      </div>
    `;
    DATA.categories.forEach((cat) => {
      const topics = DATA.topics.filter((t) => t.category === cat);
      if (topics.length === 0) return;
      html += `<div class="mastery-cat-block"><h3>${escapeHtml(cat)}</h3>`;
      topics.forEach((t) => {
        const c = loadConfidence();
        const cells = t.questions.map((q, i) => {
          const stars = c[`${t.id}/${i}`]?.stars ?? 0;
          const tip = (stars ? `${stars}★` : "Unrated") + " — " + truncate(q.q.replace(/<[^>]+>/g,""), 80);
          return `<div class="cell lvl-${stars}" data-topic="${t.id}" data-qidx="${i}">
            <div class="cell-tooltip">${tip}</div>
          </div>`;
        }).join("");
        html += `
          <div class="mastery-topic-row">
            <div class="mastery-topic-name" data-topic="${t.id}">${escapeHtml(t.title)}</div>
            <div class="mastery-cells">${cells}</div>
          </div>`;
      });
      html += "</div>";
    });
    view.innerHTML = html;
    view.querySelectorAll(".cell").forEach((cell) => {
      cell.addEventListener("click", () => {
        const t = cell.dataset.topic;
        const i = Number(cell.dataset.qidx);
        currentGlobalMode = null;
        currentMode = "quiz";
        currentTopicId = t;
        quizQuestionIndex = i;
        document.getElementById("topic-modes").classList.remove("hidden");
        document.querySelectorAll(".mode-btn").forEach((b) =>
          b.classList.toggle("active", b.dataset.mode === "quiz"));
        highlightActiveNav();
        updateBreadcrumbs();
        renderCurrentView();
      });
    });
    view.querySelectorAll(".mastery-topic-name").forEach((el) => {
      el.addEventListener("click", () => selectTopic(el.dataset.topic));
    });
  }

  /* ---------- Search ---------- */
  function renderSearchResults(query) {
    if (!query) { renderSidebar(); highlightActiveNav(); return; }
    const q = query.toLowerCase();
    const nav = document.getElementById("nav");
    nav.innerHTML = "";
    const matches = [];
    DATA.topics.forEach((topic) => {
      const titleHit = topic.title.toLowerCase().includes(q);
      const qHits = topic.questions.filter((qq) =>
        qq.q.toLowerCase().includes(q) || qq.a.toLowerCase().includes(q));
      if (titleHit || qHits.length > 0) matches.push({ topic, qHits });
    });
    if (matches.length === 0) {
      nav.innerHTML = `<div style="padding: 18px; color: #94a3b8; font-size: 13px;">No matches for "${escapeHtml(query)}"</div>`;
      return;
    }
    matches.forEach(({ topic, qHits }) => {
      const item = document.createElement("div");
      item.className = "nav-item";
      item.dataset.topicId = topic.id;
      item.innerHTML = `
        <span>${topic.title}</span>
        <span class="nav-q-count">${qHits.length || topic.questions.length}</span>
      `;
      item.addEventListener("click", () => {
        selectTopic(topic.id);
        document.getElementById("search").value = "";
        renderSidebar(); highlightActiveNav();
      });
      nav.appendChild(item);
    });
  }

  /* ---------- Helpers ---------- */
  function escapeHtml(s) {
    return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }
  function truncate(s, n) {
    const stripped = String(s).replace(/<[^>]+>/g, "");
    return stripped.length > n ? stripped.slice(0, n) + "…" : stripped;
  }

  /* ---------- Init ---------- */
  function init() {
    renderSidebar();
    document.querySelectorAll(".mode-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        currentGlobalMode = null;
        document.getElementById("topic-modes").classList.remove("hidden");
        if (currentTopicId) setMode(btn.dataset.mode);
      });
    });
    document.querySelectorAll(".gm-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        currentGlobalMode = btn.dataset.gmode;
        flashAnswerShown = false;
        document.getElementById("welcome").classList.add("hidden");
        if (currentGlobalMode === "random") randomQueue = shuffle(buildAllQuestions());
        if (currentGlobalMode === "flashcards-all") allFlashQueue = buildWeakFirstQueue();
        highlightActiveNav();
        updateBreadcrumbs();
        renderCurrentView();
      });
    });
    document.getElementById("search").addEventListener("input", (e) => {
      renderSearchResults(e.target.value.trim());
    });
    document.getElementById("reset-progress").addEventListener("click", () => {
      if (confirm("Clear all star ratings? This cannot be undone.")) {
        localStorage.removeItem(STORAGE_KEY);
        renderSidebar(); highlightActiveNav();
        if (currentGlobalMode === "mastery") renderMastery();
        else renderCurrentView();
      }
    });
    document.addEventListener("keydown", (e) => {
      if (e.target.tagName === "INPUT") return;
      const inFlash = currentMode === "flash" || currentGlobalMode === "random" || currentGlobalMode === "flashcards-all";
      if (e.key === " " && inFlash && !flashAnswerShown) {
        e.preventDefault();
        flashAnswerShown = true;
        renderCurrentView();
      }
      if (e.key === "ArrowRight") {
        const t = DATA.topics.find((x) => x.id === currentTopicId);
        if (currentMode === "quiz" && t && quizQuestionIndex < t.questions.length - 1) {
          quizQuestionIndex += 1; renderQuiz();
        } else if (currentMode === "flash" && t && flashIndex < t.questions.length - 1) {
          flashIndex += 1; flashAnswerShown = false; renderFlash();
        }
      }
      if (e.key === "ArrowLeft") {
        const t = DATA.topics.find((x) => x.id === currentTopicId);
        if (currentMode === "quiz" && quizQuestionIndex > 0) {
          quizQuestionIndex -= 1; renderQuiz();
        } else if (currentMode === "flash" && flashIndex > 0) {
          flashIndex -= 1; flashAnswerShown = false; renderFlash();
        }
      }
      if (e.key.toLowerCase() === "r") setMode("read");
      if (e.key.toLowerCase() === "q") setMode("quiz");
      if (e.key.toLowerCase() === "f") setMode("flash");
      if (e.key === "1" || e.key === "2" || e.key === "3" || e.key === "4" || e.key === "5") {
        // Quick star rating in flash mode
        const inAnyFlash = currentMode === "flash" || currentGlobalMode === "random" || currentGlobalMode === "flashcards-all";
        if (inAnyFlash && flashAnswerShown) {
          let topicId = currentTopicId, qIdx = flashIndex;
          if (currentGlobalMode === "random") {
            const head = randomQueue[0]; topicId = head.topicId; qIdx = head.qIdx;
          } else if (currentGlobalMode === "flashcards-all") {
            const head = allFlashQueue[0]; topicId = head.topicId; qIdx = head.qIdx;
          }
          setStars(topicId, qIdx, Number(e.key));
          renderSidebar(); highlightActiveNav(); renderCurrentView();
        }
      }
    });
  }
  document.addEventListener("DOMContentLoaded", init);
})();
