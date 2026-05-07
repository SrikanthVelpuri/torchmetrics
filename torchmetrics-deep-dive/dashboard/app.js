/* ============================================================
 *  TorchMetrics Deep Dive — Dashboard logic
 * ============================================================ */
(function () {
  const DATA = window.TM_DATA;
  let currentTopicId = null;
  let currentMode = "read"; // "read" | "quiz"
  let revealState = {}; // topicId -> { qIdx -> { revealed: bool, followupsRevealed: Set<string> } }

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
        item.innerHTML = `
          <span>${topic.title}</span>
          <span class="nav-q-count">${topic.questions.length}Q</span>
        `;
        item.addEventListener("click", () => selectTopic(topic.id));
        nav.appendChild(item);
      });
    });
  }

  function highlightActiveNav() {
    document.querySelectorAll(".nav-item").forEach((el) => {
      el.classList.toggle("active", el.dataset.topicId === currentTopicId);
    });
  }

  /* ---------- Topic selection & mode ---------- */
  function selectTopic(topicId) {
    currentTopicId = topicId;
    if (!revealState[topicId]) revealState[topicId] = {};
    document.getElementById("welcome").classList.add("hidden");
    highlightActiveNav();
    updateBreadcrumbs();
    renderCurrentView();
  }

  function updateBreadcrumbs() {
    const topic = DATA.topics.find((t) => t.id === currentTopicId);
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
    if (currentTopicId) renderCurrentView();
  }

  function renderCurrentView() {
    const readView = document.getElementById("read-view");
    const quizView = document.getElementById("quiz-view");
    if (!currentTopicId) {
      readView.classList.add("hidden");
      quizView.classList.add("hidden");
      document.getElementById("welcome").classList.remove("hidden");
      return;
    }
    document.getElementById("welcome").classList.add("hidden");

    if (currentMode === "read") {
      readView.classList.remove("hidden");
      quizView.classList.add("hidden");
      renderRead();
    } else {
      readView.classList.add("hidden");
      quizView.classList.remove("hidden");
      renderQuiz();
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
        <button class="jump-btn" id="go-quiz">Test yourself in Quiz mode →</button>
      </div>
    `;
    document.getElementById("go-quiz").addEventListener("click", () => setMode("quiz"));
    view.scrollTop = 0;
    document.querySelector(".content").scrollTo({ top: 0, behavior: "instant" });
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
  let quizQuestionIndex = 0;
  function renderQuiz() {
    const topic = DATA.topics.find((t) => t.id === currentTopicId);
    if (!topic || !topic.questions || topic.questions.length === 0) {
      document.getElementById("quiz-view").innerHTML = `
        <div class="empty-state"><h3>No questions yet</h3>
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
      </div>
    `;
    bindQuizControls(topic, idx);
    view.scrollTo?.({ top: 0 });
    document.querySelector(".content").scrollTo({ top: 0, behavior: "instant" });
  }

  function renderAnswerBlock(question, tState, path, level) {
    const pathKey = path.join("/");
    const isRevealed = tState.followups[pathKey]?.revealed ?? (level === 0 ? tState.revealed : false);

    if (!isRevealed) {
      return `
        <button class="reveal-btn" data-action="reveal" data-path="${pathKey}">Show answer</button>
      `;
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
        </div>
      `;
    }).join("");

    return `
      <div class="quiz-a">${question.a}</div>
      ${followBtns}
    `;
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
        </div>
      `;
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
        </div>
      `;
    }).join("");

    return `
      <div class="followup ${tagBg}">
        <div class="quiz-q"><span class="q-tag">${labelLevel}</span>${fu.q}</div>
        <div class="quiz-a">${fu.a}</div>
        ${childBtns}
      </div>
    `;
  }

  function bindQuizControls(topic, idx) {
    document.getElementById("prev-q")?.addEventListener("click", () => {
      if (quizQuestionIndex > 0) {
        quizQuestionIndex -= 1;
        renderQuiz();
      }
    });
    document.getElementById("next-q")?.addEventListener("click", () => {
      if (quizQuestionIndex < topic.questions.length - 1) {
        quizQuestionIndex += 1;
        renderQuiz();
      }
    });
    document.getElementById("reset-q")?.addEventListener("click", () => {
      revealState[currentTopicId][quizQuestionIndex] = { revealed: false, followups: {} };
      renderQuiz();
    });

    document.querySelectorAll('[data-action="reveal"]').forEach((btn) => {
      btn.addEventListener("click", () => {
        const path = btn.dataset.path.split("/").map(Number);
        const tState = revealState[currentTopicId][path[0]];
        if (path.length === 1) {
          tState.revealed = true;
        } else {
          const key = btn.dataset.path;
          tState.followups[key] = { revealed: true, shown: true };
        }
        renderQuiz();
      });
    });

    document.querySelectorAll('[data-action="show-followup"]').forEach((btn) => {
      btn.addEventListener("click", () => {
        const path = btn.dataset.path.split("/").map(Number);
        const tState = revealState[currentTopicId][path[0]];
        const key = btn.dataset.path;
        tState.followups[key] = { ...(tState.followups[key] || {}), shown: true };
        renderQuiz();
      });
    });
  }

  /* ---------- Search ---------- */
  function renderSearchResults(query) {
    if (!query) {
      renderSidebar();
      highlightActiveNav();
      return;
    }
    const q = query.toLowerCase();
    const nav = document.getElementById("nav");
    nav.innerHTML = "";

    const matches = [];
    DATA.topics.forEach((topic) => {
      const titleHit = topic.title.toLowerCase().includes(q);
      const qHits = topic.questions.filter((qq) => qq.q.toLowerCase().includes(q) || qq.a.toLowerCase().includes(q));
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
        renderSidebar();
        highlightActiveNav();
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
      btn.addEventListener("click", () => setMode(btn.dataset.mode));
    });
    document.getElementById("search").addEventListener("input", (e) => {
      renderSearchResults(e.target.value.trim());
    });
    // Keyboard shortcuts
    document.addEventListener("keydown", (e) => {
      if (e.target.tagName === "INPUT") return;
      if (currentMode === "quiz" && currentTopicId) {
        const topic = DATA.topics.find((t) => t.id === currentTopicId);
        if (e.key === "ArrowRight" && quizQuestionIndex < topic.questions.length - 1) {
          quizQuestionIndex += 1;
          renderQuiz();
        }
        if (e.key === "ArrowLeft" && quizQuestionIndex > 0) {
          quizQuestionIndex -= 1;
          renderQuiz();
        }
      }
      if (e.key === "r" || e.key === "R") setMode("read");
      if (e.key === "q" || e.key === "Q") setMode("quiz");
    });
  }
  document.addEventListener("DOMContentLoaded", init);
})();
