(function () {
  const drawer = document.getElementById("llm-interpretation-card");
  const payloadEl = document.getElementById("llm-payload-json");
  if (!drawer || !payloadEl) return;

  const chatBox = document.getElementById("llm-chat");
  const scrollArea = drawer.querySelector(".llm-drawer-body");
  const loading = document.getElementById("llm-loading");
  const errBox = document.getElementById("llm-error");
  const badge = document.getElementById("llm-provider-badge");
  const form = document.getElementById("llm-chat-form");
  const input = document.getElementById("llm-chat-input");
  const sendBtn = document.getElementById("llm-chat-send");
  const fab = document.getElementById("llm-fab");
  const fabDot = document.getElementById("llm-fab-dot");
  const closeBtn = document.getElementById("llm-close");
  const backdrop = document.getElementById("llm-backdrop");
  const enabled = drawer.dataset.llmEnabled === "true";
  let useRulesMode = !enabled;

  let payload;
  try {
    payload = JSON.parse(payloadEl.textContent);
  } catch (e) {
    showError("Não foi possível preparar os dados para interpretação.");
    return;
  }

  const history = []; // [{role, content}]
  const isDesktop = () => window.innerWidth >= 992;

  let feedbackTags = [];
  try {
    const ftEl = document.getElementById("llm-feedback-tags");
    if (ftEl) feedbackTags = JSON.parse(ftEl.textContent) || [];
  } catch (e) {
    feedbackTags = [];
  }

  // ---- Drawer open/close ----
  function openDrawer() {
    drawer.classList.add("open");
    drawer.setAttribute("aria-hidden", "false");
    fab.setAttribute("aria-expanded", "true");
    fabDot.classList.add("d-none");
    loadInterpretation();
    // No mobile, escurece o fundo (tela cheia). No desktop flutua por cima.
    if (!isDesktop()) {
      backdrop.hidden = false;
      requestAnimationFrame(() => backdrop.classList.add("show"));
    }
    if (input && !input.disabled) setTimeout(() => input.focus(), 320);
  }
  function closeDrawer() {
    drawer.classList.remove("open");
    drawer.setAttribute("aria-hidden", "true");
    fab.setAttribute("aria-expanded", "false");
    backdrop.classList.remove("show");
    setTimeout(() => { backdrop.hidden = true; }, 300);
  }
  function toggleDrawer() {
    drawer.classList.contains("open") ? closeDrawer() : openDrawer();
  }
  fab.addEventListener("click", toggleDrawer);
  closeBtn.addEventListener("click", closeDrawer);
  backdrop.addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && drawer.classList.contains("open")) closeDrawer();
  });

  // ---- 1) Interpretação inicial (carregada sob demanda, ao abrir o assistente) ----
  let interpretationLoaded = false;
  function loadInterpretation() {
    if (interpretationLoaded) return;
    interpretationLoaded = true;
    if (loading) loading.classList.remove("d-none");
    errBox.classList.add("d-none");

    fetch("/api/llm/interpret", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then((r) => r.json().then((d) => ({ ok: r.ok, data: d })))
      .then(({ ok, data }) => {
        if (loading) loading.classList.add("d-none");
        if (!ok) throw new Error(data.error || "Erro na interpretação");
        // Em fallback (Ollama indisponível) o badge do topo já indica "modo local".
        setBadge(data);
        addMessage("assistant", data.text || "", {
          feedback: true,
          question: "Interpretação inicial",
          provider: data.provider,
        });
        history.push({ role: "assistant", content: data.text || "" });
        // Chat disponível nos dois modos (Ollama e regras locais).
        form.classList.remove("d-none");
        if (input && !input.disabled) setTimeout(() => input.focus(), 50);
      })
      .catch((e) => {
        interpretationLoaded = false; // permite tentar de novo ao reabrir
        showError(friendlyError(e.message));
      });
  }

  // ---- 2) Perguntas de acompanhamento ----
  if (form) {
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      sendQuestion(input.value);
    });
  }
  function sendQuestion(q) {
    q = (q || "").trim();
    if (!q || sendBtn.disabled) return;
    input.value = "";
    addMessage("user", q);
    const priorHistory = history.slice();
    history.push({ role: "user", content: q });

    const typing = addTyping();
    setBusy(true);

    fetch("/api/llm/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode: payload.mode,
        result: payload.result,
        patient: payload.patient,
        feature_highlights: payload.feature_highlights,
        history: priorHistory,
        question: q,
        prefer_fallback: useRulesMode,
      }),
    })
      .then((r) => r.json().then((d) => ({ ok: r.ok, data: d })))
      .then(({ ok, data }) => {
        typing.remove();
        setBusy(false);
        if (!ok) throw new Error(data.error || "Erro no chat");
        setBadge(data);
        addMessage("assistant", data.text || "", {
          feedback: true,
          question: q,
          provider: data.provider,
        });
        history.push({ role: "assistant", content: data.text || "" });
      })
      .catch((e) => {
        typing.remove();
        setBusy(false);
        addMessage("assistant", "⚠️ " + friendlyError(e.message));
      });
  }

  // ---- Helpers ----
  function setBadge(data) {
    const isFallback = data.fallback_used || /fallback|regras/i.test(data.provider || "");
    useRulesMode = isFallback || !enabled;
    if (isFallback) {
      badge.textContent = "⚠️ modo regras";
      badge.title = "Ollama indisponível ou modo local — respostas por regras clínicas.";
      badge.classList.add("llm-badge-fallback");
    } else {
      badge.textContent = data.provider + " · " + data.latency_ms + "ms";
      badge.title = "";
      badge.classList.remove("llm-badge-fallback");
    }
  }

  function setBusy(b) {
    if (sendBtn) sendBtn.disabled = b;
    if (input) {
      input.disabled = b;
      if (!b) input.focus();
    }
  }

  function addMessage(role, text, opts) {
    opts = opts || {};
    const wrap = document.createElement("div");
    wrap.className = "llm-msg llm-msg-" + role;
    const bubble = document.createElement("div");
    bubble.className = "llm-bubble";
    bubble.innerHTML = role === "assistant" ? markdownToHtml(text) : escapeHtml(text);
    wrap.appendChild(bubble);
    if (opts.feedback) {
      bubble.appendChild(
        buildFeedback({ question: opts.question || "", answer: text, provider: opts.provider || "" })
      );
    }
    const time = document.createElement("span");
    time.className = "llm-msg-time";
    time.textContent = nowTime();
    wrap.appendChild(time);
    chatBox.appendChild(wrap);
    scrollToEnd();
    return wrap;
  }

  function nowTime() {
    return new Date().toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit" });
  }

  // ---- Feedback do médico (👍/👎 + motivo + tags) ----
  function buildFeedback(ctx) {
    const bar = document.createElement("div");
    bar.className = "llm-feedback";
    bar.innerHTML =
      '<span class="llm-feedback-q">Esta resposta foi útil?</span>' +
      '<button type="button" class="llm-fb-btn llm-fb-like" title="Útil" aria-label="Útil">👍</button>' +
      '<button type="button" class="llm-fb-btn llm-fb-dislike" title="Não útil" aria-label="Não útil">👎</button>';

    bar.querySelector(".llm-fb-like").addEventListener("click", () => {
      sendFeedback({ rating: "like", question: ctx.question, answer: ctx.answer, provider: ctx.provider });
      replaceWithNote(bar, "Obrigado pelo feedback! 👍");
    });
    bar.querySelector(".llm-fb-dislike").addEventListener("click", () => {
      bar.replaceWith(buildDislikeForm(ctx));
      scrollToEnd();
    });
    return bar;
  }

  function buildDislikeForm(ctx) {
    const box = document.createElement("div");
    box.className = "llm-fb-form";

    const label = document.createElement("div");
    label.className = "llm-fb-formlabel";
    label.textContent = "Ajude-nos a melhorar — o que houve?";

    const tagWrap = document.createElement("div");
    tagWrap.className = "llm-fb-tags";
    const selected = new Set();
    feedbackTags.forEach((t) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "llm-fb-tag";
      chip.textContent = t;
      chip.addEventListener("click", () => {
        chip.classList.toggle("active");
        if (selected.has(t)) selected.delete(t);
        else selected.add(t);
      });
      tagWrap.appendChild(chip);
    });

    const ta = document.createElement("textarea");
    ta.className = "form-control form-control-sm llm-fb-reason";
    ta.rows = 2;
    ta.maxLength = 500;
    ta.placeholder = "Descreva o problema (opcional)...";

    const actions = document.createElement("div");
    actions.className = "llm-fb-actions";
    const send = document.createElement("button");
    send.type = "button";
    send.className = "btn btn-sm fase2-btn-primary";
    send.textContent = "Enviar";
    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.className = "btn btn-sm btn-link text-muted p-0";
    cancel.textContent = "Cancelar";
    actions.appendChild(send);
    actions.appendChild(cancel);

    box.appendChild(label);
    box.appendChild(tagWrap);
    box.appendChild(ta);
    box.appendChild(actions);

    send.addEventListener("click", () => {
      sendFeedback({
        rating: "dislike",
        question: ctx.question,
        answer: ctx.answer,
        provider: ctx.provider,
        reason: ta.value,
        tags: Array.from(selected),
      });
      replaceWithNote(box, "Obrigado! Feedback registrado para auditoria. 🙏");
    });
    cancel.addEventListener("click", () => box.replaceWith(buildFeedback(ctx)));

    return box;
  }

  function replaceWithNote(el, msg) {
    const note = document.createElement("div");
    note.className = "llm-feedback llm-feedback-done";
    note.textContent = msg;
    el.replaceWith(note);
  }

  function sendFeedback(body) {
    const r = payload.result || {};
    const diag = r.diagnostico || (r.fase2 && r.fase2.diagnostico) || "";
    const prob = r.probabilidade != null ? r.probabilidade
      : (r.fase2 && r.fase2.probabilidade != null ? r.fase2.probabilidade : null);
    // Contexto completo do caso: permite reproduzir a mensagem na auditoria.
    const context = {
      mode: payload.mode,
      result: payload.result,
      feature_highlights: payload.feature_highlights,
    };
    fetch("/api/llm/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(
        Object.assign({ mode: payload.mode, diagnostico: diag, prob: prob, context: context }, body)
      ),
    }).catch(() => {});
  }

  function addTyping() {
    const wrap = document.createElement("div");
    wrap.className = "llm-msg llm-msg-assistant";
    wrap.innerHTML =
      '<div class="llm-bubble llm-typing"><span></span><span></span><span></span></div>';
    chatBox.appendChild(wrap);
    scrollToEnd();
    return wrap;
  }

  function scrollToEnd() {
    const el = scrollArea || chatBox;
    el.scrollTop = el.scrollHeight;
  }

  function showError(msg) {
    if (loading) loading.classList.add("d-none");
    errBox.textContent = msg;
    errBox.classList.remove("d-none");
    badge.textContent = "erro";
  }

  function friendlyError(msg) {
    msg = String(msg || "");
    // Erros típicos do Ollama local: servidor fora do ar ou modelo ausente.
    if (/Failed to fetch|NetworkError|ECONNREFUSED|Connection refused|11434/i.test(msg)) {
      return (
        "Não foi possível falar com o Ollama local. Verifique se ele está em execução " +
        "(ex.: `ollama run gpt-oss`) e se o modelo do .env está instalado."
      );
    }
    return msg;
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function markdownToHtml(md) {
    let html = md
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/^## (.+)$/gm, '<h6 class="llm-section-title">$1</h6>')
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*(?!\s)([^*]+?)\*(?!\*)/g, "$1<em>$2</em>")
      .replace(/`([^`]+?)`/g, "<code>$1</code>");

    const lines = html.split("\n");
    const out = [];
    let inList = false;
    for (const line of lines) {
      const t = line.trim();
      if (t.startsWith("<h6")) {
        if (inList) { out.push("</ul>"); inList = false; }
        out.push(t);
      } else if (t.startsWith("- ")) {
        if (!inList) { out.push('<ul class="llm-list">'); inList = true; }
        out.push("<li>" + t.slice(2) + "</li>");
      } else if (t) {
        if (inList) { out.push("</ul>"); inList = false; }
        out.push('<p class="llm-para">' + t + "</p>");
      }
    }
    if (inList) out.push("</ul>");
    return out.join("\n");
  }
})();
