/**
 * Página /configuracoes — liga controles ao FiapSettings.
 * Controles usam data-setting-key / data-setting-value (pronto para novas opções).
 */
(function () {
  "use strict";

  if (!window.FiapSettings) return;

  function syncUI() {
    var all = FiapSettings.getAll();

    document.querySelectorAll("[data-setting-key]").forEach(function (el) {
      var key = el.getAttribute("data-setting-key");
      var current = all[key];

      if (el.classList.contains("fiap-setting-choice")) {
        var val = el.getAttribute("data-setting-value");
        var on = String(current) === String(val);
        el.classList.toggle("active", on);
        el.setAttribute("aria-pressed", on ? "true" : "false");
        return;
      }

      if (el.type === "checkbox") {
        el.checked = Boolean(current);
        return;
      }

      if ("value" in el && el.getAttribute("data-setting-value") == null) {
        el.value = current == null ? "" : String(current);
      }
    });

    var resolved = FiapSettings.resolveTheme();
    var hint = document.getElementById("fiap-theme-resolved");
    if (hint) {
      hint.textContent = resolved === "dark" ? "Escuro" : "Claro";
    }
  }

  document.addEventListener("click", function (ev) {
    var btn = ev.target.closest(".fiap-setting-choice");
    if (!btn) return;
    var key = btn.getAttribute("data-setting-key");
    var value = btn.getAttribute("data-setting-value");
    if (!key) return;
    FiapSettings.set(key, value);
    syncUI();
  });

  document.addEventListener("change", function (ev) {
    var el = ev.target.closest("[data-setting-key]");
    if (!el || el.classList.contains("fiap-setting-choice")) return;
    var key = el.getAttribute("data-setting-key");
    if (!key) return;
    if (el.type === "checkbox") {
      FiapSettings.set(key, el.checked);
    } else {
      FiapSettings.set(key, el.value);
    }
    syncUI();
  });

  var resetBtn = document.getElementById("fiap-settings-reset");
  if (resetBtn) {
    resetBtn.addEventListener("click", function () {
      FiapSettings.reset();
      syncUI();
    });
  }

  FiapSettings.onChange(syncUI);
  syncUI();
})();
