/**
 * Preferências da UI (localStorage) — extensível para novas chaves.
 *
 * Uso:
 *   FiapSettings.get("theme")
 *   FiapSettings.set("theme", "dark")
 *   FiapSettings.getAll()
 *   FiapSettings.onChange(fn)
 */
(function (global) {
  "use strict";

  var STORAGE_KEY = "fiap_settings_v1";
  var DEFAULTS = {
    theme: "system", // light | dark | system
  };

  var listeners = [];

  function readStore() {
    try {
      var raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return {};
      var parsed = JSON.parse(raw);
      return parsed && typeof parsed === "object" ? parsed : {};
    } catch (e) {
      return {};
    }
  }

  function writeStore(data) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  }

  function getAll() {
    return Object.assign({}, DEFAULTS, readStore());
  }

  function get(key) {
    var all = getAll();
    return Object.prototype.hasOwnProperty.call(all, key) ? all[key] : undefined;
  }

  function set(key, value) {
    var data = readStore();
    data[key] = value;
    writeStore(data);
    notify({ key: key, value: value, all: getAll() });
    if (key === "theme") applyTheme();
    return value;
  }

  function setMany(partial) {
    var data = Object.assign(readStore(), partial || {});
    writeStore(data);
    notify({ key: null, value: null, all: getAll() });
    applyTheme();
    return getAll();
  }

  function reset() {
    localStorage.removeItem(STORAGE_KEY);
    notify({ key: null, value: null, all: getAll() });
    applyTheme();
    return getAll();
  }

  function resolveTheme(pref) {
    var theme = pref || get("theme") || DEFAULTS.theme;
    if (theme === "system") {
      return global.matchMedia && global.matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light";
    }
    return theme === "dark" ? "dark" : "light";
  }

  function applyTheme() {
    var pref = get("theme") || DEFAULTS.theme;
    var resolved = resolveTheme(pref);
    var root = document.documentElement;
    root.setAttribute("data-theme", resolved);
    root.setAttribute("data-theme-pref", pref);
    try {
      root.style.colorScheme = resolved;
    } catch (e) { /* ignore */ }
    return resolved;
  }

  function onChange(fn) {
    if (typeof fn === "function") listeners.push(fn);
    return function unsubscribe() {
      listeners = listeners.filter(function (f) { return f !== fn; });
    };
  }

  function notify(payload) {
    listeners.forEach(function (fn) {
      try { fn(payload); } catch (e) { /* ignore */ }
    });
  }

  // Reage à mudança de tema do SO quando a preferência é "system"
  if (global.matchMedia) {
    try {
      global.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () {
        if (get("theme") === "system") applyTheme();
      });
    } catch (e) { /* Safari antigo */ }
  }

  applyTheme();

  global.FiapSettings = {
    STORAGE_KEY: STORAGE_KEY,
    DEFAULTS: DEFAULTS,
    getAll: getAll,
    get: get,
    set: set,
    setMany: setMany,
    reset: reset,
    resolveTheme: resolveTheme,
    applyTheme: applyTheme,
    onChange: onChange,
  };
})(window);
