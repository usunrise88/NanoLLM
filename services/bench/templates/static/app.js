// The scripts of the public pages: filters apply on change, without a button,
// and the theme switch. Without JS the form's button does the first, and the
// theme follows the system.
"use strict";
document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll("form[data-autosubmit]").forEach(function (form) {
    form.addEventListener("change", function () {
      // Leave empty filters out of the URL.
      form.querySelectorAll("select").forEach(function (s) { s.disabled = s.value === ""; });
      form.submit();
    });
  });

  var root = document.documentElement;
  var group = document.querySelector(".theme");
  if (!group) return;
  var mq = matchMedia("(prefers-color-scheme: dark)");
  function stored() {
    try { var v = localStorage.getItem("theme"); return v === "light" || v === "dark" ? v : "system"; } catch (e) { return "system"; }
  }
  function apply(t) {
    root.classList.toggle("dark", t === "dark" || (t === "system" && mq.matches));
    root.classList.toggle("light", t === "light");
    group.querySelectorAll("button").forEach(function (b) { b.setAttribute("aria-pressed", String(b.dataset.theme === t)); });
  }
  group.hidden = false;
  apply(stored());
  group.addEventListener("click", function (e) {
    var b = e.target.closest("button[data-theme]");
    if (!b) return;
    var t = b.dataset.theme;
    try { if (t === "system") localStorage.removeItem("theme"); else localStorage.setItem("theme", t); } catch (err) {}
    apply(t);
  });
  mq.addEventListener("change", function () { if (stored() === "system") apply("system"); });
});
