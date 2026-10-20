// Runs in <head>, before the first paint: puts the remembered or system theme
// on <html> so a dark system never flashes white. The key is the one the
// landing page uses, so the choice follows the visitor between the two.
"use strict";
(function () {
  var root = document.documentElement;
  root.classList.add("js");
  var t = "system";
  try { t = localStorage.getItem("theme") || "system"; } catch (e) {}
  var dark = t === "dark" || (t !== "light" && matchMedia("(prefers-color-scheme: dark)").matches);
  root.classList.toggle("dark", dark);
  root.classList.toggle("light", t === "light");
})();
