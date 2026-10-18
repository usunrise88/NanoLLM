// The only script of the public pages: filters apply on change, without a
// button. Without JS the form's button does the same.
"use strict";
document.documentElement.classList.add("js");
document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll("form[data-autosubmit]").forEach(function (form) {
    form.addEventListener("change", function () {
      // Leave empty filters out of the URL.
      form.querySelectorAll("select").forEach(function (s) { s.disabled = s.value === ""; });
      form.submit();
    });
  });
});
