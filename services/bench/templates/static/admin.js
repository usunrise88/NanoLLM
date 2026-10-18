// The admin page: a bearer token kept in this tab's sessionStorage, and
// small tables over /admin/api.
"use strict";
(function () {
  var KEY = "benchd-admin-token";
  function token() {
    var t = null;
    try { t = sessionStorage.getItem(KEY); } catch (e) { /* storage blocked */ }
    if (!t) {
      t = window.prompt("Admin token");
      if (t) { try { sessionStorage.setItem(KEY, t); } catch (e) { /* keep in memory only */ } }
    }
    return t || "";
  }
  var status = function (msg) { document.getElementById("status").textContent = msg; };

  function api(method, path, body) {
    var opts = { method: method, headers: { "Authorization": "Bearer " + token() } };
    if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    return fetch(path, opts).then(function (r) {
      return r.json().then(function (j) {
        if (r.status === 401) { try { sessionStorage.removeItem(KEY); } catch (e) { /* ignore */ } }
        if (!r.ok) { throw new Error(j.error || r.statusText); }
        return j;
      });
    });
  }

  function cell(tr, text, href) {
    var td = document.createElement("td");
    if (href) {
      var a = document.createElement("a");
      a.href = href; a.textContent = text; td.appendChild(a);
    } else {
      td.textContent = text;
    }
    tr.appendChild(td);
    return td;
  }
  function button(td, label, fn) {
    var b = document.createElement("button");
    b.type = "button"; b.textContent = label; b.addEventListener("click", fn);
    td.appendChild(b); td.appendChild(document.createTextNode(" "));
  }
  function fill(id, rows, render) {
    var tbody = document.querySelector("#" + id + " tbody");
    tbody.textContent = "";
    rows.forEach(function (row) { var tr = document.createElement("tr"); render(tr, row); tbody.appendChild(tr); });
  }
  function report(p) { return p.then(function (j) { status(JSON.stringify(j)); return j; }, function (e) { status("error: " + e.message); }); }

  function loadQueue() {
    var state = document.querySelector("#queue-form select").value;
    return api("GET", "/admin/api/queue?state=" + encodeURIComponent(state)).then(function (j) {
      fill("queue", j.items, function (tr, it) {
        button(cell(tr, it.id + " "), "details", function () {
          api("GET", "/admin/api/submissions/" + it.id).then(function (j) {
            document.getElementById("detail").textContent = JSON.stringify(j, null, 2);
          }, function (e) { status("error: " + e.message); });
        });
        cell(tr, it.received_on); cell(tr, it.reason); cell(tr, it.decided_by);
        cell(tr, [it.system, it.cpu, it.gpu].join(" / ")); cell(tr, it.model);
        var td = cell(tr, "");
        ["published", "hidden", "queued"].forEach(function (s) {
          if (s === it.state) { return; }
          button(td, s === "published" ? "show" : s === "hidden" ? "hide" : "queue", function () {
            report(api("POST", "/admin/api/submissions/" + it.id + "/state", { state: s, reason: "admin" })).then(loadQueue);
          });
        });
      });
    }, function (e) { status("error: " + e.message); });
  }
  function loadAliases() {
    return api("GET", "/admin/api/aliases").then(function (j) {
      fill("aliases", j.aliases, function (tr, a) {
        cell(tr, a.kind); cell(tr, a.pattern); cell(tr, a.name);
        button(cell(tr, ""), "delete", function () {
          report(api("DELETE", "/admin/api/aliases?kind=" + encodeURIComponent(a.kind) + "&pattern=" + encodeURIComponent(a.pattern))).then(loadAliases);
        });
      });
    });
  }
  function loadVerified() {
    return api("GET", "/admin/api/verified").then(function (j) {
      fill("verified", j.verified, function (tr, v) {
        cell(tr, v.install_id.slice(0, 16) + "…"); cell(tr, v.label);
        button(cell(tr, ""), "revoke", function () {
          report(api("DELETE", "/admin/api/verified?install_id=" + encodeURIComponent(v.install_id))).then(loadVerified);
        });
      });
    });
  }
  function formData(form) {
    var o = {};
    new FormData(form).forEach(function (v, k) { o[k] = v; });
    return o;
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("queue-form").addEventListener("submit", function (e) { e.preventDefault(); loadQueue(); });
    document.getElementById("alias-form").addEventListener("submit", function (e) {
      e.preventDefault();
      report(api("POST", "/admin/api/aliases", formData(e.target))).then(loadAliases);
    });
    document.getElementById("verified-form").addEventListener("submit", function (e) {
      e.preventDefault();
      report(api("POST", "/admin/api/verified", formData(e.target))).then(loadVerified);
    });
    document.getElementById("reindex").addEventListener("click", function () { report(api("POST", "/admin/api/reindex")); });
    document.getElementById("logout").addEventListener("click", function (e) {
      e.preventDefault();
      try { sessionStorage.removeItem(KEY); } catch (err) { /* ignore */ }
      status("token forgotten");
    });
    loadQueue().then(loadAliases).then(loadVerified);
  });
})();
