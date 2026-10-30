/* trendlab report charts: small hand-written SVG charts, no dependencies.
   Each <figure data-chart> carries its data as JSON; pointing at (or arrow-keying
   through) a mark writes its exact values into the margin readout beside it. */
(function () {
  "use strict";

  var C = {
    strategy: "#2a78d6", bnh: "#999891", blend: "#b8860b", ink: "#0b0b0b",
    ink2: "#52514e", muted: "#898781", grid: "#e1e0d9", axis: "#c3c2b7",
    annot: "#b3261e", paper: "#fcfcfb", pale: "#9ec5f4", sand: "#c99a2e"
  };
  var SERIES_COLOR = { strategy: C.strategy, bnh: C.bnh, trend: C.strategy, sixty_forty: C.bnh, blend: C.blend };
  var NS = "http://www.w3.org/2000/svg";
  var reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function el(tag, attrs, parent) {
    var n = document.createElementNS(NS, tag);
    for (var k in attrs) if (attrs[k] !== undefined && attrs[k] !== null) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function txt(parent, x, y, s, attrs) {
    var a = { x: x, y: y, "class": "t" };
    for (var k in attrs || {}) a[k] = attrs[k];
    var t = el("text", a, parent);
    t.textContent = s;
    return t;
  }
  function minus(s) { return String(s).replace(/-/g, "−"); }
  function pct(x, d, sign) {
    if (x === null || x === undefined || isNaN(x)) return "–";
    var v = (x * 100).toFixed(d === undefined ? 0 : d);
    return minus((sign && x > 0 ? "+" : "") + v + "%");
  }
  function num(x, d, sign) {
    if (x === null || x === undefined || isNaN(x)) return "–";
    return minus((sign && x > 0 ? "+" : "") + x.toFixed(d === undefined ? 2 : d));
  }
  function pp(x) { return minus((x > 0 ? "+" : "") + (x * 100).toFixed(1) + " pp"); }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function lin(d0, d1, r0, r1) {
    var f = function (v) { return r0 + (v - d0) / (d1 - d0 || 1) * (r1 - r0); };
    f.invert = function (p) { return d0 + (p - r0) / (r1 - r0 || 1) * (d1 - d0); };
    return f;
  }
  function niceStep(span, n) {
    var raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw))), r = raw / mag;
    return (r >= 5 ? 10 : r >= 2 ? 5 : r >= 1 ? 2 : 1) * mag;
  }
  function ticks(lo, hi, n) {
    var s = niceStep(hi - lo, n), out = [];
    for (var v = Math.ceil(lo / s) * s; v <= hi + 1e-9; v += s) out.push(+v.toFixed(10));
    return out;
  }
  function dateLabel(d) {
    var m = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    var p = d.split("-");
    return p[2].replace(/^0/, "") + " " + m[+p[1] - 1] + " " + p[0];
  }
  function rows(list) {
    return '<dl class="ro">' + list.map(function (r) {
      var sw = r[2] ? '<i class="sw" style="background:' + r[2] + '"></i>' : "";
      return "<dt>" + sw + esc(r[0]) + "</dt><dd>" + esc(r[1]) + "</dd>";
    }).join("") + "</dl>";
  }

  /* ------------------------------------------------------------------ chart shell */

  function Chart(fig) {
    this.fig = fig;
    this.data = JSON.parse(fig.querySelector(".chart-payload").textContent);
    this.canvas = fig.querySelector(".chart-canvas");
    this.controls = fig.querySelector(".chart-controls");
    var side = document.getElementById(fig.getAttribute("aria-describedby"));
    this.ro = side;
    this.roDefault = side.querySelector(".readout-default");
    this.roLive = side.querySelector(".readout-live");
    this.sel = -1;
    this.drawn = false;
  }
  Chart.prototype.show = function (html) {
    this.roLive.innerHTML = html;
    this.roLive.hidden = false;
    this.roDefault.hidden = true;
  };
  Chart.prototype.clear = function () {
    if (document.activeElement === this.fig) return;
    this.roLive.hidden = true;
    this.roDefault.hidden = false;
    this.sel = -1;
    this.highlight(-1);
  };
  Chart.prototype.width = function () { return Math.max(300, Math.floor(this.canvas.clientWidth)); };
  Chart.prototype.svg = function (w, h, label) {
    this.canvas.innerHTML = "";
    var s = el("svg", { viewBox: "0 0 " + w + " " + h, width: w, height: h, "aria-hidden": "true", focusable: "false" }, this.canvas);
    this.root = s;
    return s;
  };
  Chart.prototype.buttons = function (names, current, onPick) {
    var self = this;
    this.controls.hidden = false;
    this.controls.innerHTML = "";
    this.controls.setAttribute("role", "group");
    this.controls.setAttribute("aria-label", "Choose data set");
    names.forEach(function (n) {
      var b = document.createElement("button");
      b.type = "button";
      b.textContent = n;
      b.setAttribute("aria-pressed", n === current ? "true" : "false");
      b.addEventListener("click", function () {
        Array.prototype.forEach.call(self.controls.children, function (c) { c.setAttribute("aria-pressed", "false"); });
        b.setAttribute("aria-pressed", "true");
        onPick(n);
      });
      self.controls.appendChild(b);
    });
  };
  Chart.prototype.bindKeys = function (count, cols) {
    var self = this;
    cols = cols || count;
    this.fig.onkeydown = function (e) {
      var i = self.sel < 0 ? 0 : self.sel, k = e.key;
      if (k === "ArrowRight") i += 1;
      else if (k === "ArrowLeft") i -= 1;
      else if (k === "ArrowDown") i += cols === count ? 1 : cols;
      else if (k === "ArrowUp") i -= cols === count ? 1 : cols;
      else if (k === "Home") i = 0;
      else if (k === "End") i = count - 1;
      else if (k === "Escape") { self.fig.blur(); self.clear(); return; }
      else return;
      e.preventDefault();
      self.pick(Math.max(0, Math.min(count - 1, i)));
    };
    this.fig.onfocus = function () { if (self.sel < 0) self.pick(self.defaultIndex || 0); };
    this.fig.onblur = function () { self.clear(); };
  };
  Chart.prototype.pick = function (i) { this.sel = i; this.highlight(i); this.show(this.describe(i)); };
  Chart.prototype.animate = function () {
    if (reduceMotion || this.drawn || !this.root) { this.drawn = true; return; }
    this.drawn = true;
    var s = this.root;
    s.classList.add("anim");
    s.getBoundingClientRect();
    requestAnimationFrame(function () { requestAnimationFrame(function () { s.classList.remove("anim"); }); });
  };

  /* ------------------------------------------------------------------ waterfall */

  function Waterfall(fig) { Chart.call(this, fig); this.defaultIndex = 0; }
  Waterfall.prototype = Object.create(Chart.prototype);
  Waterfall.prototype.render = function () {
    var d = this.data.steps, W = this.width(), narrow = W < 520;
    var H = narrow ? 300 : 340, m = { t: 34, r: narrow ? 58 : 92, b: 40, l: 44 };
    var s = this.svg(W, H), self = this;
    var bnh = d[d.length - 1].bnh, vmax = 0;
    d.forEach(function (x) { vmax = Math.max(vmax, x.total, x.bnh); });
    var y = lin(0, vmax * 1.12, H - m.b, m.t), n = d.length;
    var band = (W - m.l - m.r) / n, bw = Math.min(band * 0.56, 84);
    ticks(0, vmax * 1.12, 5).forEach(function (v) {
      el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), "class": v === 0 ? "axis" : "grid" }, s);
      txt(s, m.l - 8, y(v) + 4, pct(v), { "text-anchor": "end", "class": "t tick" });
    });
    el("line", { x1: m.l, x2: W - m.r + 4, y1: y(bnh), y2: y(bnh), "class": "ref" }, s);
    var shortLabels = band < 124;
    this.bars = [];
    d.forEach(function (x, i) {
      var prev = i === 0 ? 0 : d[i - 1].total, lo = Math.min(prev, x.total), hi = Math.max(prev, x.total);
      var cx = m.l + band * i + band / 2, g = el("g", { "class": "mark" }, s);
      var fill = i === 0 ? C.annot : C.strategy;
      var r = el("rect", { x: cx - bw / 2, y: y(hi), width: bw, height: Math.max(1.5, y(lo) - y(hi)), fill: fill, "class": "bar" }, g);
      if (i > 0) el("line", { x1: cx - band + bw / 2, x2: cx - bw / 2, y1: y(prev), y2: y(prev), "class": "connector" }, s);
      // keep labels clear of the buy-and-hold line so no figure reads as struck through
      var top = y(hi), lineY = y(bnh), base = top;
      if (top > lineY - 30 && top < lineY + 34) base = Math.min(top, lineY - 6);
      txt(g, cx, base - (i === 0 ? 8 : 22), pct(x.total, 1), { "text-anchor": "middle", "class": "t val" });
      if (i > 0) txt(g, cx, base - 8, pp(x.total - prev), { "text-anchor": "middle", "class": "t sub removed" });
      txt(s, cx, H - m.b + 18, shortLabels ? x.short : x.label, { "text-anchor": "middle", "class": "t cat" });
      el("rect", { x: m.l + band * i, y: m.t, width: band, height: H - m.t - m.b, fill: "transparent", "class": "hit" }, g)
        .addEventListener("pointerenter", function () { self.pick(i); });
      self.bars.push(g);
    });
    txt(s, W - m.r + 8, y(bnh) - 3, "Buy & hold", { "class": "t ref-label" });
    txt(s, W - m.r + 8, y(bnh) + 11, pct(bnh, 1), { "class": "t ref-label strong" });
    s.addEventListener("pointerleave", function () { self.clear(); });
    this.bindKeys(n);
  };
  Waterfall.prototype.highlight = function (i) {
    (this.bars || []).forEach(function (g, j) { g.classList.toggle("dim", i >= 0 && j !== i); });
  };
  Waterfall.prototype.describe = function (i) {
    var x = this.data.steps[i], prev = i > 0 ? this.data.steps[i - 1].total : null;
    var head = '<p class="ro-head">' + (i === 0 ? "Step 0 · the claim" : "Step " + i + " · " + esc(x.label)) + "</p>";
    var what = '<p class="ro-what">' + esc(x.what) + "</p>";
    var r = [["EMA50/200", pct(x.total, 1), i === 0 ? C.annot : C.strategy]];
    if (prev !== null) r.push(["Bias removed", pp(x.total - prev), C.annot]);
    r.push(["Buy & hold", pct(x.bnh, 1), C.bnh], ["Max drawdown", pct(x.mdd, 1) + " vs " + pct(x.bnh_mdd, 1)],
      ["Sharpe", num(x.sharpe) + " vs " + num(x.bnh_sharpe)], ["Round trips", String(x.trips)]);
    return head + what + rows(r);
  };

  /* ------------------------------------------------------------------ equity + drawdown */

  function Equity(fig) {
    Chart.call(this, fig);
    this.key = this.data.order[0];
    var self = this;
    if (this.data.order.length > 1) this.buttons(this.data.order, this.key, function (k) { self.key = k; self.sel = -1; self.render(); self.summary(); });
    this.summary();
  }
  Equity.prototype = Object.create(Chart.prototype);
  Equity.prototype.summary = function () {
    var st = this.data.series[this.key].stats;
    if (!st || st.sharpe === undefined) return;
    this.roDefault.innerHTML = '<p class="ro-head">' + esc(this.key) + " · " + esc(st["class"]) + "</p>" +
      rows([["Sharpe", num(st.sharpe) + " vs " + num(st.bnh_sharpe)], ["CAGR", pct(st.cagr, 1) + " vs " + pct(st.bnh_cagr, 1)],
        ["Max drawdown", pct(st.mdd) + " vs " + pct(st.bnh_mdd)]]) +
      '<p class="ro-what">EMA50/200 first, buy-and-hold second. Point along the chart to read a date.</p>';
    if (this.roLive.hidden === false && this.sel < 0) this.clear();
  };
  Equity.prototype.render = function () {
    var S = this.data.series[this.key], names = this.data.names, keys = Object.keys(S.lines);
    var W = this.width(), narrow = W < 520, H = narrow ? 330 : 380, m = { t: 12, r: 12, b: 26, l: 46 };
    var split = Math.round((H - m.t - m.b) * 0.68), gap = 18;
    var s = this.svg(W, H), self = this, n = S.dates.length;
    var dd = {}, lo = Infinity, hi = -Infinity, ddmin = 0;
    keys.forEach(function (k) {
      var peak = 1, arr = S.lines[k];
      dd[k] = arr.map(function (v) { if (v === null) return null; peak = Math.max(peak, v); var d = v / peak - 1; ddmin = Math.min(ddmin, d); return d; });
      arr.forEach(function (v) { if (v !== null) { lo = Math.min(lo, v); hi = Math.max(hi, v); } });
    });
    var x = lin(0, n - 1, m.l, W - m.r);
    var ly = lin(Math.log(lo * 0.92), Math.log(hi * 1.08), m.t + split, m.t);
    var dy = lin(ddmin * 1.05, 0, H - m.b, m.t + split + gap);
    // log ticks
    var cand = [];
    for (var e = -2; e <= 5; e++) [1, 2, 5].forEach(function (b) { cand.push(b * Math.pow(10, e)); });
    var lt = cand.filter(function (v) { return v >= lo * 0.92 && v <= hi * 1.08; });
    if (lt.length > 6) lt = lt.filter(function (v) { return String(v).charAt(0) === "1"; });
    lt.forEach(function (v) {
      el("line", { x1: m.l, x2: W - m.r, y1: ly(Math.log(v)), y2: ly(Math.log(v)), "class": v === 1 ? "axis" : "grid" }, s);
      txt(s, m.l - 8, ly(Math.log(v)) + 4, (v >= 1 ? v.toLocaleString("en-US") : v) + "×", { "text-anchor": "end", "class": "t tick" });
    });
    ticks(ddmin * 1.05, 0, 3).forEach(function (v) {
      el("line", { x1: m.l, x2: W - m.r, y1: dy(v), y2: dy(v), "class": v === 0 ? "axis" : "grid" }, s);
      txt(s, m.l - 8, dy(v) + 4, pct(v), { "text-anchor": "end", "class": "t tick" });
    });
    // year ticks
    var lastY = null, years = [];
    S.dates.forEach(function (d, i) { var yr = +d.slice(0, 4); if (yr !== lastY) { years.push([yr, i]); lastY = yr; } });
    var every = Math.ceil(years.length / (narrow ? 4 : 7));
    years.forEach(function (p, j) { if (j % every === 0 && p[1] > 0) txt(s, x(p[1]), H - 8, String(p[0]), { "text-anchor": "middle", "class": "t tick" }); });
    var order = keys.slice().sort(function (a, b) { return (a === "bnh" || a === "sixty_forty") ? -1 : 1; });
    order.forEach(function (k) {
      var col = SERIES_COLOR[k] || C.ink2, p1 = "", p2 = "", started = false;
      S.lines[k].forEach(function (v, i) {
        if (v === null) { started = false; return; }
        p1 += (started ? "L" : "M") + x(i).toFixed(1) + " " + ly(Math.log(v)).toFixed(1);
        p2 += (started ? "L" : "M") + x(i).toFixed(1) + " " + dy(dd[k][i]).toFixed(1);
        started = true;
      });
      el("path", { d: p1, stroke: col, "class": "line" + (k === "bnh" || k === "sixty_forty" ? " bench" : ""), pathLength: 1 }, s);
      el("path", { d: p2, stroke: col, "class": "line thin" + (k === "bnh" || k === "sixty_forty" ? " bench" : ""), pathLength: 1 }, s);
    });
    txt(s, m.l + 4, m.t + split + gap - 4, "Drawdown", { "class": "t panel" });
    // legend (inline, top-left)
    var lx = m.l + 6;
    keys.forEach(function (k) {
      var g = el("g", {}, s);
      el("line", { x1: lx, x2: lx + 16, y1: m.t + 10, y2: m.t + 10, stroke: SERIES_COLOR[k], "class": "legend-line" }, g);
      var t = txt(g, lx + 22, m.t + 14, names[k], { "class": "t legend" });
      lx += 30 + names[k].length * 6.6;
    });
    this.cross = el("line", { y1: m.t, y2: H - m.b, "class": "cross", visibility: "hidden" }, s);
    this.dots = keys.map(function (k) {
      return [el("circle", { r: 3.5, fill: SERIES_COLOR[k], "class": "dot", visibility: "hidden" }, s),
        el("circle", { r: 3, fill: SERIES_COLOR[k], "class": "dot", visibility: "hidden" }, s)];
    });
    this.geom = { x: x, ly: ly, dy: dy, dd: dd, keys: keys, S: S };
    var hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, s);
    hit.addEventListener("pointermove", function (ev) {
      var r = s.getBoundingClientRect(), px = (ev.clientX - r.left) * (W / r.width);
      self.pick(Math.max(0, Math.min(n - 1, Math.round(x.invert(px)))));
    });
    s.addEventListener("pointerleave", function () { self.clear(); });
    this.defaultIndex = n - 1;
    this.bindKeys(n);
    var fig = this.fig, base = this.fig.onkeydown;
    fig.onkeydown = function (e) {
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        e.preventDefault();
        var step = e.shiftKey ? 52 : 4, i = self.sel < 0 ? n - 1 : self.sel;
        self.pick(Math.max(0, Math.min(n - 1, i + (e.key === "ArrowRight" ? step : -step))));
      } else base(e);
    };
  };
  Equity.prototype.highlight = function (i) {
    var g = this.geom;
    if (!g) return;
    var vis = i >= 0 ? "visible" : "hidden";
    this.cross.setAttribute("visibility", vis);
    if (i < 0) { this.dots.forEach(function (p) { p[0].setAttribute("visibility", "hidden"); p[1].setAttribute("visibility", "hidden"); }); return; }
    var X = g.x(i);
    this.cross.setAttribute("x1", X); this.cross.setAttribute("x2", X);
    this.dots.forEach(function (p, j) {
      var v = g.S.lines[g.keys[j]][i];
      var on = v === null ? "hidden" : "visible";
      p[0].setAttribute("visibility", on); p[1].setAttribute("visibility", on);
      if (v !== null) {
        p[0].setAttribute("cx", X); p[0].setAttribute("cy", g.ly(Math.log(v)));
        p[1].setAttribute("cx", X); p[1].setAttribute("cy", g.dy(g.dd[g.keys[j]][i]));
      }
    });
  };
  Equity.prototype.describe = function (i) {
    var g = this.geom, names = this.data.names;
    var r = g.keys.map(function (k) {
      var v = g.S.lines[k][i];
      return [names[k], v === null ? "–" : v.toFixed(2) + "× · " + pct(g.dd[k][i]) + " dd", SERIES_COLOR[k]];
    });
    return '<p class="ro-head">' + esc(this.key === "Portfolios" ? "Week ending " : this.key + " · week ending ") + dateLabel(g.S.dates[i]) + "</p>" +
      rows(r) + '<p class="ro-what">Growth of $1 and drawdown from the running peak.</p>';
  };

  /* ------------------------------------------------------------------ heatmap */

  function mix(a, b, t) {
    var pa = [1, 3, 5].map(function (o) { return parseInt(a.substr(o, 2), 16); });
    var pb = [1, 3, 5].map(function (o) { return parseInt(b.substr(o, 2), 16); });
    return "rgb(" + pa.map(function (v, j) { return Math.round(v + (pb[j] - v) * t); }).join(",") + ")";
  }
  function Heatmap(fig) {
    Chart.call(this, fig);
    this.key = this.data.order[0];
    var self = this;
    this.buttons(this.data.order, this.key, function (k) { self.key = k; self.sel = -1; self.render(); self.summary(); self.animate(); });
    this.summary();
  }
  Heatmap.prototype = Object.create(Chart.prototype);
  Heatmap.prototype.summary = function () {
    var A = this.data.assets[this.key];
    this.roDefault.innerHTML = '<p class="ro-head">' + esc(this.key) + " · " + A.cells.length + " crossovers</p>" +
      rows([["Buy & hold Sharpe", num(A.bnh_sharpe)], ["Overfit probability", pct(A.pbo)], ["Best cell beats B&H", "p = " + num(A.dsr) + " (deflated)"],
        ["Window from", A.start ? dateLabel(A.start) : "–"]]) +
      '<p class="ro-what">Blue beats buy-and-hold, ochre trails it; the outlined cell is 50/200. Point at a cell or use the arrow keys.</p>';
  };
  Heatmap.prototype.render = function () {
    var A = this.data.assets[this.key], cells = A.cells, self = this;
    var fasts = [], slows = [];
    cells.forEach(function (c) { if (fasts.indexOf(c.fast) < 0) fasts.push(c.fast); if (slows.indexOf(c.slow) < 0) slows.push(c.slow); });
    fasts.sort(function (a, b) { return a - b; }); slows.sort(function (a, b) { return a - b; });
    var W = this.width(), narrow = W < 520, m = { t: 8, r: 4, b: 40, l: 52 };
    var cw = (W - m.l - m.r) / fasts.length, ch = narrow ? 30 : 36, H = m.t + ch * slows.length + m.b;
    var s = this.svg(W, H), lo = Infinity, hi = -Infinity;
    cells.forEach(function (c) { lo = Math.min(lo, c.sharpe); hi = Math.max(hi, c.sharpe); });
    var span = Math.max(hi - A.bnh_sharpe, A.bnh_sharpe - lo, 0.05);
    this.cells = []; this.cols = fasts.length; this.grid = {};
    slows.forEach(function (sl, r) { txt(s, m.l - 8, m.t + ch * r + ch / 2 + 4, String(sl), { "text-anchor": "end", "class": "t tick" }); });
    fasts.forEach(function (f, c) { txt(s, m.l + cw * c + cw / 2, H - m.b + 16, String(f), { "text-anchor": "middle", "class": "t tick" }); });
    txt(s, m.l + (W - m.l - m.r) / 2, H - 6, "Fast EMA (bars)", { "text-anchor": "middle", "class": "t axis-title" });
    var yt = txt(s, 12, m.t + ch * slows.length / 2, "Slow EMA", { "text-anchor": "middle", "class": "t axis-title", transform: "rotate(-90 12 " + (m.t + ch * slows.length / 2) + ")" });
    var ordered = [];
    slows.forEach(function (sl, r) {
      fasts.forEach(function (f, c) {
        var cell = null;
        cells.forEach(function (q) { if (q.fast === f && q.slow === sl) cell = q; });
        if (!cell) return;
        var t = (cell.sharpe - A.bnh_sharpe) / span;
        var fill = t >= 0 ? mix(C.paper, C.strategy, Math.min(1, 0.12 + t * 0.88)) : mix(C.paper, C.sand, Math.min(1, 0.12 - t * 0.7));
        var g = el("g", { "class": "mark cellg" }, s);
        el("rect", { x: m.l + cw * c + 1, y: m.t + ch * r + 1, width: cw - 2, height: ch - 2, fill: fill, "class": "cell" }, g);
        var dark = t > 0.55;
        if (cw > 30) txt(g, m.l + cw * c + cw / 2, m.t + ch * r + ch / 2 + 4, num(cell.sharpe), { "text-anchor": "middle", "class": "t cellv" + (dark ? " on-dark" : "") });
        var idx = ordered.length;
        g.addEventListener("pointerenter", function () { self.pick(idx); });
        ordered.push({ cell: cell, g: g, r: r, c: c });
        if (cell.fast === self.data.mark[0] && cell.slow === self.data.mark[1]) {
          el("rect", { x: m.l + cw * c + 1, y: m.t + ch * r + 1, width: cw - 2, height: ch - 2, "class": "mark-outline" }, s);
          self.defaultIndex = idx;
        }
      });
    });
    this.ordered = ordered;
    this.focusRect = el("rect", { "class": "cell-focus", visibility: "hidden", width: cw - 2, height: ch - 2 }, s);
    this.cw = cw; this.ch = ch; this.m = m;
    s.addEventListener("pointerleave", function () { self.clear(); });
    this.bindKeys(ordered.length);
    var fig = this.fig;
    fig.onkeydown = function (e) {
      var cur = self.ordered[self.sel < 0 ? self.defaultIndex : self.sel], tr = cur.r, tc = cur.c;
      if (e.key === "ArrowRight") tc++; else if (e.key === "ArrowLeft") tc--; else if (e.key === "ArrowDown") tr++; else if (e.key === "ArrowUp") tr--;
      else if (e.key === "Escape") { fig.blur(); return; } else return;
      e.preventDefault();
      for (var j = 0; j < self.ordered.length; j++) if (self.ordered[j].r === tr && self.ordered[j].c === tc) { self.pick(j); return; }
    };
  };
  Heatmap.prototype.highlight = function (i) {
    if (!this.focusRect) return;
    if (i < 0) { this.focusRect.setAttribute("visibility", "hidden"); return; }
    var o = this.ordered[i];
    this.focusRect.setAttribute("x", this.m.l + this.cw * o.c + 1);
    this.focusRect.setAttribute("y", this.m.t + this.ch * o.r + 1);
    this.focusRect.setAttribute("visibility", "visible");
  };
  Heatmap.prototype.describe = function (i) {
    var c = this.ordered[i].cell, A = this.data.assets[this.key];
    var tb = c.fast === this.data.mark[0] && c.slow === this.data.mark[1];
    return '<p class="ro-head">' + esc(this.key) + " · EMA" + c.fast + "/" + c.slow + (tb ? " (textbook)" : "") + "</p>" +
      rows([["Sharpe", num(c.sharpe)], ["vs buy & hold", num(c.sharpe - A.bnh_sharpe, 2, true)], ["CAGR", pct(c.cagr, 1)],
        ["Max drawdown", pct(c.max_drawdown, 1)], ["Trades", String(c.n_trades)]]);
  };

  /* ------------------------------------------------------------------ forest (CI per row) */

  function Forest(fig) { Chart.call(this, fig); this.defaultIndex = 0; }
  Forest.prototype = Object.create(Chart.prototype);
  Forest.prototype.render = function () {
    var R = this.data.rows, W = this.width(), narrow = W < 520, self = this;
    var rh = narrow ? 24 : 26, m = { t: 6, r: 14, b: 34, l: narrow ? 64 : 76 }, H = m.t + rh * R.length + m.b;
    var s = this.svg(W, H), lo = 0, hi = 0;
    R.forEach(function (r) { lo = Math.min(lo, r.lo); hi = Math.max(hi, r.hi); });
    var x = lin(lo - 0.04, hi + 0.04, m.l, W - m.r);
    ticks(lo, hi, narrow ? 4 : 6).forEach(function (v) {
      el("line", { x1: x(v), x2: x(v), y1: m.t, y2: H - m.b, "class": Math.abs(v) < 1e-9 ? "zero" : "grid" }, s);
      txt(s, x(v), H - m.b + 16, num(v, 1, true), { "text-anchor": "middle", "class": "t tick" });
    });
    txt(s, m.l + (W - m.l - m.r) / 2, H - 4, "Sharpe difference (EMA50/200 − buy & hold)", { "text-anchor": "middle", "class": "t axis-title" });
    this.rowsG = R.map(function (r, i) {
      var cy = m.t + rh * i + rh / 2, g = el("g", { "class": "mark" }, s);
      el("rect", { x: 0, y: cy - rh / 2, width: W, height: rh, "class": "rowband" }, g);
      txt(g, m.l - 10, cy + 4, r.asset, { "text-anchor": "end", "class": "t cat-left" });
      el("line", { x1: x(r.lo), x2: x(r.hi), y1: cy, y2: cy, "class": "whisker" }, g);
      el("circle", { cx: x(r.diff), cy: cy, r: 4.5, "class": "pt" + (r.q !== null && r.q < 0.05 ? " solid" : "") }, g);
      g.addEventListener("pointerenter", function () { self.pick(i); });
      return g;
    });
    s.addEventListener("pointerleave", function () { self.clear(); });
    this.bindKeys(R.length);
  };
  Forest.prototype.highlight = function (i) { (this.rowsG || []).forEach(function (g, j) { g.classList.toggle("on", j === i); }); };
  Forest.prototype.describe = function (i) {
    var r = this.data.rows[i];
    return '<p class="ro-head">' + esc(r.asset) + " · " + Math.round(r.years) + " years</p>" +
      rows([["Sharpe", num(r.sharpe) + " vs " + num(r.bnh_sharpe)], ["Difference", num(r.diff, 2, true), C.strategy],
        ["95% interval", num(r.lo) + " to " + num(r.hi)], ["p (one-sided)", num(r.p)], ["BH q", num(r.q)]]) +
      '<p class="ro-what">' + (r.lo < 0 && r.hi > 0 ? "The interval includes zero: not distinguishable from luck." : "The interval excludes zero.") + "</p>";
  };

  /* ------------------------------------------------------------------ crises dot plot */

  function Crises(fig) { Chart.call(this, fig); this.defaultIndex = 0; }
  Crises.prototype = Object.create(Chart.prototype);
  Crises.prototype.render = function () {
    var R = this.data.rows, W = this.width(), narrow = W < 520, self = this;
    var rh = narrow ? 40 : 32, m = { t: 6, r: 14, b: 34, l: narrow ? 8 : 176 }, H = m.t + rh * R.length + m.b;
    var s = this.svg(W, H), lo = 0, hi = 0;
    R.forEach(function (r) { lo = Math.min(lo, r.bnh, r.rule); hi = Math.max(hi, r.bnh, r.rule); });
    var x = lin(lo - 0.04, Math.max(hi, 0) + 0.04, m.l, W - m.r);
    ticks(lo, Math.max(hi, 0), 5).forEach(function (v) {
      el("line", { x1: x(v), x2: x(v), y1: m.t, y2: H - m.b, "class": Math.abs(v) < 1e-9 ? "zero" : "grid" }, s);
      txt(s, x(v), H - m.b + 16, pct(v), { "text-anchor": "middle", "class": "t tick" });
    });
    txt(s, m.l + (W - m.l - m.r) / 2, H - 4, "Return inside the episode", { "text-anchor": "middle", "class": "t axis-title" });
    this.rowsG = R.map(function (r, i) {
      var top = m.t + rh * i, cy = narrow ? top + rh - 12 : top + rh / 2, g = el("g", { "class": "mark" }, s);
      el("rect", { x: 0, y: top, width: W, height: rh, "class": "rowband" }, g);
      if (narrow) txt(g, m.l, top + 13, r.crisis, { "class": "t cat-left small" });
      else txt(g, m.l - 12, cy + 4, r.crisis, { "text-anchor": "end", "class": "t cat-left" });
      el("line", { x1: x(r.bnh), x2: x(r.rule), y1: cy, y2: cy, "class": "link" }, g);
      el("circle", { cx: x(r.bnh), cy: cy, r: 4.5, fill: C.bnh, "class": "dotb" }, g);
      el("circle", { cx: x(r.rule), cy: cy, r: 4.5, fill: C.strategy, "class": "dots" }, g);
      g.addEventListener("pointerenter", function () { self.pick(i); });
      return g;
    });
    s.addEventListener("pointerleave", function () { self.clear(); });
    this.bindKeys(R.length);
  };
  Crises.prototype.highlight = Forest.prototype.highlight;
  Crises.prototype.describe = function (i) {
    var r = this.data.rows[i];
    var list = [["Buy & hold", pct(r.bnh, 1), C.bnh], ["EMA50/200", pct(r.rule, 1), C.strategy], ["Difference", pp(r.rule - r.bnh)]];
    if (r.exit) list.push(["Exited", dateLabel(r.exit)]);
    if (r.reentry) list.push(["Re-entered", dateLabel(r.reentry)]);
    if (r.missed !== null && r.missed !== undefined && r.reentry) list.push(["Rebound missed", pct(r.missed)]);
    if (r.edge !== null && r.edge !== undefined) list.push(["Edge to recovery", pp(r.edge)]);
    return '<p class="ro-head">' + esc(r.crisis) + "</p>" + '<p class="ro-what">' + dateLabel(r.start) + " to " + dateLabel(r.end) + "</p>" + rows(list) +
      (r.exit ? "" : '<p class="ro-what">No exit before the trough: the fall was faster than the filter.</p>');
  };

  /* ------------------------------------------------------------------ boot */

  var KINDS = { waterfall: Waterfall, equity: Equity, portfolio: Equity, heatmap: Heatmap, forest: Forest, crises: Crises };
  var charts = [];
  Array.prototype.forEach.call(document.querySelectorAll("figure[data-chart]"), function (fig) {
    var K = KINDS[fig.getAttribute("data-chart")];
    if (!K) return;
    try {
      var c = new K(fig);
      c.render();
      fig.classList.add("live");
      charts.push(c);
    } catch (err) {
      if (window.console) console.error("chart failed", err);
    }
  });
  if ("IntersectionObserver" in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (!en.isIntersecting) return;
        charts.forEach(function (c) { if (c.fig === en.target) c.animate(); });
        io.unobserve(en.target);
      });
    }, { threshold: 0.35 });
    charts.forEach(function (c) { io.observe(c.fig); });
  }
  var lastW = window.innerWidth, t = null;
  window.addEventListener("resize", function () {
    if (Math.abs(window.innerWidth - lastW) < 8) return;
    lastW = window.innerWidth;
    clearTimeout(t);
    t = setTimeout(function () { charts.forEach(function (c) { c.sel = -1; c.render(); c.drawn = true; }); }, 120);
  });
})();
