// A tiny, cinematic transformer.
//
// Setup: 4 tokens × d_model = 8 × 2 heads (d_head = 4), single layer,
// self-attention with Q = K = V = E (weight matrices folded into identity so
// the displayed numbers stay legible — the point is to show dataflow, not
// teach W_Q). Forward pass builds left→right; backward retraces it in reverse,
// tinted crimson.
//
// Performance rules (lessons from v1's stutter):
//   • No color-mix in the CSS hot path. Cell fill colors are precomputed at
//     build time; per-frame work is just opacity on overlay rects.
//   • textContent writes are gated — we skip writes when the displayed value
//     has changed by < 0.005.
//   • Style writes are cached (last-written value stored on the node) so we
//     never re-write identical values.
//   • Fade in/out of whole groups uses transform + opacity; individual cells
//     only animate opacity.
//
// Everything is pure SVG + rAF. No libraries.

// ────────────────────────────────────────────────────────────────────
// math
// ────────────────────────────────────────────────────────────────────

function shape(m) { return [m.length, m[0].length]; }
function zeros(r, c) { return Array.from({ length: r }, () => new Array(c).fill(0)); }
function transpose(a) {
  const [r, c] = shape(a);
  const out = zeros(c, r);
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[j][i] = a[i][j];
  return out;
}
function matmul(a, b) {
  const [r, k] = shape(a);
  const [, c] = shape(b);
  const out = zeros(r, c);
  for (let i = 0; i < r; i++)
    for (let j = 0; j < c; j++) {
      let s = 0;
      for (let t = 0; t < k; t++) s += a[i][t] * b[t][j];
      out[i][j] = s;
    }
  return out;
}
function scaleMat(a, k) { return a.map((row) => row.map((v) => v * k)); }
function addMat(a, b) { return a.map((row, i) => row.map((v, j) => v + b[i][j])); }
function sliceCols(a, start, end) {
  return a.map((row) => row.slice(start, end));
}
function concatCols(parts) {
  const rows = parts[0].length;
  return Array.from({ length: rows }, (_, i) => parts.flatMap((p) => p[i]));
}
function softmaxRows(a) {
  return a.map((row) => {
    const m = Math.max(...row);
    const exps = row.map((v) => Math.exp(v - m));
    const s = exps.reduce((p, q) => p + q, 0);
    return exps.map((v) => v / s);
  });
}
function softmaxBack(dA, A) {
  return A.map((arow, i) => {
    const dot = arow.reduce((s, a, k) => s + a * dA[i][k], 0);
    return arow.map((a, j) => a * (dA[i][j] - dot));
  });
}

function precompute() {
  // Four tokens — loosely evocative of ML research ("read the RL paper").
  const tokens = ["read", "the", "rl", "paper"];

  // 4 × 8 embedding matrix, hand-tuned so the resulting attention is varied.
  const E = [
    [ 0.6, -0.2,  0.8,  0.1,  0.3,  0.5, -0.4,  0.2],
    [-0.1,  0.7,  0.2, -0.3,  0.6,  0.1,  0.4,  0.5],
    [ 0.4,  0.3, -0.5,  0.7, -0.2,  0.6,  0.3, -0.1],
    [ 0.2, -0.4,  0.3,  0.5,  0.1, -0.3,  0.7,  0.4],
  ];
  const dHead = 4;
  const nHeads = 2;
  const scale = 1 / Math.sqrt(dHead);

  // Per-head Q/K/V = the relevant column slice of E (our simplification).
  const heads = [];
  for (let h = 0; h < nHeads; h++) {
    const Qh = sliceCols(E, h * dHead, (h + 1) * dHead);
    const Kh = Qh, Vh = Qh;
    const Sh = scaleMat(matmul(Qh, transpose(Kh)), scale);
    const Ah = softmaxRows(Sh);
    const Oh = matmul(Ah, Vh);
    heads.push({ Q: Qh, K: Kh, V: Vh, S: Sh, A: Ah, O: Oh });
  }

  // Concat head outputs along the column axis.
  const C = concatCols(heads.map((h) => h.O));

  // Simplification: W_o = I, so final output = C.
  const O = C.map((r) => r.slice());

  // Backward. Use L = ½‖O‖² so dL/dO = O.
  const dO = O.map((r) => r.slice());
  const dC = dO.map((r) => r.slice());
  // Split dC back to per-head outputs.
  const dOh = [];
  for (let h = 0; h < nHeads; h++) {
    dOh.push(sliceCols(dC, h * dHead, (h + 1) * dHead));
  }
  // Per-head backward.
  for (let h = 0; h < nHeads; h++) {
    const head = heads[h];
    const dVh = matmul(transpose(head.A), dOh[h]);
    const dAh = matmul(dOh[h], transpose(head.V));
    const dSh = scaleMat(softmaxBack(dAh, head.A), scale);
    const dQh = matmul(dSh, head.K);
    const dKh = matmul(transpose(dSh), head.Q);
    head.dV = dVh;
    head.dA = dAh;
    head.dS = dSh;
    head.dQ = dQh;
    head.dK = dKh;
    head.dO = dOh[h];
  }

  // dE = concat over columns of (dQh + dKh + dVh) per head.
  const dEparts = heads.map((h) => addMat(addMat(h.dQ, h.dK), h.dV));
  const dE = concatCols(dEparts);

  return { tokens, E, dE, heads, C, O, dO, dC, nHeads, dHead };
}

// ────────────────────────────────────────────────────────────────────
// svg helpers
// ────────────────────────────────────────────────────────────────────

const NS = "http://www.w3.org/2000/svg";

function sv(name, attrs = {}, text) {
  const e = document.createElementNS(NS, name);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text != null) e.textContent = text;
  return e;
}
function fmt(v) {
  const n = Number(v).toFixed(2);
  return n === "-0.00" ? "0.00" : n;
}
function clamp01(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
function ease(u) { return u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2; }
function lerp(a, b, u) { return a + (b - a) * u; }
function smoothstep(e0, e1, x) {
  const t = clamp01((x - e0) / (e1 - e0));
  return t * t * (3 - 2 * t);
}

// Precomputed palette. Index by "lane": embed/q/k/v/s/a/o/c/out.
const LANE_RGB = {
  base: [20, 24, 36],
  embed: [94, 234, 212],   // teal
  q:     [94, 234, 212],   // teal
  k:     [125, 211, 252],  // sky
  v:     [167, 139, 250],  // violet
  s:     [125, 211, 252],  // sky (scores blend)
  a:     [94, 234, 212],   // teal (softmax = forward output)
  o:     [240, 231, 196],  // warm ivory — the "output" reads special
  c:     [240, 231, 196],  // concat same as outputs
  out:   [94, 234, 212],   // final out, teal
  bwd:   [239, 68, 68],    // crimson
};

// Produce an rgb string by blending base gray toward the lane color by t ∈ [0,1].
function laneFill(lane, t) {
  const [r0, g0, b0] = LANE_RGB.base;
  const [r1, g1, b1] = LANE_RGB[lane] || LANE_RGB.embed;
  const r = Math.round(r0 + (r1 - r0) * t);
  const g = Math.round(g0 + (g1 - g0) * t);
  const b = Math.round(b0 + (b1 - b0) * t);
  return `rgb(${r},${g},${b})`;
}
// value → fill intensity
function intensityOf(v) { return Math.min(1, Math.abs(v) * 1.7); }

// ────────────────────────────────────────────────────────────────────
// scene layout
// ────────────────────────────────────────────────────────────────────

const VB_W = 1440;
const VB_H = 620;
const CY = VB_H / 2;

// column x-centers — hand-tuned so every matrix has at least ~30px air
// between its edge and the next column's edge. The two 8-column matrices
// (embed, concat, out) are wider so they need more generous spacing.
const COL = {
  tokens:  90,
  embed:   270,
  scores:  580,
  attn:    750,
  headOut: 920,
  concat:  1130,
  out:     1340,
};

const HEAD_Y = [CY - 170, CY + 170]; // upper head y, lower head y

// ────────────────────────────────────────────────────────────────────
// scene construction
// ────────────────────────────────────────────────────────────────────

function makeDefs(svg) {
  const defs = sv("defs");
  // arrow heads
  for (const [id, color] of [
    ["tv-arrow-fwd", "#5eead4"],
    ["tv-arrow-bwd", "#ef4444"],
    ["tv-arrow-muted", "#2a3142"],
  ]) {
    const m = sv("marker", {
      id, viewBox: "0 0 10 10",
      refX: 9, refY: 5,
      markerWidth: 6, markerHeight: 6,
      orient: "auto-start-reverse",
    });
    m.appendChild(sv("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: color }));
    defs.appendChild(m);
  }
  svg.appendChild(defs);
  return defs;
}

function buildTokens(svg, tokens) {
  const g = sv("g", { class: "tv-tokens" });
  const yStep = 90;
  const y0 = CY - ((tokens.length - 1) * yStep) / 2;
  const items = tokens.map((tok, i) => {
    const y = y0 + i * yStep;
    const wrap = sv("g", { class: "tv-token", transform: `translate(${COL.tokens} ${y})`, opacity: 0 });
    const bg = sv("rect", {
      x: -52, y: -22, width: 104, height: 44, rx: 22, ry: 22,
      class: "tv-token__bg",
    });
    const bgBwd = sv("rect", {
      x: -52, y: -22, width: 104, height: 44, rx: 22, ry: 22,
      class: "tv-token__bg tv-token__bg--bwd",
      opacity: 0,
    });
    const txt = sv("text", {
      x: 0, y: 5, "text-anchor": "middle",
      class: "tv-token__txt",
    }, tok);
    wrap.appendChild(bg);
    wrap.appendChild(bgBwd);
    wrap.appendChild(txt);
    g.appendChild(wrap);
    return { g: wrap, bgBwd, y, _lastOp: 0, _lastTx: COL.tokens, _lastBwd: 0 };
  });
  svg.appendChild(g);
  return { g, items };
}

function buildMatrix(svg, mat, cfg) {
  // cfg: { cx, cy, cellW, cellH, lane, label, labelPos, mute }
  const [rows, cols] = [mat.length, mat[0].length];
  const w = cols * cfg.cellW;
  const h = rows * cfg.cellH;
  const x0 = cfg.cx - w / 2;
  const y0 = cfg.cy - h / 2;
  const g = sv("g", {
    class: `tv-mat tv-mat--${cfg.lane}${cfg.mute ? " tv-mat--mute" : ""}`,
    opacity: cfg.mute ? 0.08 : 0,
  });
  if (cfg.label) {
    const ly = cfg.labelPos === "bottom" ? y0 + h + 18 : y0 - 10;
    g.appendChild(sv("text", {
      x: cfg.cx, y: ly, "text-anchor": "middle",
      class: "tv-mat__label",
    }, cfg.label));
  }
  const cells = [];
  for (let i = 0; i < rows; i++) {
    const row = [];
    for (let j = 0; j < cols; j++) {
      const v = mat[i][j];
      const intensity = intensityOf(v);
      const fwdColor = laneFill(cfg.lane, intensity);
      const cg = sv("g", {
        class: "tv-cell",
        transform: `translate(${x0 + j * cfg.cellW} ${y0 + i * cfg.cellH})`,
      });
      const pad = 1.2;
      const cw = cfg.cellW - pad * 2;
      const ch = cfg.cellH - pad * 2;
      const base = sv("rect", {
        x: pad, y: pad, width: cw, height: ch,
        rx: 2, ry: 2,
        fill: "#14182a",
        stroke: "#242a3d",
        "stroke-width": 0.6,
        class: "tv-cell__base",
      });
      const fwd = sv("rect", {
        x: pad, y: pad, width: cw, height: ch,
        rx: 2, ry: 2,
        fill: fwdColor,
        class: "tv-cell__fwd",
        opacity: 0,
      });
      const bwd = sv("rect", {
        x: pad, y: pad, width: cw, height: ch,
        rx: 2, ry: 2,
        fill: "#ef4444",
        class: "tv-cell__bwd",
        opacity: 0,
      });
      const text = sv("text", {
        x: cfg.cellW / 2,
        y: cfg.cellH / 2 + 3.5,
        "text-anchor": "middle",
        class: "tv-cell__val",
        opacity: 0,
      }, "0.00");
      cg.appendChild(base);
      cg.appendChild(fwd);
      cg.appendChild(bwd);
      cg.appendChild(text);
      g.appendChild(cg);
      row.push({
        g: cg, fwd, bwd, text,
        val: v,
        intensity,
        _lastTxt: "0.00",
        _lastFwdOp: 0,
        _lastBwdOp: 0,
      });
    }
    cells.push(row);
  }
  svg.appendChild(g);
  return {
    g, cells,
    origin: [x0, y0],
    size: [w, h],
    cellSize: [cfg.cellW, cfg.cellH],
    center: [cfg.cx, cfg.cy],
    lane: cfg.lane,
    _lastOp: cfg.mute ? 0.08 : 0,
  };
}

function buildConn(parent, x1, y1, x2, y2, cls) {
  const len = Math.hypot(x2 - x1, y2 - y1);
  const line = sv("line", {
    x1, y1, x2, y2,
    class: `tv-conn ${cls || ""}`,
    "stroke-dasharray": `${len.toFixed(1)}`,
    "stroke-dashoffset": `${len.toFixed(1)}`,
  });
  const tip = sv("circle", {
    cx: x1, cy: y1, r: 3.4,
    class: "tv-conn__tip",
    opacity: 0,
  });
  parent.appendChild(line);
  parent.appendChild(tip);
  return {
    line, tip, len, x1, y1, x2, y2,
    _lastOffset: len,
    _lastArr: `idle`,
    _lastClass: `tv-conn ${cls || ""}`,
    _lastTipOp: 0,
    _lastTipCx: x1,
    _lastTipCy: y1,
    _lastTipCls: "tv-conn__tip",
    _mode: "idle",
  };
}

function buildDepthSilhouettes(svg) {
  // Two faint background panels suggesting depth — "this is one of N layers".
  const g = sv("g", { class: "tv-depth", "pointer-events": "none" });
  for (let i = 0; i < 2; i++) {
    const shift = (i + 1) * 12;
    const sx = shift;
    const sy = -shift;
    g.appendChild(sv("rect", {
      x: 30 + sx, y: 40 + sy,
      width: VB_W - 60, height: VB_H - 80,
      rx: 6, ry: 6,
      fill: "none",
      stroke: "#1b2030",
      "stroke-width": 1,
      opacity: 0.6 - i * 0.25,
    }));
  }
  svg.insertBefore(g, svg.firstChild);
}

function buildScene(root, data, labels) {
  const svg = root.querySelector(".tviz__svg");
  svg.innerHTML = "";
  svg.setAttribute("viewBox", `0 0 ${VB_W} ${VB_H}`);
  makeDefs(svg);
  buildDepthSilhouettes(svg);

  // tokens
  const tokens = buildTokens(svg, data.tokens);

  // E matrix (4 × 8)
  const embed = buildMatrix(svg, data.E, {
    cx: COL.embed, cy: CY,
    cellW: 24, cellH: 28,
    lane: "embed", label: "E  (4 × 8)",
  });

  // Per-head chains.
  const heads = data.heads.map((head, h) => ({
    s: buildMatrix(svg, head.S, {
      cx: COL.scores, cy: HEAD_Y[h],
      cellW: 32, cellH: 32,
      lane: "s", label: h === 0 ? "S₁ = Q₁K₁ᵀ/√d" : "S₂ = Q₂K₂ᵀ/√d",
    }),
    a: buildMatrix(svg, head.A, {
      cx: COL.attn, cy: HEAD_Y[h],
      cellW: 32, cellH: 32,
      lane: "a", label: h === 0 ? "A₁ = softmax" : "A₂ = softmax",
    }),
    o: buildMatrix(svg, head.O, {
      cx: COL.headOut, cy: HEAD_Y[h],
      cellW: 32, cellH: 32,
      lane: "o", label: h === 0 ? "O₁ = A₁V₁" : "O₂ = A₂V₂",
    }),
  }));

  // Concat (4 × 8)
  const concat = buildMatrix(svg, data.C, {
    cx: COL.concat, cy: CY,
    cellW: 24, cellH: 28,
    lane: "c", label: "concat",
  });

  // Final output (4 × 8)
  const out = buildMatrix(svg, data.O, {
    cx: COL.out, cy: CY,
    cellW: 24, cellH: 28,
    lane: "out", label: "O",
  });

  // Loss badge
  const lossG = sv("g", {
    class: "tv-loss",
    opacity: 0,
    transform: `translate(${COL.out} ${CY - 180}) scale(0.8)`,
  });
  lossG.appendChild(sv("circle", { r: 28, cx: 0, cy: 0, class: "tv-loss__ring" }));
  lossG.appendChild(sv("text", { x: 0, y: -1, "text-anchor": "middle", class: "tv-loss__big" }, "L"));
  lossG.appendChild(sv("text", { x: 0, y: 16, "text-anchor": "middle", class: "tv-loss__sub" }, "½‖O‖²"));
  svg.appendChild(lossG);

  // Connectors — drawn in a dedicated group so we can stack them appropriately.
  const connG = sv("g", { class: "tv-conns" });
  svg.insertBefore(connG, embed.g);

  // token → E
  const tokToE = data.tokens.map((_, i) => {
    const ty = tokens.items[i].y;
    const ey = embed.origin[1] + (i + 0.5) * embed.cellSize[1];
    return buildConn(connG, COL.tokens + 54, ty, embed.origin[0] - 4, ey, "tv-conn--fwd");
  });

  // E → each head's scores grid (fan)
  const eToHead = heads.map((h, idx) =>
    buildConn(
      connG,
      embed.origin[0] + embed.size[0] + 4, CY,
      h.s.origin[0] - 4, HEAD_Y[idx],
      "tv-conn--fwd tv-conn--q"
    )
  );
  // Within each head: S → A, A → O
  const sToA = heads.map((h) =>
    buildConn(
      connG,
      h.s.origin[0] + h.s.size[0] + 4, h.s.center[1],
      h.a.origin[0] - 4, h.a.center[1],
      "tv-conn--fwd tv-conn--a"
    )
  );
  const aToO = heads.map((h) =>
    buildConn(
      connG,
      h.a.origin[0] + h.a.size[0] + 4, h.a.center[1],
      h.o.origin[0] - 4, h.o.center[1],
      "tv-conn--fwd tv-conn--a"
    )
  );
  // Each head's O → concat
  const oToConcat = heads.map((h, idx) =>
    buildConn(
      connG,
      h.o.origin[0] + h.o.size[0] + 4, h.o.center[1],
      concat.origin[0] - 4, CY + (idx === 0 ? -20 : 20),
      "tv-conn--fwd tv-conn--v"
    )
  );
  // concat → out
  const concatToOut = buildConn(
    connG,
    concat.origin[0] + concat.size[0] + 4, CY,
    out.origin[0] - 4, CY,
    "tv-conn--fwd"
  );

  const conns = { tokToE, eToHead, sToA, aToO, oToConcat, concatToOut };

  return {
    root, svg, data, labels,
    tokens, embed, heads, concat, out, lossG, conns,
    caption: root.querySelector(".tviz__caption"),
    scrub: buildScrub(root, STAGES.length),
    _captionText: "",
    _lastStage: -1,
  };
}

function buildScrub(root, count) {
  const ol = root.querySelector(".tviz__scrub");
  if (!ol) return { dots: [] };
  ol.innerHTML = "";
  const dots = [];
  for (let i = 0; i < count; i++) {
    const li = document.createElement("li");
    li.className = "tviz__dot";
    ol.appendChild(li);
    dots.push(li);
  }
  return { dots };
}

// ────────────────────────────────────────────────────────────────────
// low-level setters (with dirty tracking)
// ────────────────────────────────────────────────────────────────────

function setOpGroup(rec, op) {
  if (rec._lastOp === op) return;
  rec._lastOp = op;
  rec.g.setAttribute("opacity", op.toFixed(3));
}
function setFwdOp(cell, op) {
  if (cell._lastFwdOp === op) return;
  cell._lastFwdOp = op;
  cell.fwd.setAttribute("opacity", op.toFixed(3));
}
function setBwdOp(cell, op) {
  if (cell._lastBwdOp === op) return;
  cell._lastBwdOp = op;
  cell.bwd.setAttribute("opacity", op.toFixed(3));
}
function setCellText(cell, val, reveal) {
  // Only write text when the value changed by >= 0.005 OR when reveal opacity jumps across threshold.
  const txt = fmt(val);
  if (txt !== cell._lastTxt) {
    cell._lastTxt = txt;
    cell.text.textContent = txt;
  }
  const op = reveal > 0.15 ? reveal : 0;
  if (cell.text._lastOp !== op) {
    cell.text._lastOp = op;
    cell.text.setAttribute("opacity", op.toFixed(3));
  }
}

// ── connectors ──
// Four states, each setter handles dirty tracking:
//   • setConnShoot(c, u, klass)     — ray travels source→dest; u=1 = fully lit.
//   • setConnShootBack(c, u, klass) — ray travels dest→source; u=1 = emptied.
//   • setConnSteady(c, klass)       — full beam, no tip, no animation.
//   • setConnIdle(c)                — invisible.
// Visual idea: a bright "head" dot flies along the connector while the line
// reveals behind it. When u=1 the dot fades out and the line is a stable beam.

function setConnClass(c, klass) {
  if (klass && klass !== c._lastClass) {
    c._lastClass = klass;
    c.line.setAttribute("class", klass);
  }
}
function setTip(c, cx, cy, op, cls) {
  if (op > 0) {
    if (Math.abs(cx - c._lastTipCx) > 0.4) {
      c._lastTipCx = cx;
      c.tip.setAttribute("cx", cx.toFixed(1));
    }
    if (Math.abs(cy - c._lastTipCy) > 0.4) {
      c._lastTipCy = cy;
      c.tip.setAttribute("cy", cy.toFixed(1));
    }
  }
  const cls2 = cls || "tv-conn__tip";
  if (cls2 !== c._lastTipCls) {
    c._lastTipCls = cls2;
    c.tip.setAttribute("class", cls2);
  }
  if (Math.abs(op - c._lastTipOp) > 0.015) {
    c._lastTipOp = op;
    c.tip.setAttribute("opacity", op.toFixed(3));
  }
}

function setConnShoot(c, u, klass) {
  const eu = ease(clamp01(u));
  // Line reveals source → dest.
  const off = c.len * (1 - eu);
  if (c._mode !== "shoot" || c._lastArr !== "full") {
    c.line.setAttribute("stroke-dasharray", c.len.toFixed(1));
    c._lastArr = "full";
    c._mode = "shoot";
  }
  if (Math.abs(off - c._lastOffset) > 0.3) {
    c._lastOffset = off;
    c.line.setAttribute("stroke-dashoffset", off.toFixed(1));
  }
  setConnClass(c, klass);
  // Tip rides the leading edge; fades in the last 15% of the shoot.
  const cx = c.x1 + (c.x2 - c.x1) * eu;
  const cy = c.y1 + (c.y2 - c.y1) * eu;
  const tipOp = eu < 0.85 ? 1 : Math.max(0, 1 - (eu - 0.85) / 0.15);
  setTip(c, cx, cy, tipOp, "tv-conn__tip");
}

function setConnShootBack(c, u, klass) {
  const eu = ease(clamp01(u));
  // Line is consumed from the dest end. Visible length shrinks from len to 0.
  const visible = c.len * (1 - eu);
  const arrKey = `bk-${visible.toFixed(1)}`;
  if (c._lastArr !== arrKey) {
    c._lastArr = arrKey;
    c.line.setAttribute(
      "stroke-dasharray",
      `${visible.toFixed(1)} ${c.len.toFixed(1)}`
    );
    if (c._lastOffset !== 0) {
      c._lastOffset = 0;
      c.line.setAttribute("stroke-dashoffset", "0");
    }
  }
  c._mode = "shootBack";
  setConnClass(c, klass);
  const cx = c.x2 + (c.x1 - c.x2) * eu;
  const cy = c.y2 + (c.y1 - c.y2) * eu;
  const tipOp = eu < 0.85 ? 1 : Math.max(0, 1 - (eu - 0.85) / 0.15);
  setTip(c, cx, cy, tipOp, "tv-conn__tip tv-conn__tip--bwd");
}

function setConnSteady(c, klass) {
  if (c._mode !== "steady") {
    c._mode = "steady";
    c._lastArr = "full";
    c._lastOffset = 0;
    c.line.setAttribute("stroke-dasharray", c.len.toFixed(1));
    c.line.setAttribute("stroke-dashoffset", "0");
  }
  setConnClass(c, klass);
  setTip(c, 0, 0, 0, c._lastTipCls);
}

function setConnIdle(c) {
  if (c._mode !== "idle") {
    c._mode = "idle";
    c._lastArr = "idle";
    c._lastOffset = c.len;
    c.line.setAttribute("stroke-dasharray", c.len.toFixed(1));
    c.line.setAttribute("stroke-dashoffset", c.len.toFixed(1));
  }
  if (c._lastClass !== "tv-conn") {
    c._lastClass = "tv-conn";
    c.line.setAttribute("class", "tv-conn");
  }
  setTip(c, 0, 0, 0, "tv-conn__tip");
}

// ────────────────────────────────────────────────────────────────────
// reveal a matrix over u with stagger
// ────────────────────────────────────────────────────────────────────

function revealMat(mat, values, u, { stagger = 0.12 } = {}) {
  setOpGroup(mat, Math.min(1, u * 2.2));
  const rows = mat.cells.length;
  const cols = mat.cells[0].length;
  const n = rows * cols;
  const maxDelay = stagger;
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const idx = i * cols + j;
      const delay = (idx / (n - 1 || 1)) * maxDelay;
      const local = clamp01((u - delay) / (1 - maxDelay));
      const eu = ease(local);
      const cell = mat.cells[i][j];
      setFwdOp(cell, eu);
      setCellText(cell, values[i][j] * eu, eu);
    }
  }
}
function holdMat(mat, values) {
  setOpGroup(mat, 1);
  for (let i = 0; i < mat.cells.length; i++) {
    for (let j = 0; j < mat.cells[0].length; j++) {
      const cell = mat.cells[i][j];
      setFwdOp(cell, 1);
      setBwdOp(cell, cell._lastBwdOp); // no-op unless set elsewhere
      setCellText(cell, values[i][j], 1);
    }
  }
}
function gradMat(mat, fwdVals, bwdVals, u) {
  // Apply backward tint, keep cells visible. Number interpolates fwd→bwd.
  setOpGroup(mat, 1);
  const rows = mat.cells.length;
  const cols = mat.cells[0].length;
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const idx = i * cols + j;
      const delay = (idx / (rows * cols)) * 0.35;
      const local = clamp01((u - delay) / (1 - delay + 0.01));
      const eu = ease(local);
      const cell = mat.cells[i][j];
      setBwdOp(cell, eu * 0.78);
      setCellText(cell, lerp(fwdVals[i][j], bwdVals[i][j], eu), 1);
    }
  }
}
function clearBwd(mat) {
  for (const row of mat.cells) for (const cell of row) setBwdOp(cell, 0);
}
function hideMat(mat) {
  setOpGroup(mat, 0);
  for (const row of mat.cells) {
    for (const cell of row) {
      setFwdOp(cell, 0);
      setBwdOp(cell, 0);
      cell._lastTxt = "0.00";
      cell.text.textContent = "0.00";
      cell.text._lastOp = 0;
      cell.text.setAttribute("opacity", "0");
    }
  }
}

// ────────────────────────────────────────────────────────────────────
// stage apply
// ────────────────────────────────────────────────────────────────────

const STAGES = [
  { id: "tokens",   at:  0.3, dur: 1.1 },
  { id: "embed",    at:  1.2, dur: 1.9 },
  { id: "split",    at:  2.9, dur: 1.3 },
  { id: "scores",   at:  4.0, dur: 2.2 },
  { id: "softmax",  at:  6.0, dur: 1.5 },
  { id: "heads",    at:  7.3, dur: 2.1 },
  { id: "concat",   at:  9.2, dur: 1.3 },
  { id: "proj",     at: 10.3, dur: 1.2 },
  { id: "dwell",    at: 11.4, dur: 0.8 },
  { id: "loss",     at: 12.1, dur: 0.9 },
  { id: "gradOut",  at: 12.9, dur: 1.2 },
  { id: "gradAV",   at: 14.0, dur: 1.6 },
  { id: "gradQKE",  at: 15.5, dur: 2.2 },
  { id: "cooldown", at: 17.5, dur: 1.5 },
];
const CYCLE = 19.1;

const APPLY = {
  tokens(u, s) {
    const items = s.tokens.items;
    for (let i = 0; i < items.length; i++) {
      const delay = i * 0.14;
      const local = clamp01((u - delay) / (1 - delay + 0.01));
      const uu = ease(local);
      const it = items[i];
      const tx = COL.tokens - (1 - uu) * 14;
      if (Math.abs(tx - it._lastTx) > 0.1) {
        it._lastTx = tx;
        it.g.setAttribute("transform", `translate(${tx.toFixed(1)} ${it.y})`);
      }
      if (Math.abs(uu - it._lastOp) > 0.01) {
        it._lastOp = uu;
        it.g.setAttribute("opacity", uu.toFixed(3));
      }
    }
  },

  embed(u, s) {
    // Shoot each token→E connector quickly, staggered; then hold steady.
    const shootU = clamp01(u / 0.45);
    for (let i = 0; i < s.conns.tokToE.length; i++) {
      const delay = i * 0.1;
      const local = clamp01((shootU - delay) / (1 - delay + 0.01));
      if (local < 1) setConnShoot(s.conns.tokToE[i], local, "tv-conn tv-conn--fwd");
      else setConnSteady(s.conns.tokToE[i], "tv-conn tv-conn--fwd");
    }
    revealMat(s.embed, s.data.E, u, { stagger: 0.35 });
  },

  split(u, s) {
    const shootU = clamp01(u / 0.6);
    for (let h = 0; h < s.heads.length; h++) {
      const delay = h * 0.08;
      const local = clamp01((shootU - delay) / (1 - delay + 0.01));
      if (local < 1) setConnShoot(s.conns.eToHead[h], local, "tv-conn tv-conn--fwd tv-conn--q");
      else setConnSteady(s.conns.eToHead[h], "tv-conn tv-conn--fwd tv-conn--q");
    }
  },

  scores(u, s) {
    // Connectors already steady from split. Just reveal scores in parallel.
    for (let h = 0; h < s.heads.length; h++) {
      setConnSteady(s.conns.eToHead[h], "tv-conn tv-conn--fwd tv-conn--q");
      revealMat(s.heads[h].s, s.data.heads[h].S, u, { stagger: 0.35 });
    }
  },

  softmax(u, s) {
    const shootU = clamp01(u / 0.45);
    for (let h = 0; h < s.heads.length; h++) {
      if (shootU < 1) setConnShoot(s.conns.sToA[h], shootU, "tv-conn tv-conn--fwd tv-conn--a");
      else setConnSteady(s.conns.sToA[h], "tv-conn tv-conn--fwd tv-conn--a");
      revealMat(s.heads[h].a, s.data.heads[h].A, u, { stagger: 0.30 });
    }
  },

  heads(u, s) {
    const shootU = clamp01(u / 0.45);
    for (let h = 0; h < s.heads.length; h++) {
      if (shootU < 1) setConnShoot(s.conns.aToO[h], shootU, "tv-conn tv-conn--fwd tv-conn--v");
      else setConnSteady(s.conns.aToO[h], "tv-conn tv-conn--fwd tv-conn--v");
      revealMat(s.heads[h].o, s.data.heads[h].O, u, { stagger: 0.30 });
    }
  },

  concat(u, s) {
    const shootU = clamp01(u / 0.5);
    for (let h = 0; h < s.heads.length; h++) {
      const delay = h * 0.08;
      const local = clamp01((shootU - delay) / (1 - delay + 0.01));
      if (local < 1) setConnShoot(s.conns.oToConcat[h], local, "tv-conn tv-conn--fwd");
      else setConnSteady(s.conns.oToConcat[h], "tv-conn tv-conn--fwd");
    }
    revealMat(s.concat, s.data.C, u, { stagger: 0.25 });
  },

  proj(u, s) {
    const shootU = clamp01(u / 0.45);
    if (shootU < 1) setConnShoot(s.conns.concatToOut, shootU, "tv-conn tv-conn--fwd");
    else setConnSteady(s.conns.concatToOut, "tv-conn tv-conn--fwd");
    revealMat(s.out, s.data.O, u, { stagger: 0.25 });
  },

  dwell(u, s) {
    // Breathing pulse on the final output — the one part of the scene that
    // stays alive while the caption settles on "forward".
    const breath = 0.92 + 0.08 * Math.sin(u * Math.PI * 2);
    setOpGroup(s.out, breath);
  },

  loss(u, s) {
    const eu = ease(u);
    const scale_ = 0.8 + 0.25 * eu;
    s.lossG.setAttribute("opacity", eu.toFixed(3));
    s.lossG.setAttribute(
      "transform",
      `translate(${COL.out} ${CY - 180}) scale(${scale_.toFixed(3)})`
    );
  },

  gradOut(u, s) {
    // Reverse-shoot concat→out, tint out + concat crimson.
    if (u < 1) setConnShootBack(s.conns.concatToOut, u, "tv-conn tv-conn--bwd");
    else setConnIdle(s.conns.concatToOut);
    gradMat(s.out, s.data.O, s.data.dO, u);
    gradMat(s.concat, s.data.C, s.data.dC, Math.max(0, u - 0.25));
  },

  gradAV(u, s) {
    // Reverse oToConcat and aToO per head; tint O and A matrices.
    for (let h = 0; h < s.heads.length; h++) {
      if (u < 1) {
        setConnShootBack(s.conns.oToConcat[h], u, "tv-conn tv-conn--bwd");
        setConnShootBack(s.conns.aToO[h], u, "tv-conn tv-conn--bwd tv-conn--v");
      } else {
        setConnIdle(s.conns.oToConcat[h]);
        setConnIdle(s.conns.aToO[h]);
      }
      const head = s.data.heads[h];
      gradMat(s.heads[h].o, head.O, head.dO, Math.min(1, u * 1.4));
      gradMat(s.heads[h].a, head.A, head.dA, Math.max(0, u - 0.25));
    }
  },

  gradQKE(u, s) {
    // Reverse sToA and eToHead per head; tint S; fold grads back into E;
    // finally reverse tokToE so tokens feel the gradient too.
    for (let h = 0; h < s.heads.length; h++) {
      if (u < 1) {
        setConnShootBack(s.conns.sToA[h], u, "tv-conn tv-conn--bwd tv-conn--a");
        setConnShootBack(s.conns.eToHead[h], u, "tv-conn tv-conn--bwd tv-conn--q");
      } else {
        setConnIdle(s.conns.sToA[h]);
        setConnIdle(s.conns.eToHead[h]);
      }
      const head = s.data.heads[h];
      gradMat(s.heads[h].s, head.S, head.dS, Math.min(1, u * 1.3));
    }
    gradMat(s.embed, s.data.E, s.data.dE, Math.max(0, u - 0.35));
    // tokToE fires slightly later so gradients appear to arrive at tokens.
    const tokU = clamp01((u - 0.5) / 0.5);
    for (let i = 0; i < s.conns.tokToE.length; i++) {
      const delay = i * 0.06;
      const local = clamp01((tokU - delay) / (1 - delay + 0.01));
      if (local <= 0) setConnSteady(s.conns.tokToE[i], "tv-conn tv-conn--bwd");
      else if (local < 1) setConnShootBack(s.conns.tokToE[i], local, "tv-conn tv-conn--bwd");
      else setConnIdle(s.conns.tokToE[i]);
    }
    // tokens get a crimson halo as the gradient arrives.
    const tokTint = clamp01((u - 0.65) / 0.35);
    for (const it of s.tokens.items) {
      if (Math.abs(tokTint - it._lastBwd) > 0.02) {
        it._lastBwd = tokTint;
        it.bgBwd.setAttribute("opacity", tokTint.toFixed(3));
      }
    }
  },

  cooldown(u, s) {
    // Only a visual fade — state reset is handled by resetScene() on cycle
    // wrap so we can't miss it via frame timing.
    const f = 1 - ease(u);
    const groups = [
      s.tokens.g, s.embed.g, s.concat.g, s.out.g, s.lossG,
      ...s.heads.flatMap((h) => [h.s.g, h.a.g, h.o.g]),
    ];
    for (const g of groups) g.setAttribute("opacity", f.toFixed(3));
    const everyConn = [
      ...s.conns.tokToE,
      ...s.conns.eToHead,
      ...s.conns.sToA,
      ...s.conns.aToO,
      ...s.conns.oToConcat,
      s.conns.concatToOut,
    ];
    for (const c of everyConn) setConnIdle(c);
  },
};

// Hard reset on cycle boundary. Called once per wrap, before any stage
// APPLYs run. Guarantees every tracker agrees with the actual DOM so
// dirty-checks in the next cycle don't no-op on stale values.
function resetScene(s) {
  // tokens container + items
  s.tokens.g.setAttribute("opacity", "1");
  for (const it of s.tokens.items) {
    it._lastOp = 0;
    it._lastTx = COL.tokens;
    it._lastBwd = 0;
    it.g.setAttribute("opacity", "0");
    it.g.setAttribute("transform", `translate(${COL.tokens} ${it.y})`);
    it.bgBwd.setAttribute("opacity", "0");
  }

  // matrices — hideMat resets group _lastOp + every cell's fwd/bwd/text
  const mats = [
    s.embed, s.concat, s.out,
    ...s.heads.flatMap((h) => [h.s, h.a, h.o]),
  ];
  for (const m of mats) hideMat(m);

  // loss badge
  s.lossG.setAttribute("opacity", "0");
  s.lossG.setAttribute(
    "transform",
    `translate(${COL.out} ${CY - 180}) scale(0.8)`
  );

  // connectors
  const allConns = [
    ...s.conns.tokToE,
    ...s.conns.eToHead,
    ...s.conns.sToA,
    ...s.conns.aToO,
    ...s.conns.oToConcat,
    s.conns.concatToOut,
  ];
  for (const c of allConns) setConnIdle(c);
}

// ────────────────────────────────────────────────────────────────────
// caption + scrubber
// ────────────────────────────────────────────────────────────────────

function updateCaption(s, idx) {
  if (!s.caption) return;
  const stages = s.labels && s.labels.stages;
  if (!stages) return;
  const text = stages[idx] || "";
  if (text === s._captionText) return;
  s._captionText = text;
  // crossfade via opacity; swap text on same frame to avoid layout jump
  s.caption.style.opacity = "0";
  requestAnimationFrame(() => {
    s.caption.textContent = text;
    s.caption.style.opacity = "1";
  });
}

function updateScrub(s, idx) {
  const { dots } = s.scrub;
  const bwdStart = 10; // gradOut onwards
  for (let i = 0; i < dots.length; i++) {
    const d = dots[i];
    const cls = [];
    if (i === idx) cls.push("is-current");
    else if (i < idx) cls.push("is-past");
    if (i >= bwdStart) cls.push("is-bwd");
    const want = cls.join(" ");
    if (d._lastCls !== want) {
      d._lastCls = want;
      d.className = "tviz__dot " + want;
    }
  }
}

// ────────────────────────────────────────────────────────────────────
// mount
// ────────────────────────────────────────────────────────────────────

function prefersReducedMotion() {
  try { return window.matchMedia("(prefers-reduced-motion: reduce)").matches; }
  catch { return false; }
}

// ── floating math glyph atmosphere ──
// Spawns ~16 ML/math expressions absolutely-positioned inside the hero,
// each drifting on its own randomized CSS animation. Pure decoration —
// CSS does the work, JS just seeds the parameters.
const GLYPHS = [
  "∇L", "π(a|s)", "Q(s,a)", "∂L/∂θ", "softmax(QKᵀ/√d)",
  "½‖O‖²", "log π_θ", "V*(s)", "A·V", "exp(s_ij)",
  "argmax_a", "Σ_t γᵗ r_t", "KL(p‖q)", "H(π)", "𝔼[R]",
  "GAE(λ)", "clip(ρ, 1±ε)", "softmax", "⟨q, k⟩", "∂A/∂S",
];

function mountGlyphDrift(hero, count = 16) {  if (!hero) return;
  const layer = document.createElement("div");
  layer.className = "tv-glyphs";
  layer.setAttribute("aria-hidden", "true");

  const pool = [...GLYPHS];
  for (let i = 0; i < count; i++) {
    const sym = pool.length
      ? pool.splice((Math.random() * pool.length) | 0, 1)[0]
      : GLYPHS[(Math.random() * GLYPHS.length) | 0];
    const el = document.createElement("span");
    el.className = "tv-glyph";
    el.textContent = sym;

    const top = 4 + Math.random() * 92;       // 4..96 %
    const left = 2 + Math.random() * 96;      // 2..98 %
    const dur = 55 + Math.random() * 55;      // 55..110 s — slow drift
    const delay = -Math.random() * dur;       // begin mid-cycle
    const dx = (Math.random() - 0.5) * 280;
    const dy = (Math.random() - 0.5) * 180;
    const max = 0.045 + Math.random() * 0.075; // 0.045..0.12
    const fontSize = 11 + Math.random() * 4.5;
    const tint = Math.random() < 0.18 ? "tv-glyph--violet"
               : Math.random() < 0.30 ? "tv-glyph--sky"
               : "";
    if (tint) el.classList.add(tint);

    el.style.cssText =
      `top:${top.toFixed(1)}%;` +
      `left:${left.toFixed(1)}%;` +
      `font-size:${fontSize.toFixed(1)}px;` +
      `animation-duration:${dur.toFixed(1)}s;` +
      `animation-delay:${delay.toFixed(1)}s;` +
      `--tv-dx:${dx.toFixed(0)}px;` +
      `--tv-dy:${dy.toFixed(0)}px;` +
      `--tv-max-op:${max.toFixed(3)};`;
    layer.appendChild(el);
  }
  hero.insertBefore(layer, hero.firstChild);
}

// ── training telemetry ──
// A thin mono line at the top of the hero whose values are loosely tied
// to the animation cycle: step counts up, loss decays slowly, ‖∇‖ peaks
// during the backward stages and is ~0 during forward. Pure decoration —
// no real training is happening — but the numbers are deterministic given
// (cycleCount, cycleT) so they feel like a coherent run, not random.

function setupTelemetry(hero) {
  if (!hero) return null;
  const root = hero.querySelector(".tv-telemetry");
  if (!root) return null;
  const slots = {};
  root.querySelectorAll("[data-tele]").forEach((el) => {
    slots[el.dataset.tele] = { el, last: el.textContent };
  });
  return {
    root,
    slots,
    lastUpdate: 0,
    write(key, text) {
      const slot = this.slots[key];
      if (!slot) return;
      if (slot.last === text) return;
      slot.last = text;
      slot.el.textContent = text;
    },
  };
}

function fmt3(x) { return Number(x).toFixed(3); }
function fmt2(x) { return Number(x).toFixed(2); }
function pad(n, w) {
  const s = String(n);
  return s.length >= w ? s : "0".repeat(w - s.length) + s;
}

function tickTelemetry(t, cycleCount, cycleT, now) {
  if (!t) return;
  if (now - t.lastUpdate < 140) return; // ~7 fps is plenty for text
  t.lastUpdate = now;

  // step: ramps within a cycle, plus accumulated cycles.
  const STEPS_PER_CYCLE = 13;
  const stepWithin = Math.floor((cycleT / CYCLE) * STEPS_PER_CYCLE);
  const step = cycleCount * STEPS_PER_CYCLE + stepWithin + 1;

  // loss: smooth decay across cycles + per-cycle wobble that follows the
  // forward/backward shape (settles low during forward dwell, jumps up on
  // gradient stages, then settles again).
  const baseDecay = Math.exp(-cycleCount * 0.04 - cycleT / CYCLE * 0.04);
  const wobble = 0.04 * Math.sin(cycleT * 0.9 + cycleCount);
  const loss = 0.06 + 1.18 * baseDecay + wobble;

  // ‖∇‖: zero in forward (cycleT < 12.1), grows during backward, decays in cooldown.
  let grad = 0;
  if (cycleT > 12.0 && cycleT < 17.6) {
    const gu = (cycleT - 12.0) / 5.6;
    grad = 0.85 * Math.sin(gu * Math.PI) + 0.05 * Math.sin(cycleT * 5.3);
    if (grad < 0) grad = -grad;
  } else if (cycleT >= 17.6) {
    grad = 0.04 * Math.max(0, 1 - (cycleT - 17.6) / 1.5);
  }

  // lr: piecewise constant — drops once every 8 epochs, just to feel real.
  const lrSteps = [3.0e-4, 1.5e-4, 7.5e-5, 3.8e-5];
  const lrIdx = Math.min(lrSteps.length - 1, Math.floor(cycleCount / 8));
  const lr = lrSteps[lrIdx];

  t.write("step",  `step ${pad(step, 4)}`);
  t.write("loss",  `loss ${fmt3(loss)}`);
  t.write("grad",  `‖∇‖ ${fmt2(grad)}`);
  t.write("lr",    `lr ${lr.toExponential(1)}`);
  t.write("cycle", `epoch ${cycleCount + 1}`);
}

function runForwardStill(s) {
  // Show a static end-of-forward frame, skip captions animation.
  APPLY.tokens(1, s);
  APPLY.embed(1, s);
  APPLY.split(1, s);
  APPLY.scores(1, s);
  APPLY.softmax(1, s);
  APPLY.heads(1, s);
  APPLY.concat(1, s);
  APPLY.proj(1, s);
  const stages = (s.labels && s.labels.stages) || [];
  if (s.caption) s.caption.textContent = (s.labels && s.labels.staticPreview) || stages[7] || "";
  for (const d of s.scrub.dots) d.classList.add("is-past");
}

export function mountTransformerViz(root) {
  if (!root) return () => {};
  let labels = {};
  try { labels = JSON.parse(root.dataset.captions || "{}"); } catch {}
  const data = precompute();
  const scene = buildScene(root, data, labels);

  // floating glyph backdrop (skipped under reduced motion — CSS handles fade)
  const hero = root.closest(".hero--viz");
  if (hero && !prefersReducedMotion()) mountGlyphDrift(hero);

  // training telemetry strip — bound to the cycle counter so the numbers
  // feel like they're reacting to the visual underneath.
  const telemetry = setupTelemetry(hero);

  if (prefersReducedMotion()) {
    runForwardStill(scene);
    return () => {};
  }

  let running = true;
  let t0 = performance.now();
  let lastCaption = -2;
  let lastCycleT = 0;
  let cycleCount = 0;
  let rafId = 0;

  // Start from a known-clean state so cycle 1 matches every later cycle.
  resetScene(scene);

  function step(now) {
    if (!running) return;
    const cycleT = ((now - t0) / 1000) % CYCLE;
    // Cycle wrapped — re-sync every tracker with the DOM so the next pass
    // starts from the same clean state as the first.
    if (cycleT < lastCycleT) {
      resetScene(scene);
      cycleCount += 1;
    }
    lastCycleT = cycleT;
    // Find the current stage index, then apply all stages up to and including it.
    let current = 0;
    for (let i = 0; i < STAGES.length; i++) {
      const st = STAGES[i];
      const u = clamp01((cycleT - st.at) / st.dur);
      if (u > 0) APPLY[st.id](u, scene);
      if (cycleT >= st.at && cycleT < st.at + st.dur) current = i;
    }
    if (current !== lastCaption) {
      updateCaption(scene, current);
      updateScrub(scene, current);
      lastCaption = current;
    }
    tickTelemetry(telemetry, cycleCount, cycleT, now);
    rafId = requestAnimationFrame(step);
  }
  rafId = requestAnimationFrame(step);

  // Pause only when the tab is hidden (saves battery, no visible regression).
  // Hover-pause was removed: it made the animation freeze whenever the cursor
  // entered the hero, which on long visits felt like a stutter rather than a
  // feature. The animation now plays continuously while the tab is visible.
  const onVis = () => {
    if (document.hidden) {
      running = false;
      cancelAnimationFrame(rafId);
    } else if (!running) {
      running = true;
      // resume at the same stage we paused in
      const resumeAt = lastCaption >= 0 ? STAGES[lastCaption].at : 0;
      t0 = performance.now() - resumeAt * 1000;
      lastCycleT = 0;
      rafId = requestAnimationFrame(step);
    }
  };
  document.addEventListener("visibilitychange", onVis);

  return () => {
    running = false;
    cancelAnimationFrame(rafId);
    document.removeEventListener("visibilitychange", onVis);
  };
}
