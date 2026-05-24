// Post page bootstrap.
//
// 1. Balance the title and summary via pretext.
// 2. Drive the top reading-progress bar from scroll position.
// 3. Highlight in-post TOC entries as their section enters the viewport.
// 4. Fade body landmarks (h2/pre/figure/display-math) in as the reader
//    scrolls past them.

import { balanceAll, watchResize, whenFontsReady } from "./shared.js";

function readingProgress() {
  const bar = document.querySelector(".reading-progress > span");
  if (!bar) return;
  const update = () => {
    const doc = document.documentElement;
    const scrolled = window.scrollY || doc.scrollTop;
    const max = (doc.scrollHeight - window.innerHeight) || 1;
    const pct = Math.min(1, Math.max(0, scrolled / max));
    bar.style.transform = `scaleX(${pct})`;
  };
  update();
  window.addEventListener("scroll", update, { passive: true });
  window.addEventListener("resize", update, { passive: true });
}

function tocSpy() {
  const tocLinks = Array.from(
    document.querySelectorAll(".post__toc a[href^='#']")
  );
  if (!tocLinks.length) return;
  const map = new Map();
  for (const a of tocLinks) {
    const id = decodeURIComponent(a.getAttribute("href").slice(1));
    const target = document.getElementById(id);
    if (target) map.set(target, a);
  }
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        const a = map.get(e.target);
        if (!a) continue;
        if (e.isIntersecting) {
          tocLinks.forEach((l) => l.classList.remove("is-current"));
          a.classList.add("is-current");
        }
      }
    },
    { rootMargin: "0px 0px -70% 0px", threshold: 0.1 }
  );
  for (const target of map.keys()) io.observe(target);
}

// Long code blocks get folded after ~10 lines with a Medium-style fade and
// an Expand pill. Threshold is generous enough that small snippets (yaml
// frontmatter, three-line bash) stay fully visible.
function collapseLongCodeBlocks() {
  const blocks = Array.from(document.querySelectorAll(".code-block"));
  if (!blocks.length) return;
  const COLLAPSE_AT_LINES = 14;
  const SHOW_LINES = 10;
  const isZh = (document.documentElement.lang || "en").toLowerCase().startsWith("zh");
  const linesWord = isZh ? "行" : "lines";
  const collapseLabel = isZh ? "收起 ↑" : "Collapse ↑";

  for (const block of blocks) {
    const pre = block.querySelector("pre");
    if (!pre) continue;
    const code = pre.querySelector("code") || pre;
    const text = code.textContent || "";
    const totalLines = text.replace(/\n$/, "").split("\n").length;
    if (totalLines <= COLLAPSE_AT_LINES) continue;

    const cs = getComputedStyle(pre);
    const lineH = parseFloat(cs.lineHeight) || 22;
    const padTop = parseFloat(cs.paddingTop) || 0;
    const cap = Math.round(lineH * SHOW_LINES + padTop);
    block.style.setProperty("--cb-collapsed-max", `${cap}px`);

    block.classList.add("is-collapsible", "is-collapsed");

    const hidden = totalLines - SHOW_LINES;
    const expandLabel = isZh
      ? `展开 ↓  +${hidden} ${linesWord}`
      : `Expand ↓  +${hidden} ${linesWord}`;

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "code-block__toggle";
    btn.textContent = expandLabel;
    btn.setAttribute("aria-expanded", "false");
    btn.addEventListener("click", () => {
      const nowCollapsed = block.classList.toggle("is-collapsed");
      btn.setAttribute("aria-expanded", String(!nowCollapsed));
      btn.textContent = nowCollapsed ? expandLabel : collapseLabel;
    });
    block.appendChild(btn);
  }
}

// Quietly fade body landmarks in as the reader moves through the post.
// Above-the-fold content reveals immediately so we don't pop content in
// behind the reader; below-the-fold waits for IntersectionObserver.
function bodyReveal() {
  const body = document.querySelector(".post__body");
  if (!body) return;
  const reduceMotion =
    window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduceMotion || !("IntersectionObserver" in window)) return;

  const targets = Array.from(
    body.querySelectorAll(
      ":scope > h2, :scope > h3, :scope > pre, :scope > .code-block, :scope > figure, :scope > .katex-display, :scope > blockquote"
    )
  );
  if (!targets.length) return;

  body.classList.add("is-armed");

  const viewportH = window.innerHeight || 800;
  for (const el of targets) {
    const r = el.getBoundingClientRect();
    if (r.top < viewportH * 0.92) el.classList.add("is-in");
  }

  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          e.target.classList.add("is-in");
          io.unobserve(e.target);
        }
      }
    },
    { rootMargin: "0px 0px -8% 0px", threshold: 0.02 }
  );
  for (const el of targets) {
    if (!el.classList.contains("is-in")) io.observe(el);
  }
}

async function init() {
  await whenFontsReady();
  balanceAll();
  watchResize();
  readingProgress();
  tocSpy();
  collapseLongCodeBlocks();
  bodyReveal();
  document.body.classList.add("ready");
}

init();
