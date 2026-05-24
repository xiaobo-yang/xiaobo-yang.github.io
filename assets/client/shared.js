// Pretext-driven layout helpers shared by every page.
//
// `balanceText` is the heart of the look: instead of letting browsers wrap
// long titles or taglines into ragged lines, we use @chenglou/pretext to
// binary-search the *narrowest* width that still fits in the requested
// line-count. The result feels like a hand-set headline.
//
// We do this in JS at runtime because it depends on the rendered font.
// Re-run on resize.

import {
  prepareWithSegments,
  measureLineStats,
  measureNaturalWidth,
} from "@chenglou/pretext";

export function fontStringFor(el) {
  const cs = getComputedStyle(el);
  // canvas font shorthand: [style] [weight] size family
  return `${cs.fontStyle} ${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
}

const preparedCache = new WeakMap();

function preparedFor(el, text, font) {
  let entry = preparedCache.get(el);
  if (entry && entry.text === text && entry.font === font) return entry.prepared;
  const prepared = prepareWithSegments(text, font);
  preparedCache.set(el, { text, font, prepared });
  return prepared;
}

export function balanceText(el, targetLines = 2) {
  const text = (el.textContent || "").replace(/\s+/g, " ").trim();
  if (!text) return;
  const parent = el.parentElement || el;
  const parentWidth =
    parent.getBoundingClientRect().width || el.getBoundingClientRect().width;
  if (parentWidth < 80) return;

  const font = fontStringFor(el);
  const prepared = preparedFor(el, text, font);

  // upper bound: parent width
  const naturalWidth = Math.min(measureNaturalWidth(prepared), parentWidth);
  const naturalLines = measureLineStats(prepared, parentWidth).lineCount;

  // Decide our actual target. If the parent already fits the text in fewer
  // lines than requested, target = naturalLines (no point growing). Never
  // squeeze BELOW naturalLines or we'd force overflow.
  const target = Math.max(targetLines, naturalLines);

  let lo = Math.max(60, naturalWidth * 0.35);
  let hi = parentWidth;
  // 0.5px precision is plenty
  while (hi - lo > 0.5) {
    const mid = (lo + hi) / 2;
    const count = measureLineStats(prepared, mid).lineCount;
    if (count <= target) hi = mid;
    else lo = mid;
  }

  const chosen = Math.ceil(hi);
  el.style.maxWidth = `${chosen}px`;
  el.style.marginInline = "auto";
  el.dataset.balanced = "1";
}

export function balanceAll(root = document) {
  const els = root.querySelectorAll('[data-pretext="balance"]');
  for (const el of els) {
    const target = Number(el.dataset.targetLines || 2) || 2;
    balanceText(el, target);
  }
}

export function watchResize(root = document) {
  let timer = 0;
  window.addEventListener(
    "resize",
    () => {
      cancelAnimationFrame(timer);
      timer = requestAnimationFrame(() => balanceAll(root));
    },
    { passive: true }
  );
}

export async function whenFontsReady() {
  if (document.fonts && document.fonts.ready) {
    try {
      await document.fonts.ready;
    } catch {}
  }
}
