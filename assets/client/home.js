// Home page bootstrap.
//
// Runs after fonts settle, then balances every [data-pretext="balance"]
// element on the page. Re-balances on resize so the home title stays tight
// at every breakpoint. Also mounts the animated transformer hero if present.

import { balanceAll, balanceText, watchResize, whenFontsReady } from "./shared.js";
import { mountTransformerViz } from "./transformer-viz.js";

async function init() {
  await whenFontsReady();
  const tviz = document.querySelector(".tviz");
  if (tviz) mountTransformerViz(tviz);
  balanceAll();
  watchResize();
  mountTaglineRotator();
  mountTocReveal();
  // tiny stagger reveal — everything is already legible without JS,
  // we just give it a softer entrance.
  document.body.classList.add("ready");
}

// Rotate the hero tagline through a small set of variants. Each swap fades
// out, re-runs pretext's shrinkwrap, fades back in — so the line literally
// rebreathes its line count every cycle. Skipped under reduced-motion.
function mountTaglineRotator() {
  const el = document.querySelector(".hero__tagline[data-taglines]");
  if (!el) return;
  let variants;
  try {
    variants = JSON.parse(el.dataset.taglines);
  } catch {
    return;
  }
  if (!Array.isArray(variants) || variants.length < 2) return;
  const reduceMotion =
    window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduceMotion) return;

  el.classList.add("hero__tagline--rotating");
  const target = Number(el.dataset.targetLines || 2) || 2;
  let i = 0;
  let timer = 0;

  const swap = () => {
    if (document.hidden) return;
    i = (i + 1) % variants.length;
    el.classList.add("is-fading");
    // Wait the fade-out, swap text, re-balance under the new content, fade in.
    setTimeout(() => {
      el.textContent = variants[i];
      // pretext caches by element identity + text; new text invalidates it.
      el.removeAttribute("data-balanced");
      el.style.maxWidth = "";
      balanceText(el, target);
      // Force layout settle then fade back in.
      requestAnimationFrame(() => {
        el.classList.remove("is-fading");
      });
    }, 480);
  };

  const start = () => {
    stop();
    timer = window.setInterval(swap, 6200);
  };
  const stop = () => {
    if (timer) {
      clearInterval(timer);
      timer = 0;
    }
  };

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stop();
    else start();
  });
  start();
}

// Stagger the TOC items into view as the reader scrolls past them. Items
// already on-screen at boot animate in immediately with a small per-item
// delay; ones below the fold wait for IntersectionObserver. Items above the
// fold (scrolled past on reload) skip the animation so we don't pop them in
// behind the reader.
function mountTocReveal() {
  const list = document.querySelector(".toc__list");
  if (!list) return;
  const items = Array.from(list.querySelectorAll(".toc__item"));
  if (!items.length) return;

  const reduceMotion =
    window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduceMotion || !("IntersectionObserver" in window)) {
    for (const it of items) it.classList.add("is-in");
    return;
  }

  list.classList.add("is-armed");

  const viewportH = window.innerHeight || 800;
  let firstVisible = -1;
  items.forEach((it, idx) => {
    const r = it.getBoundingClientRect();
    if (r.top < viewportH * 0.95 && r.bottom > 0 && firstVisible === -1) {
      firstVisible = idx;
    }
    // Items already scrolled past — render them at rest, no animation.
    if (r.bottom <= 0) it.classList.add("is-in");
  });

  // Reveal initially-visible items with a small stagger.
  if (firstVisible >= 0) {
    items.forEach((it, idx) => {
      if (it.classList.contains("is-in")) return;
      const r = it.getBoundingClientRect();
      if (r.top < viewportH * 0.95) {
        const delay = (idx - firstVisible) * 90;
        setTimeout(() => it.classList.add("is-in"), Math.max(0, delay));
      }
    });
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
    { rootMargin: "0px 0px -8% 0px", threshold: 0.05 }
  );
  for (const it of items) {
    if (!it.classList.contains("is-in")) io.observe(it);
  }
}

init();
