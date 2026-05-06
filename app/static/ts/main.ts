// ── Form loading state ──────────────────────────────────────────────────────
function initForm(): void {
  const form    = document.getElementById("sentiment-form") as HTMLFormElement | null;
  const btn     = document.getElementById("submit-btn")     as HTMLButtonElement | null;
  const spinner = document.getElementById("spinner")        as HTMLElement | null;
  const label   = document.getElementById("btn-label")      as HTMLElement | null;

  form?.addEventListener("submit", () => {
    if (btn)     btn.disabled = true;
    if (spinner) spinner.classList.remove("hidden");
    if (label)   label.textContent = "Analysing…";
  });
}

// ── Animate confidence bars from 0 → target ────────────────────────────────
function animateBars(): void {
  const bars = document.querySelectorAll<HTMLElement>("[data-target]");
  // Start from 0 then animate on next frame so CSS transition fires
  requestAnimationFrame(() => {
    setTimeout(() => {
      bars.forEach(bar => {
        const pct = parseFloat(bar.dataset.target ?? "0") * 100;
        bar.style.width = `${pct}%`;
      });
    }, 80);
  });
}

// ── Token tooltip ───────────────────────────────────────────────────────────
function initTokenTooltips(): void {
  const tokens = document.querySelectorAll<HTMLElement>("[data-attention]");
  const tooltip = document.getElementById("token-tooltip") as HTMLElement | null;

  if (!tooltip) return;

  tokens.forEach(token => {
    token.addEventListener("mouseenter", (e: MouseEvent) => {
      const raw   = parseFloat(token.dataset.attention ?? "0");
      const pct   = (raw * 100).toFixed(1);
      tooltip.textContent = `Attention: ${pct}%`;
      tooltip.classList.remove("opacity-0");
      positionTooltip(e, tooltip);
    });
    token.addEventListener("mousemove", (e: MouseEvent) => positionTooltip(e, tooltip));
    token.addEventListener("mouseleave", () => tooltip.classList.add("opacity-0"));
  });
}

function positionTooltip(e: MouseEvent, tooltip: HTMLElement): void {
  tooltip.style.left = `${e.pageX + 12}px`;
  tooltip.style.top  = `${e.pageY - 32}px`;
}

// ── Boot ────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initForm();
  animateBars();
  initTokenTooltips();
});
