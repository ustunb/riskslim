// riskslim report: mount each component into its slot in template.html, where the slot's markup
// is the component's in-DOM template. Python computes everything; this file only hands the JSON
// data block to the components.

// Both plot components draw their figure the same way; only the figure differs.
// Python names the page's CSS custom properties -- "var(--rs-ink)", "var(--rs-plot-height)" --
// and they are resolved against :root here, so styles.css stays the one place a value is written.
function resolveStyleTokens(value, styles) {
  if (typeof value === "string") {
    const resolved = value.replace(/var\((--[\w-]+)\)/g,
      (_, token) => styles.getPropertyValue(token).trim());
    return /^-?\d*\.?\d+px$/.test(resolved) ? parseFloat(resolved) : resolved;
  }
  if (Array.isArray(value)) return value.map((item) => resolveStyleTokens(item, styles));
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [key, resolveStyleTokens(item, styles)]));
  }
  return value;
}

function drawFigure(element, figure) {
  const styles = getComputedStyle(document.documentElement);
  const spec = resolveStyleTokens(structuredClone(figure), styles);
  Plotly.newPlot(element, spec.data, spec.layout,
    { displayModeBar: false, responsive: true, scrollZoom: false });
}

(function () {
  "use strict";
  const data = Vue.markRaw(JSON.parse(document.getElementById("report-data").textContent));
  const COMPONENTS = {
    "model-card": window.ModelCard,
    "summary-table": window.SummaryTable,
    "roc-plot": window.RocPlot,
    "calibration-plot": window.CalibrationPlot,
  };
  document.querySelectorAll("#report [data-component]").forEach((slot) => {
    const name = slot.dataset.component;
    const component = COMPONENTS[name];
    if (!component) throw new Error(`report template asks for unknown component "${name}"`);
    Vue.createApp(component, { data }).mount(slot);
  });
})();
