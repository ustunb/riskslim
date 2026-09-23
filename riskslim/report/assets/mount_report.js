// riskslim report: mount each component into its slot in template.html, where the slot's markup
// is the component's in-DOM template. Python computes everything; this file only hands the JSON
// data block to the components.

// Both plot components draw their figure the same way; only the figure differs.
function drawFigure(element, figure) {
  const spec = structuredClone(figure);
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
