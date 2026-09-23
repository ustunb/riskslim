// riskslim report: mount each component into its slot in template.html. Python computes
// everything; this file only hands the JSON data block to the components.
(function () {
  "use strict";
  const data = Vue.markRaw(JSON.parse(document.getElementById("report-data").textContent));
  const COMPONENTS = {
    "model-card": window.ModelCard,
    "summary-table": window.SummaryTable,
    "plotly-chart": window.PlotlyChart,
  };
  document.querySelectorAll("#report [data-component]").forEach((slot) => {
    const name = slot.dataset.component;
    const component = COMPONENTS[name];
    if (!component) throw new Error(`report template asks for unknown component "${name}"`);
    const props = { data };
    if (slot.dataset.figure) props.figure = slot.dataset.figure;
    Vue.createApp(component, props).mount(slot);
  });
})();
