// riskslim report: one Plotly figure, drawn exactly as Python built it in model_report.py.
// ``figure`` names a key of data.figures ("roc" or "calibration").
window.PlotlyChart = {
  props: ["data", "figure"],
  mounted() {
    const figure = structuredClone(this.data.figures[this.figure]);
    Plotly.newPlot(this.$refs.plot, figure.data, figure.layout,
      { displayModeBar: false, responsive: true, scrollZoom: false });
  },
  template: `<div ref="plot" class="rs-plot" :data-figure="figure"></div>`,
};
