// riskslim report: the ROC card's plot, drawn exactly as Python built it in model_report.py.
// The markup is the in-DOM template in the "roc-plot" slot of template.html.
window.RocPlot = {
  props: ["data"],
  mounted() {
    drawFigure(this.$refs.plot, this.data.figures.roc);
  },
};
