// riskslim report: the calibration card's plot, drawn exactly as Python built it in
// model_report.py. The markup is the in-DOM template in the "calibration-plot" slot of
// template.html.
window.CalibrationPlot = {
  props: ["data"],
  mounted() {
    drawFigure(this.$refs.plot, this.data.figures.calibration);
  },
};
