// riskslim report: the summary card's body -- one flat table, a header row naming the samples
// and then every row of data.summary, in the order Python built them.
// The markup is the in-DOM template in the "summary-table" slot of template.html.
window.SummaryTable = {
  props: ["data"],
  setup(props) {
    const summary = props.data.summary;
    // a row carrying one value where there are several samples is not per-sample: stretch it
    const samples = summary.columns.length - 1;
    const span = (row) => (row.values.length === 1 ? samples : 1);
    return { summary, span };
  },
};
