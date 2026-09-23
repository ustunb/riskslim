// riskslim report: the summary card's body -- one flat table, a header row naming the samples
// and then every row of data.summary, in the order Python built them.
// The markup is the in-DOM template in the "summary-table" slot of template.html.
window.SummaryTable = {
  props: ["data"],
  // Python writes every string and every colspan; this component only renders them
  setup: (props) => ({ summary: props.data.summary }),
};
