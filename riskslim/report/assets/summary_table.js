// riskslim report: the summary card's body -- every block of data.summary, in the order Python
// built them, each with its own header row.
// The markup is the in-DOM template in the "summary-table" slot of template.html.
window.SummaryTable = {
  props: ["data"],
  setup(props) {
    const blocks = props.data.summary;
    // blocks with fewer columns stretch their last cell to the table width
    const width = Math.max(...blocks.map((block) => block.columns.length));
    const span = (block, j) => (j === block.columns.length - 2 ? width - block.columns.length + 1 : 1);
    return { blocks, span };
  },
};
