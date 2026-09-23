// riskslim report: the summary card's body -- every block of data.summary, in the order Python
// built them, each with its own header row.
window.SummaryTable = {
  props: ["data"],
  setup(props) {
    const blocks = props.data.summary;
    // blocks with fewer columns stretch their last cell to the table width
    const width = Math.max(...blocks.map((block) => block.columns.length));
    const span = (block, j) => (j === block.columns.length - 2 ? width - block.columns.length + 1 : 1);
    return { blocks, span };
  },
  template: `
    <table class="rs-summary-table">
      <tbody v-for="block in blocks" :class="'rs-block-' + block.key">
        <tr class="rs-block-head">
          <th>{{ block.title }}</th>
          <th v-for="(column, j) in block.columns.slice(1)" class="rs-num"
              :colspan="span(block, j)">{{ column }}</th>
        </tr>
        <tr v-for="row in block.rows">
          <td>{{ row[0] }}</td>
          <td v-for="(value, j) in row.slice(1)" class="rs-num"
              :colspan="span(block, j)">{{ value }}</td>
        </tr>
      </tbody>
    </table>`,
};
