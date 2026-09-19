// riskslim report: renders the layout rows in the JSON data block. Python computes everything;
// this file only maps component names to Vue components and draws Python's Plotly figures.
(function () {
  "use strict";
  const DATA = Vue.markRaw(JSON.parse(document.getElementById("report-data").textContent));
  const pct = (v) => (100 * v).toFixed(1) + "%";

  const Card = {
    props: ["title"],
    template: `
      <article class="rs-card">
        <header><h3>{{ title }}</h3></header>
        <slot></slot>
      </article>`,
  };

  const ModelCard = {
    components: { Card },
    props: ["data", "options"],
    setup(props) {
      const model = props.data.model;
      const isChecklist = model.type === "checklist";
      const hasNegative = model.items.some((item) => item.points < 0);
      const pointsLabel = (item) => {
        if (isChecklist) return item.points > 0 ? "+" : "−";
        if (!item.binary) return `${item.points} × value`;
        return `${item.points} ${Math.abs(item.points) === 1 ? "point" : "points"}`;
      };
      const itemName = (item) => item.binary ? item.name
        : `${item.name} (${item.value_range[0]}–${item.value_range[1]})`;
      return { model, isChecklist, hasNegative, pointsLabel, itemName, pct };
    },
    template: `
      <Card :title="options.title">
        <table class="rs-model-table" :class="isChecklist ? 'rs-checklist' : 'rs-risk-score'">
          <tbody>
            <tr v-for="(item, i) in model.items" class="rs-item-row">
              <td v-if="isChecklist" class="rs-box" aria-hidden="true">☐</td>
              <td v-else class="rs-index">{{ i + 1 }}.</td>
              <td class="rs-name">{{ itemName(item) }}</td>
              <td v-if="!isChecklist || hasNegative" class="rs-points">{{ pointsLabel(item) }}</td>
              <td v-if="!isChecklist" class="rs-tally">{{ i === 0 ? "" : "+" }} …</td>
            </tr>
            <tr v-if="model.items.length === 0"><td colspan="4">No items: every row gets the same score.</td></tr>
          </tbody>
          <tfoot v-if="!isChecklist && model.items.length">
            <tr>
              <td></td>
              <td>ADD POINTS FROM ROWS 1–{{ model.items.length }}</td>
              <td class="rs-points">SCORE</td>
              <td class="rs-tally">= …</td>
            </tr>
          </tfoot>
        </table>
        <p v-if="isChecklist" class="rs-rule">{{ model.rule }}</p>
        <div class="rs-score-risk">
          <table>
            <tr>
              <th>{{ isChecklist ? "NET CHECKED" : "SCORE" }}</th>
              <td v-for="s in model.score_to_risk.scores"
                  :class="{ 'rs-positive': isChecklist && s >= model.checklist_m }">{{ s }}</td>
            </tr>
            <tr>
              <th>RISK</th>
              <td v-for="(r, j) in model.score_to_risk.risk"
                  :class="{ 'rs-positive': isChecklist && model.score_to_risk.scores[j] >= model.checklist_m }">{{ pct(r) }}</td>
            </tr>
          </table>
        </div>
      </Card>`,
  };

  const SummaryTable = {
    components: { Card },
    props: ["data", "options"],
    setup(props) {
      const blocks = props.options.blocks
        .map((key) => props.data.summary.find((block) => block.key === key))
        .filter(Boolean);
      // blocks with fewer columns stretch their last cell to the table width
      const width = Math.max(...blocks.map((block) => block.columns.length));
      const span = (block, j) => (j === block.columns.length - 2 ? width - block.columns.length + 1 : 1);
      return { blocks, span };
    },
    template: `
      <Card :title="options.title">
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
        </table>
      </Card>`,
  };

  // One Plotly component per figure key in data.figures (built in figures.py).
  const PlotlyChart = (figureKey) => ({
    components: { Card },
    props: ["data", "options"],
    mounted() {
      const figure = structuredClone(this.data.figures[figureKey]);
      Plotly.newPlot(this.$refs.plot, figure.data, figure.layout,
        { displayModeBar: false, responsive: true, scrollZoom: false });
    },
    template: `
      <Card :title="options.title">
        <div ref="plot" class="rs-plot" :data-figure="'${figureKey}'"></div>
      </Card>`,
  });

  // Component registry: a layout.py dataclass name -> a Vue component.
  const REGISTRY = {
    ModelCard,
    SummaryTable,
    RocPlot: PlotlyChart("roc"),
    CalibrationPlot: PlotlyChart("calibration"),
  };

  Vue.createApp({
    setup() {
      return { data: DATA, registry: REGISTRY };
    },
    template: `
      <header class="rs-head"><h2>{{ data.title }}</h2></header>
      <section v-for="row in data.layout" class="rs-row">
        <component v-for="spec in row.components" :is="registry[spec.component]"
                   :data="data" :options="spec.options"></component>
      </section>`,
  }).mount("#report");
})();
