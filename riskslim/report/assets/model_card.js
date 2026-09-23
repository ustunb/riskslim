// riskslim report: the model card's body -- a points table (risk score) or a list of items
// (checklist), the checklist rule, and the score-to-risk row. Python computes every number.
window.ModelCard = {
  props: ["data"],
  setup(props) {
    const model = props.data.model;
    const isChecklist = model.type === "checklist";
    const hasNegative = model.items.some((item) => item.points < 0);
    const pct = (v) => (100 * v).toFixed(1) + "%";
    const pointsLabel = (item) => {
      if (isChecklist) return item.points > 0 ? "+" : "−";
      if (!item.binary) return `${item.points} × value`;
      return `${item.points} ${Math.abs(item.points) === 1 ? "point" : "points"}`;
    };
    const itemName = (item) => item.binary ? item.name
      : `${item.name} (${item.value_range[0]}–${item.value_range[1]})`;
    const positive = (s) => isChecklist && s >= model.checklist_m;
    // a single row only fits a dozen scores; wider ranges wrap into a grid
    const wrapScores = model.score_to_risk.scores.length > 12;
    return { model, isChecklist, hasNegative, pointsLabel, itemName, pct, positive, wrapScores };
  },
  template: `
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
    <div v-if="wrapScores" class="rs-score-grid">
      <div v-for="(r, j) in model.score_to_risk.risk"
           class="rs-score-cell"
           :class="{ 'rs-positive': positive(model.score_to_risk.scores[j]) }">
        <span class="rs-score-label">{{ isChecklist ? "≥" : "" }}{{ model.score_to_risk.scores[j] }}</span>
        <span class="rs-score-value">{{ pct(r) }}</span>
      </div>
    </div>
    <div v-else class="rs-score-risk">
      <table>
        <tr>
          <th>{{ isChecklist ? "NET CHECKED" : "SCORE" }}</th>
          <td v-for="s in model.score_to_risk.scores"
              :class="{ 'rs-positive': positive(s) }">{{ s }}</td>
        </tr>
        <tr>
          <th>RISK</th>
          <td v-for="(r, j) in model.score_to_risk.risk"
              :class="{ 'rs-positive': positive(model.score_to_risk.scores[j]) }">{{ pct(r) }}</td>
        </tr>
      </table>
    </div>`,
};
