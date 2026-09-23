// riskslim report: the model card's body -- a points table (risk score) or a list of items
// (checklist), the checklist rule, and the score-to-risk row. Python computes every number.
// The markup is the in-DOM template in the "model-card" slot of template.html.
window.ModelCard = {
  props: ["data"],
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
    // a single row only fits a dozen cells; wider strips wrap into a grid
    const wrapScores = model.score_to_risk.length > 12;
    return { model, isChecklist, hasNegative, pointsLabel, itemName, wrapScores };
  },
};
