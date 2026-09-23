// riskslim report: the model card's body -- a points table (risk score) or a list of items
// (checklist), the checklist rule, and the score-to-risk row. Python computes every number.
// The markup is the in-DOM template in the "model-card" slot of template.html.
window.ModelCard = {
  props: ["data"],
  setup(props) {
    const model = props.data.model;
    const isChecklist = model.type === "checklist";
    const pointsLabel = (item) => {
      if (isChecklist) return item.points > 0 ? "+" : "−";
      if (!item.binary) return `${item.points} × value`;
      return `${item.points} ${Math.abs(item.points) === 1 ? "point" : "points"}`;
    };
    const itemName = (item) => item.binary ? item.name
      : `${item.name} (${item.value_range[0]}–${item.value_range[1]})`;
    // the strip is one wrapping grid at every width -- styles.css decides how many cells fit
    return { model, isChecklist, pointsLabel, itemName };
  },
};
