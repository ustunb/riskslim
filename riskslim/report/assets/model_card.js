// riskslim report: the model card's body -- a points table (risk score) or a list of items
// (checklist), the checklist rule, and the score-to-risk row. Python computes every number.
// The markup is the in-DOM template in the "model-card" slot of template.html.
window.ModelCard = {
  props: ["data"],
  // Python writes every label, every header and the checkbox glyph
  setup: (props) => ({ model: props.data.model }),
};
