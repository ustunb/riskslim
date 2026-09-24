// riskslim report: draw the figures and make the Model card add up. The markup is all
// Jinja's; Python computes every number, so the card only sums points x value over its inputs
// and looks the total up in model.cell_by_total, which names the strip cell whose risk it shows --
// the browser derives no model fact.
(function () {
  "use strict";
  const data = JSON.parse(document.getElementById("report-data").textContent);
  // Python builds the whole figure, styling included; it is drawn once, so Plotly may write
  // into it directly
  document.querySelectorAll("#report [data-figure]").forEach((element) => {
    const figure = data.figures[element.dataset.figure];
    Plotly.newPlot(element, figure.data, figure.layout,
      { displayModeBar: false, responsive: true, scrollZoom: false });
  });

  // the page may leave the Model card out
  if (!document.getElementById("rs-total")) return;
  const inputs = document.querySelectorAll(".rs-model-table input");
  const cells = document.querySelectorAll(".rs-score-grid [data-cell]");
  function update() {
    let total = 0;
    inputs.forEach((input) => {
      // a typed number can leave the item's range; min and max only bound the spinner
      const value = input.type === "checkbox" ? Number(input.checked)
        : Math.min(Math.max(Number(input.value), input.min), input.max);
      total += Number(input.dataset.points) * value;
    });
    // missing only for a total no value set reaches, e.g. a non-integer value typed in
    const current = data.model.cell_by_total[String(total)];
    document.getElementById("rs-total").textContent = total;
    document.getElementById("rs-risk").textContent =
      current === undefined ? "—" : data.model.score_to_risk[current].risk;
    cells.forEach((cell, i) => cell.classList.toggle("rs-current", current === i));
  }
  inputs.forEach((input) => input.addEventListener("input", update));
  update();
})();
