// riskslim report: make the Model card add up and draw the figures. The markup is all
// Jinja's; Python computes every number, so the card only sums points x value over its inputs
// and looks the total up in model.cell_by_total, which names the strip cell whose risk it shows --
// the browser derives no model fact.
(function () {
  "use strict";
  const data = JSON.parse(document.getElementById("report-data").textContent);

  // the key of a total in model.cell_by_total: Python's total_key, step for step (the same IEEE
  // multiply, add and floor, then integer arithmetic), so a sum in another order finds its cell
  function totalKey(total) {
    const nano = Math.floor(total * 1e9 + 0.5);
    const fraction = Math.abs(nano) % 1e9;
    const whole = (Math.abs(nano) - fraction) / 1e9;
    return (nano < 0 ? "-" : "") + whole +
      (fraction ? ("." + String(fraction).padStart(9, "0")).replace(/0+$/, "") : "");
  }

  // the Model card first: it needs no library, so it works even when Plotly fails to load
  // (the page may leave the card out)
  if (document.getElementById("rs-total")) {
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
      const key = totalKey(total);
      const current = data.model.cell_by_total[key];
      document.getElementById("rs-total").textContent =
        current === undefined ? key : data.model.total_labels[key];
      document.getElementById("rs-risk").textContent =
        current === undefined ? "—" : data.model.score_to_risk[current].risk;
      cells.forEach((cell, i) => cell.classList.toggle("rs-current", current === i));
    }
    inputs.forEach((input) => input.addEventListener("input", update));
    update();
  }

  // Python builds the whole figure, styling included; it is drawn once, so Plotly may write
  // into it directly. Without Plotly (offline, blocked) the figure boxes stay empty.
  if (typeof Plotly === "undefined") return;
  const plots = [...document.querySelectorAll("#report [data-figure]")];
  plots.forEach((element) => {
    const figure = data.figures[element.dataset.figure];
    Plotly.newPlot(element, figure.data, figure.layout,
      { displayModeBar: false, responsive: true, scrollZoom: false });
  });

  // the legend shows and hides a sample on the whole page, not in one plot: a click toggles it, a
  // double click shows it alone (or, when it already is, every sample again), as Plotly does in
  // one plot. A trace's name is its legend entry, which starts with the sample's name.
  const sampleOf = (trace) => trace.name.split("<br>")[0];
  function showOnly(shown) {
    plots.forEach((plot) => Plotly.restyle(plot, {
      visible: plot.data.map((trace) => shown.has(sampleOf(trace)) || "legendonly"),
    }));
  }
  plots.forEach((plot) => {
    const shownAndClicked = (event) => [
      new Set(event.data.filter((trace) => trace.visible !== "legendonly").map(sampleOf)),
      sampleOf(event.data[event.curveNumber]),
    ];
    plot.on("plotly_legendclick", (event) => {
      const [shown, clicked] = shownAndClicked(event);
      shown.has(clicked) ? shown.delete(clicked) : shown.add(clicked);
      showOnly(shown);
      return false;
    });
    plot.on("plotly_legenddoubleclick", (event) => {
      const [shown, clicked] = shownAndClicked(event);
      const alone = shown.size === 1 && shown.has(clicked);
      showOnly(alone ? new Set(event.data.map(sampleOf)) : new Set([clicked]));
      return false;
    });
  });
})();
