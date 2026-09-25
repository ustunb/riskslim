// riskslim report: make the Model card add up and draw the figures. The markup is all
// Jinja's; Python computes every number, so the card only sums points x value over its inputs
// and finds the strip cell whose risk that score shows: a discrete score is looked up in
// model.cell_by_score, a continuous one is compared with the bin edges in model.score_bins and
// printed with its score_digits decimals -- the browser derives no model fact.
(function () {
  "use strict";
  const data = JSON.parse(document.getElementById("report-data").textContent);

  // the Model card first: it needs no library, so it works even when Plotly fails to load
  // (the page may leave the card out)
  if (document.getElementById("rs-score")) {
    const { cell_by_score: cellByScore, score_bins: scoreBins, score_to_risk: strip } = data.model;
    const inputs = document.querySelectorAll(".rs-model-table input");
    const cells = document.querySelectorAll(".rs-score-grid [data-cell]");
    function update() {
      let score = 0;
      inputs.forEach((input) => {
        // a typed number can leave the item's range; min and max only bound the spinner
        const value = input.type === "checkbox" ? Number(input.checked)
          : Math.min(Math.max(Number(input.value), input.min), input.max);
        score += Number(input.dataset.points) * value;
      });
      let cell, label;
      if (scoreBins) {
        // a continuous score is in the bin after every edge at or below it; a bin that holds no
        // training row has no cell (and no risk), but the score still prints
        cell = scoreBins.cells[scoreBins.edges.filter((edge) => score >= edge).length];
        label = score.toFixed(scoreBins.score_digits);
      } else {
        // a discrete score is an integer, keyed as Python writes it ("3"); missing only for a
        // score no value set reaches, e.g. a non-integer value typed in
        const key = String(score);
        cell = cellByScore[key]?.cell;
        label = cellByScore[key]?.label ?? key;
      }
      document.getElementById("rs-score").textContent = label;
      document.getElementById("rs-risk").textContent = cell == null ? "—" : strip[cell].risk;
      cells.forEach((element, i) => element.classList.toggle("rs-current", cell === i));
    }
    inputs.forEach((input) => input.addEventListener("input", update));
    update();
  }

  // the strip is one row; one too long for the card's width wraps instead of running off it
  const strip = document.querySelector(".rs-score-grid");
  let stripWidth;
  function fitStrip() {
    if (!strip || strip.clientWidth === stripWidth) return;
    stripWidth = strip.clientWidth;
    strip.classList.remove("rs-wrapped");
    strip.classList.toggle("rs-wrapped", strip.scrollWidth > strip.clientWidth);
  }
  fitStrip();

  // Python builds the whole figure, styling included; it is drawn once, so Plotly may write
  // into it directly. Without Plotly (offline, blocked) the figure boxes stay empty.
  const plots = typeof Plotly === "undefined" ? []
    : [...document.querySelectorAll("#report [data-figure]")];
  // one handler sizes everything on the page, once per frame
  let resizing;
  window.addEventListener("resize", () => {
    cancelAnimationFrame(resizing);
    resizing = requestAnimationFrame(() => {
      fitStrip();
      plots.forEach(fitToWidth);
    });
  });

  // Python's figure is the desktop layout, with the legend inside the panel; data.narrow holds
  // what changes on a plot narrower than its max_width (the legend under the plot, tighter margins
  // and type, smaller circles) and how to undo it. The figure then grows by the legend's height,
  // so the panel stays square.
  const LEGEND_GAP_PX = 8;  // between the x-axis title and the legend under it
  const LABEL_GAP_PX = 1;  // two score labels closer than this overlap
  plots.forEach((plot) => {
    const figure = data.figures[plot.dataset.figure];
    // showTips: false keeps Plotly's "double-click to isolate" toast off the page
    Plotly.newPlot(plot, figure.data, figure.layout,
      { displayModeBar: false, scrollZoom: false, showTips: false });
    // newPlot drew the desktop layout at the box's width: a desktop plot needs nothing more
    plot.narrow = false;
    plot.fittedWidth = plot.clientWidth;
    plot.on("plotly_afterplot", () => hideCoveredLabels(plot));
    plot.on("plotly_restyle", ([update]) => "visible" in update && mirrorVisibility(plot));
    fitToWidth(plot);
  });

  async function fitToWidth(plot) {
    const width = plot.clientWidth;
    const narrow = width < data.narrow.max_width;
    if (width === plot.fittedWidth && narrow === plot.narrow) return;
    plot.fittedWidth = width;
    const overrides = data.narrow.figures[plot.dataset.figure];
    // autosize: true in a relayout fits the figure to its box as it redraws
    if (narrow !== plot.narrow) {
      plot.narrow = narrow;
      plot.style.height = "";
      const { traces, layout } = narrow ? overrides.apply : overrides.undo;
      await Plotly.update(plot, traces, { ...layout, autosize: true });
    } else {
      await Plotly.relayout(plot, { autosize: true });
    }
    if (!narrow) return;
    // the legend is now under the plot, laid out at this width; a single sample has none
    const legend = plot.querySelector(".legend");
    const legendHeight = legend ? legend.getBoundingClientRect().height + LEGEND_GAP_PX : 0;
    const { "margin.t": top, "margin.r": right, "margin.b": bottom, "margin.l": left } =
      overrides.apply.layout;
    plot.style.height = `${Math.ceil(width - left - right + top + bottom + legendHeight)}px`;
    await Plotly.relayout(plot, { "margin.b": bottom + legendHeight, autosize: true });
  }

  // Plotly draws every score label of a sample above every circle of that sample. A label over
  // a neighbouring circle still reads (the circles of a sample share a colour), but where circles
  // nearly coincide (the high-risk corner) two labels print on top of one another. The label
  // underneath is hidden instead, as its circle is: the hover still names it. Every label is
  // measured before any is hidden, so the page lays out once.
  function hideCoveredLabels(plot) {
    const traces = [...plot.querySelectorAll(".scatterlayer .trace")].map((trace) =>
      [...trace.querySelectorAll(".text .textpoint")]
        .map((label) => ({ label, box: label.getBoundingClientRect() })));
    traces.forEach((labels) => labels.forEach(({ label, box }, i) => {
      const covered = labels.slice(i + 1).some(({ box: later }) =>
        box.left < later.right + LABEL_GAP_PX && later.left < box.right + LABEL_GAP_PX &&
        box.top < later.bottom + LABEL_GAP_PX && later.top < box.bottom + LABEL_GAP_PX);
      label.style.visibility = covered ? "hidden" : "";
    }));
  }

  // the legend shows and hides a sample on the whole page, not in one plot: Plotly handles a
  // click (toggle) or double click (show it alone, or every sample again) in the plot clicked,
  // and the other plots copy its visibility, sample by sample (a trace's meta names its sample)
  let mirroring = false;
  async function mirrorVisibility(source) {
    if (mirroring) return;  // the restyles below fire plotly_restyle too
    mirroring = true;
    const visible = new Map(source.data.map((trace) => [trace.meta, trace.visible ?? true]));
    try {
      await Promise.all(plots.filter((plot) => plot !== source).map((plot) =>
        Plotly.restyle(plot, { visible: plot.data.map((trace) => visible.get(trace.meta) ?? true) })));
    } finally {
      mirroring = false;
    }
  }
})();
