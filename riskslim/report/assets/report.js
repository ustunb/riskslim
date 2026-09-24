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

  // the strip is one row; one too long for the card's width wraps instead of running off it
  const strip = document.querySelector(".rs-score-grid");
  function fitStrip() {
    strip.classList.remove("rs-wrapped");
    strip.classList.toggle("rs-wrapped", strip.scrollWidth > strip.clientWidth);
  }
  if (strip) {
    fitStrip();
    window.addEventListener("resize", fitStrip);
  }

  // Python builds the whole figure, styling included; it is drawn once, so Plotly may write
  // into it directly. Without Plotly (offline, blocked) the figure boxes stay empty.
  if (typeof Plotly === "undefined") return;

  // Python's figure is the desktop layout, with the legend inside the panel. A plot narrower than
  // NARROW_PX (a phone, or a two-column page on a small screen) has no room for it there: the
  // legend moves under the plot, the margins and axis type tighten so the panel keeps its size,
  // and the calibration circles shrink by NARROW_CIRCLES. The figure grows by the legend's height,
  // so the panel stays square. Wider again, every change is undone.
  const NARROW_PX = 480;
  const NARROW_CIRCLES = 20 / 24;
  const NARROW_LAYOUT = {
    "legend.orientation": "h", "legend.xref": "paper", "legend.x": 0, "legend.xanchor": "left",
    "legend.yref": "container", "legend.y": 0, "legend.yanchor": "bottom",
    "margin.t": 24, "margin.r": 16, "margin.b": 56, "margin.l": 56,
    "xaxis.tickfont.size": 12, "yaxis.tickfont.size": 12,
    "xaxis.title.font.size": 13, "yaxis.title.font.size": 13,
  };
  const LEGEND_GAP_PX = 8;  // between the x-axis title and the legend under it
  const LABEL_GAP_PX = 1;  // two score labels closer than this overlap
  const plots = [...document.querySelectorAll("#report [data-figure]")];
  plots.forEach((element) => {
    const figure = data.figures[element.dataset.figure];
    // the circle sizes Python set, to scale from and to restore (Plotly writes into the figure)
    element.circleSizes = figure.data.map((trace) => [trace.marker.size, trace.textfont?.size]);
    Plotly.newPlot(element, figure.data, figure.layout,
      { displayModeBar: false, responsive: true, scrollZoom: false });
    element.on("plotly_afterplot", () => hideCoveredLabels(element));
    fitToWidth(element);
  });
  let resizing;
  window.addEventListener("resize", () => {
    cancelAnimationFrame(resizing);
    resizing = requestAnimationFrame(() => plots.forEach(fitToWidth));
  });

  async function fitToWidth(plot) {
    const narrow = plot.clientWidth < NARROW_PX;
    if (narrow !== plot.narrow) {
      plot.narrow = narrow;
      // null takes a key back to the value the figure and its template give it
      await Plotly.relayout(plot, Object.fromEntries(Object.entries(NARROW_LAYOUT)
        .map(([key, value]) => [key, narrow ? value : null])));
      const scale = narrow ? NARROW_CIRCLES : 1;
      if (plot.dataset.figure === "calibration") {
        await Plotly.restyle(plot, {
          "marker.size": plot.circleSizes.map(([size]) => size * scale),
          "textfont.size": plot.circleSizes.map(([, text]) => text * scale),
        });
      }
    }
    if (!narrow) {
      if (plot.style.height) {
        plot.style.height = "";
        await Plotly.Plots.resize(plot);
      }
      return;
    }
    // a single sample has no legend
    const legend = plot.querySelector(".legend");
    const legendHeight = legend ? legend.getBoundingClientRect().height + LEGEND_GAP_PX : 0;
    const margin = Object.fromEntries(["t", "r", "b", "l"].map((side) => [side, NARROW_LAYOUT[`margin.${side}`]]));
    const panel = plot.clientWidth - margin.l - margin.r;
    plot.style.height = `${Math.ceil(panel + margin.t + margin.b + legendHeight)}px`;
    await Plotly.relayout(plot, { "margin.b": margin.b + legendHeight });
    await Plotly.Plots.resize(plot);
  }

  // Plotly draws every score label of a sample above every circle of that sample. A label over
  // a neighbouring circle still reads (the circles of a sample share a colour), but where circles
  // nearly coincide (the high-risk corner) two labels print on top of one another. The label
  // underneath is hidden instead, as its circle is: the hover still names it.
  function hideCoveredLabels(plot) {
    plot.querySelectorAll(".scatterlayer .trace").forEach((trace) => {
      const labels = [...trace.querySelectorAll(".text .textpoint")];
      const boxes = labels.map((label) => label.getBoundingClientRect());
      labels.forEach((label, i) => {
        const box = boxes[i];
        const covered = boxes.slice(i + 1).some((later) =>
          box.left < later.right + LABEL_GAP_PX && later.left < box.right + LABEL_GAP_PX &&
          box.top < later.bottom + LABEL_GAP_PX && later.top < box.bottom + LABEL_GAP_PX);
        label.style.visibility = covered ? "hidden" : "";
      });
    });
  }

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
