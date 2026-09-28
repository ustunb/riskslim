// riskslim report: run the Model card and draw the figures. The markup is all Jinja's; Python
// computes every number and display string, so the card only sums its rows' contributions, finds
// the strip cell of that score by comparing it with the score edges in model.score_bins and
// prints it with their score_digits decimals -- the browser derives no model fact.
(function () {
  "use strict";

  // ---- helpers ----------------------------------------------------------------------------

  // a number as the card prints it: a true minus sign, digits decimals when given
  function formatNumber(value, digits) {
    const text = digits == null ? String(value) : value.toFixed(digits);
    return text.replace(/^-/, "−");
  }

  // a renamed label with its condition kept on one line, as Python keeps its own labels
  // (model.condition_operators): it may break before the operator, never between it and its value
  function conditionOnOneLine(label, operators) {
    return label.replace(new RegExp(` (${operators.join("|")}) `, "g"), " $1 ");
  }

  // "{name}" placeholders of a message (the template's data-*-message) filled in
  function fillMessage(message, values) {
    return message.replace(/\{(\w+)\}/g, (_, name) => values[name]);
  }

  // ---- value inputs --------------------------------------------------------------------------
  // Interchangeable ways to enter a continuous feature's value, behind one small protocol. Each
  // hydrates the template's markup for it:
  //   .value      a number or null (unset); setting it repaints and fires nothing
  //   .disabled   true while the card is being edited: the value cannot change
  //   "input"     fired whenever the reader changes the value
  // A row talks to nothing else, so one kind can replace another. The model picks one kind for
  // all its continuous features (model.value_input, Python's): VALUE_INPUTS below.

  class ValueInput extends EventTarget {
    constructor() {
      super();
      this.current = null;
      this.isDisabled = false;
    }

    get value() { return this.current; }

    // an unchanged value (or disabled state) repaints nothing
    set value(value) {
      if ((value ?? null) === this.current) return;
      this.current = value ?? null;
      this.paint();
    }

    get disabled() { return this.isDisabled; }

    set disabled(disabled) {
      if (disabled === this.isDisabled) return;
      this.isDisabled = disabled;
      this.paintDisabled();
    }

    // the reader's change: update, then tell the row
    choose(value) {
      if (this.isDisabled || value === this.current) return;
      this.value = value;
      this.dispatchEvent(new Event("input"));
    }
  }

  // A value box (the value, or "–" unset) with a pip meter beside it: numbered pips, one per
  // level, filled up to the value, running right to left from the box (pip 1 nearest it). The
  // meter's space is reserved while closed (styles.css), so opening it moves and covers nothing.
  // Opens: a mouse resting on the box ~150ms; a click or tap; Enter or Space. A mouse-opened
  // meter closes once the pointer has left box and meter for ~250ms. A click on the box toggles
  // it (the first click on a hover-opened meter pins it instead). A choice, Escape, focus or a
  // click leaving it close it. Keyboard, on the box or the meter: arrows step (← increases, as
  // the meter runs), Home/End jump, a digit picks that level (0 = 10), Backspace/Delete clear.
  class PipMeterValue extends ValueInput {
    static HOVER_OPEN_MS = 150;
    static HOVER_CLOSE_MS = 250;
    static CLOSE_AFTER_CHOICE_MS = 150;

    // root: the template's .rs-value-compact; levels: the feature's, ascending
    constructor(root, levels, signal) {
      super();
      this.root = root;
      this.levels = levels;
      this.box = root.querySelector(".rs-value-box");
      this.tray = root.querySelector(".rs-value-tray");
      this.meter = root.querySelector(".rs-value-meter");
      this.pips = [...this.meter.querySelectorAll(".rs-value-pip")];
      this.preview = null;  // the level under the pointer, shown in the box before it is chosen
      this.openedBy = null;
      this.timer = null;
      const on = (target, type, listener, options) =>
        target.addEventListener(type, listener, { ...options, signal });
      const { HOVER_OPEN_MS, HOVER_CLOSE_MS } = PipMeterValue;

      this.pips.forEach((pip, i) => {
        // event.detail is 0 for a keyboard click: choose; a pointer click on the value clears it
        on(pip, "click", (event) => {
          this.choose(event.detail && this.current === levels[i] ? null : levels[i]);
          this.closeSoon();
        });
        on(pip, "mouseenter", () => this.showPreview(i));
      });
      on(this.meter, "mouseleave", () => this.showPreview(null));
      on(this.meter, "keydown", (event) => {
        const next = this.keyed(event);
        if (!next) return;
        event.preventDefault();
        this.choose(next.value);
        this.focusPip();
        if (next.commit) this.closeSoon();
      });
      on(this.tray, "keydown", (event) => {
        if (event.key !== "Escape") return;
        event.preventDefault();
        this.close();
      });
      on(root, "focusout", (event) => { if (!root.contains(event.relatedTarget)) this.close(); });
      on(this.box, "keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          this.open("key");
          return;
        }
        const next = this.keyed(event);
        if (!next) return;
        event.preventDefault();
        this.choose(next.value);
      });
      on(root, "pointerenter", (event) => {
        if (event.pointerType !== "mouse") return;
        if (this.isOpen) clearTimeout(this.timer);
        else this.later(() => this.open("hover"), HOVER_OPEN_MS);
      });
      on(root, "pointerleave", (event) => {
        if (event.pointerType !== "mouse") return;
        if (!this.isOpen) clearTimeout(this.timer);
        else if (this.openedBy === "hover") this.later(() => this.close(), HOVER_CLOSE_MS);
      });
      on(this.box, "click", () => {
        clearTimeout(this.timer);
        if (!this.isOpen) this.open("click");
        else if (this.openedBy === "hover") { this.openedBy = "click"; this.focusPip(); }  // pin it
        else this.close();
      });
      // while open, a pointer pressed anywhere else closes it (added in open, removed in close)
      this.signal = signal;
      this.closeOnOutsidePointer = (event) => { if (!root.contains(event.target)) this.close(); };
      signal.addEventListener("abort", () => clearTimeout(this.timer));
      this.paint();
    }

    get isOpen() { return this.root.classList.contains("rs-value-open"); }

    later(action, delay) {
      clearTimeout(this.timer);
      this.timer = setTimeout(action, delay);
    }

    open(openedBy) {
      if (this.isDisabled) return;
      this.openedBy = openedBy;
      this.root.classList.add("rs-value-open");
      document.addEventListener("pointerdown", this.closeOnOutsidePointer, { capture: true, signal: this.signal });
      if (openedBy !== "hover") this.focusPip();
    }

    close() {
      clearTimeout(this.timer);
      if (!this.isOpen) return;
      document.removeEventListener("pointerdown", this.closeOnOutsidePointer, { capture: true });
      const hadFocus = this.tray.contains(document.activeElement);
      this.root.classList.remove("rs-value-open");
      this.openedBy = null;
      this.showPreview(null);
      if (hadFocus) this.box.focus();
    }

    // after a choice: let it show first
    closeSoon() { this.later(() => this.close(), PipMeterValue.CLOSE_AFTER_CHOICE_MS); }

    focusPip() { this.pips.find((pip) => pip.tabIndex === 0)?.focus(); }

    // index: the pip under the pointer, or null when the pointer leaves the meter
    showPreview(index) {
      this.preview = index == null ? null : this.levels[index];
      this.meter.classList.toggle("rs-value-previewing", index != null);
      this.pips.forEach((pip, i) => pip.classList.toggle("rs-value-preview", index != null && i <= index));
      this.paint();
    }

    // the level a key picks, as {value, commit}, or undefined for a key that is not ours
    keyed(event) {
      if (this.isDisabled) return undefined;
      const { levels } = this;
      const i = levels.indexOf(this.current);
      const step = { ArrowLeft: 1, ArrowUp: 1, ArrowRight: -1, ArrowDown: -1 }[event.key];
      if (step) return { value: i < 0 ? levels[0] : levels[Math.min(levels.length - 1, Math.max(0, i + step))], commit: false };
      if (event.key === "Home") return { value: levels[0], commit: false };
      if (event.key === "End") return { value: levels[levels.length - 1], commit: false };
      if (event.key === "Backspace" || event.key === "Delete") return { value: null, commit: true };
      if (/^[0-9]$/.test(event.key)) {
        const digit = event.key === "0" && levels.includes(10) ? 10 : Number(event.key);
        if (levels.includes(digit)) return { value: digit, commit: true };
      }
      return undefined;
    }

    paint() {
      const at = this.levels.indexOf(this.current);
      this.pips.forEach((pip, i) => {
        pip.setAttribute("aria-checked", String(i === at));
        pip.tabIndex = (at < 0 ? i === 0 : i === at) ? 0 : -1;
        pip.classList.toggle("rs-value-on", i <= at);
        pip.classList.toggle("rs-value-current", i === at);
      });
      const shown = this.preview ?? this.current;
      const { box } = this;
      box.textContent = shown == null ? box.dataset.unsetText : formatNumber(shown);
      box.classList.toggle("rs-value-unset", shown == null);
      box.classList.toggle("rs-value-previewing", this.preview != null && this.preview !== this.current);
      if (this.current == null) {
        box.removeAttribute("aria-valuenow");
        box.setAttribute("aria-valuetext", box.dataset.unsetValuetext);
      } else {
        box.setAttribute("aria-valuenow", this.current);
        box.removeAttribute("aria-valuetext");
      }
    }

    paintDisabled() {
      if (this.isDisabled) this.close();
      this.box.setAttribute("aria-disabled", String(this.isDisabled));
      this.meter.setAttribute("aria-disabled", String(this.isDisabled));
      this.pips.forEach((pip) => { pip.disabled = this.isDisabled; });
    }
  }

  // A number box: any number, typed. Blank (or not a number) is unset.
  class NumberBoxValue extends ValueInput {
    constructor(input, signal) {
      super();
      this.input = input;
      input.addEventListener("input", () => {
        const text = input.value.trim();
        this.choose(text === "" || !Number.isFinite(Number(text)) ? null : Number(text));
      }, { signal });
    }

    // leaves the text alone while it already reads the value ("2." while typing "2.5")
    paint() {
      const { input } = this;
      const reads = input.value.trim() === "" ? null : Number(input.value);
      if (reads !== this.current) input.value = this.current == null ? "" : String(this.current);
    }

    paintDisabled() { this.input.disabled = this.isDisabled; }
  }

  // model.value_input -> the value input a continuous row hydrates
  const VALUE_INPUTS = {
    pip_meter: (node, feature, signal) =>
      new PipMeterValue(node.querySelector(".rs-value-compact"), feature.levels, signal),
    number: (node, feature, signal) => new NumberBoxValue(node.querySelector(".rs-value-number"), signal),
  };

  // ---- the Model card's rows -----------------------------------------------------------------

  // One feature's row: its label, drag handle and rename. A row kind adds what the row counts
  // and how the reader changes it; ROW_KINDS below maps a feature's kind to its row. Rows only
  // read the card's state and ask the card to act (card.dispatch); they never change state.
  class FeatureRow {
    constructor(card, node, feature) {
      this.card = card;
      this.node = node;
      this.feature = feature;
      this.label = node.querySelector(".rs-feature-label");
      this.shownLabel = feature.label;  // Jinja printed it
      this.handle = node.querySelector(".rs-handle");
      const { signal } = card;
      // while editing: drag the handle, or focus it and press the arrow keys, to move the row;
      // a click on it is part of a drag, never a toggle
      this.handle.addEventListener("click", (event) => event.stopPropagation(), { signal });
      this.handle.addEventListener("pointerdown", (event) => card.startDrag(this, event), { signal });
      this.handle.addEventListener("keydown", (event) => {
        const step = { ArrowUp: -1, ArrowDown: 1 }[event.key];
        if (!step) return;
        event.preventDefault();
        card.moveBy(this, step);
        this.handle.focus();
      }, { signal });
      // while editing, a double-click on the label renames the feature
      this.label.addEventListener("dblclick", () => { if (card.state.editing) this.rename(); }, { signal });
    }

    // the reorder group: a row moves only among rows of its own kind
    get group() { return this.feature.kind; }

    // the label shown: the reader's rename, else Python's. The handle, the row's toggle and its
    // value input are named by this label (aria-labelledby), so they follow a rename
    showLabel(label) {
      if (this.shownLabel === label) return;
      this.shownLabel = label;
      this.label.textContent = label;
    }

    // rename in place: Enter or leaving the label saves, Escape cancels; an empty name cancels
    rename() {
      const { label, node } = this;
      const before = this.shownLabel;
      node.classList.add("rs-renaming");
      label.contentEditable = "plaintext-only";
      label.focus();
      getSelection().selectAllChildren(label);
      const renaming = new AbortController();
      const finish = (save) => {
        renaming.abort();
        label.removeAttribute("contenteditable");
        node.classList.remove("rs-renaming");
        const text = conditionOnOneLine(label.textContent.replace(/\s+/g, " ").trim(),
          this.card.model.condition_operators);
        this.shownLabel = null;
        if (save && text && text !== before) this.card.dispatch({ type: "rename", id: this.feature.id, label: text });
        else this.showLabel(before);
      };
      label.addEventListener("keydown", (event) => {
        event.stopPropagation();  // the row's Space/Enter toggle and the card's Escape stay out
        if (event.key === "Enter") { event.preventDefault(); label.blur(); }
        else if (event.key === "Escape") { event.preventDefault(); finish(false); }
      }, { signal: renaming.signal });
      label.addEventListener("blur", () => finish(true), { signal: renaming.signal });
      this.card.signal.addEventListener("abort", () => renaming.abort(), { signal: renaming.signal });
    }

    render(state, label) {
      this.showLabel(label);
      this.node.classList.toggle("rs-active", this.counts(state));
      this.handle.tabIndex = state.editing ? 0 : -1;
    }
  }

  // A binary feature: present or absent. The whole row toggles it (a click; Space or Enter on
  // its toggle); it contributes its weight while checked.
  class BinaryRow extends FeatureRow {
    constructor(card, node, feature) {
      super(card, node, feature);
      this.toggle = node.querySelector(".rs-toggle");
      // beside continuous features, the row's value column reads 0 or 1
      this.valueSlot = node.querySelector("[data-binary-value]");
      const toggle = () => card.dispatch({ type: "toggle", id: feature.id });
      node.addEventListener("click", toggle, { signal: card.signal });
      this.toggle.addEventListener("keydown", (event) => {
        if (event.target !== this.toggle || (event.key !== " " && event.key !== "Enter")) return;
        event.preventDefault();
        toggle();
      }, { signal: card.signal });
    }

    // its input as Python ships it: checked or not
    initialInput() { return this.feature.checked; }

    counts(state) { return state.inputs[this.feature.id]; }

    contribution(state) { return this.counts(state) ? this.feature.weight : 0; }

    render(state, label) {
      super.render(state, label);
      const checked = this.counts(state);
      this.toggle.setAttribute("aria-checked", String(checked));
      if (this.valueSlot) this.valueSlot.textContent = checked ? "1" : "0";
    }
  }

  // A continuous feature: its value entered in its value input; it contributes weight x value,
  // and nothing (null) while the value is unset.
  class ContinuousRow extends FeatureRow {
    constructor(card, node, feature) {
      super(card, node, feature);
      this.input = VALUE_INPUTS[card.model.value_input](node, feature, card.signal);
      this.input.addEventListener("input", () =>
        card.dispatch({ type: "setValue", id: feature.id, value: this.input.value }));
    }

    // its input as Python ships it: the value, or null (unset)
    initialInput() { return this.feature.value; }

    counts(state) { return state.inputs[this.feature.id] != null; }

    contribution(state) {
      const value = state.inputs[this.feature.id];
      return value == null ? null : this.feature.weight * value;
    }

    render(state, label) {
      super.render(state, label);
      this.input.value = state.inputs[this.feature.id];
      this.input.disabled = state.editing;
    }
  }

  // a feature's kind -> its row. A new kind (a combined indicator, say) adds a row class with
  // initialInput(), counts(), contribution() and render(), and its markup in template.html
  const ROW_KINDS = { binary: BinaryRow, continuous: ContinuousRow };

  // ---- the Model card ------------------------------------------------------------------------

  // Where the reader's lasting changes (renames, the rows' order) would persist: load() gives
  // saved overrides or null, save(overrides) keeps them. Nothing persists yet; a localStorage
  // or JSON-file adapter implements the same two methods.
  const NO_STORAGE = { load: () => null, save: () => {} };

  // The Model card: hydrates one [data-model-card] root the template rendered in full. Three
  // kinds of data, kept apart:
  //   defaults   the model's feature elements (Python's), never changed
  //   state      this view: each row's input (checked, or the value entered), whether the reader
  //              has touched the card, editing, collapsed
  //   overrides  the reader's lasting changes (renamed labels, the rows' order), through storage
  // Every change goes through dispatch(action); render() then writes text, classes and ARIA.
  class ModelCard {
    constructor(root, model, storage = NO_STORAGE) {
      this.root = root;
      this.model = model;
      this.storage = storage;
      this.listening = new AbortController();
      this.signal = this.listening.signal;
      this.defaults = new Map(model.features.map((feature) => [feature.id, feature]));
      this.overrides = { labels: {}, order: null, ...storage.load() };

      this.list = root.querySelector(".rs-features");
      this.rows = [...this.list.querySelectorAll(".rs-feature")].map((node) => {
        const feature = this.defaults.get(node.dataset.featureId);
        return new ROW_KINDS[feature.kind](this, node, feature);
      });
      this.state = { touched: false, editing: false, collapsed: false, inputs: this.initialInputs() };
      this.totalOutput = root.querySelector(".rs-total-value");
      this.strip = root.querySelector(".rs-strip");
      this.stripCells = [...this.strip.querySelectorAll(".rs-strip-cell")];
      this.announcer = root.querySelector("[data-announcer]");
      this.editButton = root.querySelector(".rs-edit");
      this.collapseButton = root.querySelector(".rs-collapse");
      this.bindCard();
      if (this.overrides.order) this.showOrder(this.overrides.order);
      this.render();
    }

    // {id: input} of every row as Python ships it: every binary row unchecked, every value unset
    initialInputs() {
      return Object.fromEntries(this.rows.map((row) => [row.feature.id, row.initialInput()]));
    }

    bindCard() {
      const { root, signal } = this;
      this.editButton.addEventListener("click",
        () => this.dispatch({ type: "setEditing", editing: !this.state.editing }), { signal });
      this.collapseButton.addEventListener("click",
        () => this.dispatch({ type: "setCollapsed", collapsed: !this.state.collapsed }), { signal });
      // the last pointer's type: a tap (touch) shows the definition of the row it focuses, a
      // mouse click does not (hovering the name does)
      root.addEventListener("pointerdown", (event) => { root.dataset.pointer = event.pointerType; },
        { capture: true, signal });
      // Escape hides an open definition: it blurs the focused row or name
      document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && this.list.contains(document.activeElement)) document.activeElement.blur();
      }, { signal });
      // the table wraps at the card's width: mark each line's ends whenever its size changes,
      // and once the labels' web font has arrived
      this.stripObserver = new ResizeObserver(() => this.markStripLines());
      this.stripObserver.observe(this.strip);
      document.fonts?.ready.then(() => { if (!signal.aborted) this.markStripLines(); });
    }

    destroy() {
      this.listening.abort();
      this.dragging?.abort();
      this.stripObserver.disconnect();
      modelCards.delete(this.root);
    }

    dispatch(action) {
      const { state, overrides } = this;
      switch (action.type) {
        case "toggle":  // editing freezes the score
          if (state.editing) return;
          state.inputs[action.id] = !state.inputs[action.id];
          state.touched = true;
          break;
        case "setValue":
          if (state.editing) break;  // render puts the value back
          state.inputs[action.id] = action.value;
          state.touched = true;
          break;
        case "rename":
          overrides.labels[action.id] = action.label;
          this.storage.save(overrides);
          break;
        case "move":
          overrides.order = action.order;
          this.storage.save(overrides);
          break;
        case "setEditing":
          state.editing = action.editing;
          break;
        case "setCollapsed":
          state.collapsed = action.collapsed;
          break;
        case "reset":  // back to the untouched card
          Object.assign(state, { touched: false, inputs: this.initialInputs() });
          break;
        default:
          throw new Error(`unknown Model card action: ${action.type}`);
      }
      const { score, cell } = this.render();
      if (action.type === "toggle" || action.type === "setValue") this.announceScore(score, cell);
      if (action.announce) this.announce(action.announce);
    }

    // The Total, or null while a continuous value is unset (no score to show yet)
    score() {
      let total = 0;
      for (const row of this.rows) {
        const part = row.contribution(this.state);
        if (part == null) return null;
        total += part;
      }
      return total;
    }

    // the strip cell of a score: the bin after every edge at or below it; null when that bin
    // holds no cell (no training row reached it)
    cellOf(score) {
      const { edges, cells } = this.model.score_bins;
      return cells[edges.filter((edge) => score >= edge).length] ?? null;
    }

    // writes the card; returns the score shown and its strip cell (null for none)
    render() {
      const { state, overrides, root } = this;
      this.rows.forEach((row) => row.render(state, overrides.labels[row.feature.id] ?? row.feature.label));
      // Untouched (before the first click or value): no Total, no cell. Then the Total, or "—"
      // while a value is unset, and the current cell
      const score = state.touched ? this.score() : null;
      this.totalOutput.textContent = !state.touched ? ""
        : score == null ? this.totalOutput.dataset.missingText
        : formatNumber(score, this.model.score_bins.score_digits);
      const cell = score == null ? null : this.cellOf(score);
      this.stripCells.forEach((node) => {
        const current = Number(node.dataset.cell) === cell;
        node.classList.toggle("rs-current", current);
        if (current) node.setAttribute("aria-current", "true");
        else node.removeAttribute("aria-current");
      });
      root.classList.toggle("rs-editing", state.editing);
      this.editButton.setAttribute("aria-pressed", String(state.editing));
      root.classList.toggle("rs-collapsed", state.collapsed);
      const { collapseLabel, expandLabel } = this.collapseButton.dataset;
      const label = state.collapsed ? expandLabel : collapseLabel;
      this.collapseButton.setAttribute("aria-expanded", String(!state.collapsed));
      this.collapseButton.setAttribute("aria-label", label);
      this.collapseButton.title = label;
      return { score, cell };
    }

    announce(message) { this.announcer.textContent = message; }

    // "Score 6, Risk 73.1%" once there is a score; cell is its strip cell
    announceScore(score, cell) {
      if (score == null) return;
      this.announce(fillMessage(this.root.dataset.scoreMessage, {
        score: formatNumber(score, this.model.score_bins.score_digits),
        risk: cell == null ? "—" : this.model.score_to_risk[cell].risk,
      }));
    }

    // the rows in the order of these ids (the saved order); rows it does not name keep theirs
    showOrder(order) {
      const total = this.list.querySelector(".rs-total");
      const byId = new Map(this.rows.map((row) => [row.feature.id, row]));
      order.forEach((id) => { if (byId.has(id)) total.before(byId.get(id).node); });
      this.syncOrder();
    }

    // the rows follow the DOM: their order, as ids, is the "move" action's
    syncOrder() {
      const order = [...this.list.querySelectorAll(".rs-feature")];
      this.rows.sort((a, b) => order.indexOf(a.node) - order.indexOf(b.node));
      return this.rows.map((row) => row.feature.id);
    }

    // move a row one place (the keyboard), among rows of its group; the order only, the points
    // are unchanged
    moveBy(row, step) {
      if (!this.state.editing) return;
      const index = this.rows.indexOf(row);
      const other = this.rows[index + step];
      if (!other || other.group !== row.group) return;
      if (step < 0) other.node.before(row.node); else other.node.after(row.node);
      const order = this.syncOrder();
      this.dispatch({ type: "move", order, announce: fillMessage(this.root.dataset.moveMessage, {
        label: row.shownLabel, position: index + step + 1, count: this.rows.length }) });
    }

    // drag a row by its handle: it swaps into the place of whichever neighbour of its group the
    // pointer passes the middle of. Pointer events, so a mouse, a pen and a finger all work
    startDrag(row, event) {
      if (!this.state.editing || event.button !== 0) return;
      event.preventDefault();
      // on the window, not pointer capture on the handle: moving the row would release it
      const { pointerId } = event;
      const node = row.node;
      node.classList.add("rs-dragging");
      document.body.classList.add("rs-drag-active");
      const rowOf = new Map(this.rows.map((other) => [other.node, other]));
      const sameGroup = (other) => rowOf.get(other)?.group === row.group;
      // the midpoint of a row: hit-testing only
      const middle = (other) => { const box = other.getBoundingClientRect(); return box.top + box.height / 2; };
      this.dragging = new AbortController();
      const options = { signal: this.dragging.signal };
      window.addEventListener("pointermove", (moveEvent) => {
        if (moveEvent.pointerId !== pointerId) return;
        // a fast pointer can pass several rows between two events: swap until it is in place
        for (;;) {
          const previous = node.previousElementSibling;
          const next = node.nextElementSibling;
          if (sameGroup(previous) && moveEvent.clientY < middle(previous)) previous.before(node);
          else if (sameGroup(next) && moveEvent.clientY > middle(next)) next.after(node);
          else break;
        }
      }, options);
      const end = (endEvent) => {
        if (endEvent.pointerId !== pointerId) return;
        this.dragging.abort();
        node.classList.remove("rs-dragging");
        document.body.classList.remove("rs-drag-active");
        this.dispatch({ type: "move", order: this.syncOrder() });
      };
      window.addEventListener("pointerup", end, options);
      window.addEventListener("pointercancel", end, options);
    }

    // The table wraps onto lines by itself (styles.css); read where each line begins and ends
    // off the layout and mark those cells, which draw each line's box and its labels. A label
    // wider than the fixed cell switches the table to the wide cells.
    markStripLines() {
      const [scores, risks] = [...this.strip.querySelectorAll("tr")].map((row) =>
        [...row.querySelectorAll(".rs-strip-cell")]);
      const { classList } = this.strip;
      if (!classList.contains("rs-strip-wide")
          && [...scores, ...risks].some((cell) => cell.scrollWidth > cell.clientWidth)) {
        classList.add("rs-strip-wide");
      }
      // every read before any write; unchanged lines write nothing (a write would call the
      // observer back)
      const tops = risks.map((cell) => cell.offsetTop);
      const lineStarts = tops.map((top, i) => i === 0 || top !== tops[i - 1]);
      const pattern = lineStarts.join();
      if (pattern === this.stripLinePattern) return;
      this.stripLinePattern = pattern;
      risks.forEach((cell, i) => {
        const end = i === risks.length - 1 || lineStarts[i + 1];
        [cell, scores[i]].forEach((node) => {
          node.classList.toggle("rs-line-start", lineStarts[i]);
          node.classList.toggle("rs-line-end", end);
        });
      });
    }
  }

  // every card on the page, once: mounting again leaves a mounted card alone
  const modelCards = new WeakMap();
  function mountModelCards(scope, model, storage) {
    scope.querySelectorAll("[data-model-card]").forEach((root) => {
      if (!modelCards.has(root)) modelCards.set(root, new ModelCard(root, model, storage));
    });
  }

  // ---- the page ------------------------------------------------------------------------------

  const data = JSON.parse(document.getElementById("report-data").textContent);

  // the Model card first: it needs no library, so it works even when Plotly fails to load
  mountModelCards(document, data.model);

  // Escape hides the definition shown under a summary row's label (its tooltip), until the
  // pointer leaves the label's cell or the label is focused again
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    document.querySelectorAll(".rs-label:hover, .rs-label:focus-within").forEach((cell) => {
      if (cell.classList.contains("rs-dismissed")) return;
      cell.classList.add("rs-dismissed");
      // whichever comes first restores it and drops both listeners
      const listening = new AbortController();
      const restore = () => { cell.classList.remove("rs-dismissed"); listening.abort(); };
      cell.addEventListener("mouseleave", restore, { signal: listening.signal });
      cell.addEventListener("focusin", restore, { signal: listening.signal });
    });
    if (document.activeElement?.classList.contains("rs-term")) document.activeElement.blur();
  });

  // Python builds the whole figure, styling included; it is drawn once, so Plotly may write
  // into it directly. Without Plotly (offline, blocked) the figure boxes stay empty.
  const plots = typeof Plotly === "undefined" ? []
    : [...document.querySelectorAll("#report [data-figure]")];
  // one handler sizes everything on the page, once per frame
  let resizing;
  window.addEventListener("resize", () => {
    cancelAnimationFrame(resizing);
    resizing = requestAnimationFrame(() => plots.forEach(fitToWidth));
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
