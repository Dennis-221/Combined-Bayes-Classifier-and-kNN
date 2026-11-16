"use strict";

const $ = (id) => document.getElementById(id);

/* ============================================================
   Helpers
   ============================================================ */

// Builds DOM nodes without innerHTML, so CSV-derived text (column names,
// categories, cell values) can never be interpreted as markup.
function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "text") node.textContent = value;
    else node.setAttribute(key, value);
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

// Same idea for SVG elements (they need the SVG namespace).
const SVG_NS = "http://www.w3.org/2000/svg";
function s(tag, attrs = {}, ...children) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "text") node.textContent = value;
    else node.setAttribute(key, value);
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child);
  }
  return node;
}

// replaceChildren() takes separate nodes, not arrays: flatten and skip empties first.
function fill(parent, ...children) {
  parent.replaceChildren(
    ...children.flat(Infinity).filter((c) => c !== null && c !== undefined && c !== false)
  );
}

const num = (n) => Number(n).toLocaleString();
const pct = (v) => `${(v * 100).toFixed(1)}%`;
const pctProb = (p) => (p > 0 && p < 0.001 ? "<0.1%" : pct(p));
const plural = (n, word, pluralForm = `${word}s`) => `${num(n)} ${n === 1 ? word : pluralForm}`;

async function request(url, options) {
  let response;
  try {
    response = await fetch(url, options);
  } catch (_) {
    return { ok: false, status: 0, data: { error: "Could not reach the server. Try again." } };
  }
  let data;
  try {
    data = await response.json();
  } catch (_) {
    data = { error: `The server sent an unexpected response (HTTP ${response.status}).` };
  }
  return { ok: response.ok, status: response.status, data };
}

const postJSON = (url, body = {}) =>
  request(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

const errorsOf = (data) =>
  data.errors && data.errors.length ? data.errors : [data.error || "Something went wrong."];

function showInline(el, message) {
  el.textContent = message;
  el.hidden = false;
}

function hideInline(el) {
  el.hidden = true;
  el.textContent = "";
}

function notice(kind, message) {
  return h(
    "div",
    { class: `notice notice--${kind}` },
    h("span", { class: "notice__tag", text: kind === "error" ? "Error" : "Warning" }),
    h("span", { text: message })
  );
}

function renderNotices(box, { errors = [], warnings = [], heading = null } = {}) {
  box.replaceChildren();
  if (errors.length && heading) box.append(h("p", { class: "notices-heading", text: heading }));
  errors.forEach((message) => box.append(notice("error", message)));
  warnings.forEach((message) => box.append(notice("warn", message)));
  box.hidden = box.childElementCount === 0;
}

function statCard(value, label, tone) {
  return h(
    "div",
    { class: tone ? `stat stat--${tone}` : "stat" },
    h("span", { class: "stat__value", text: value }),
    h("span", { class: "stat__label", text: label })
  );
}

function barRow(label, fraction, valueText, isTop) {
  return h(
    "div",
    { class: isTop ? "bar bar--top" : "bar" },
    h("span", { class: "bar__label", title: label, text: label }),
    h(
      "span",
      { class: "bar__track" },
      h("span", { class: "bar__fill", style: `width: ${Math.max(0, Math.min(1, fraction)) * 100}%` })
    ),
    h("span", { class: "bar__value", text: valueText })
  );
}

function dataTable(headers, rows, className) {
  return h(
    "div",
    { class: "table-wrap" },
    h(
      "table",
      { class: className },
      h("thead", {}, h("tr", {}, headers)),
      h("tbody", {}, rows)
    )
  );
}

function setLocked(sectionId, locked) {
  const section = $(sectionId);
  section.classList.toggle("is-locked", locked);
  section.querySelectorAll("button, select, input").forEach((el) => {
    el.disabled = locked;
  });
  if (!locked && sectionId === "targetSection") {
    $("analyzeBtn").disabled = !$("targetSelect").value;
  }
  if (!locked && sectionId === "trainSection") syncKGrid();
}

// Runs a button's action with a loading state. The button stays disabled
// afterwards only if its section has been locked in the meantime.
async function run(button, task, onFailure) {
  button.classList.add("is-loading");
  button.disabled = true;
  try {
    await task();
  } catch (err) {
    console.error(err);
    onFailure("Something went wrong while showing the results. See the browser console for details.");
  } finally {
    button.classList.remove("is-loading");
    button.disabled = button.closest(".pipeline-section").classList.contains("is-locked");
  }
}

/* ============================================================
   Page state
   ============================================================ */

let targetInfo = null;                 // last /target response
let activeAlgorithm = "naive_bayes";   // algorithm behind the results currently shown
let knnData = null;                    // last KNN run response
let predictModelId = null;             // which trained model the predict form was built for

function clearPredict() {
  fill($("predictFields"));
  $("predictKRow").hidden = true;
  $("predictResult").hidden = true;
  renderNotices($("predictErrors"));
  predictModelId = null;
  setLocked("predictSection", true);
}

function clearTrain() {
  $("trainResult").hidden = true;
  $("knnBlocks").hidden = true;
  $("knnPlotResult").hidden = true;
  hideInline($("plotError"));
  knnData = null;
  renderNotices($("trainNotices"));
  clearPredict();
}

function clearTarget() {
  $("targetResult").hidden = true;
  hideInline($("targetError"));
  targetInfo = null;
  setLocked("trainSection", true);
  clearTrain();
}

function clearExplore() {
  $("exploreResult").hidden = true;
  hideInline($("exploreError"));
}

/* ============================================================
   01 Upload
   ============================================================ */

const dropzone = $("dropzone");
const fileInput = $("fileInput");

function setReading(isReading) {
  $("dropzoneIdle").hidden = isReading;
  $("dropzoneLoading").hidden = !isReading;
}

function resetAll() {
  $("uploadSummary").hidden = true;
  dropzone.hidden = false;
  hideInline($("uploadError"));
  fileInput.value = "";
  setLocked("exploreSection", true);
  setLocked("targetSection", true);
  clearExplore();
  clearTarget();
}

function onUploaded(data) {
  $("uploadFilename").textContent = data.filename;
  $("uploadRows").textContent = num(data.rows);
  $("uploadCols").textContent = num(data.columns.length);
  dropzone.hidden = true;
  $("uploadSummary").hidden = false;

  fill(
    $("targetSelect"),
    h("option", { value: "", text: "Choose a column…" }),
    data.columns.map((c) =>
      h("option", { value: c.name, text: `${c.name}  (${c.dtype}, ${num(c.unique)} unique)` })
    )
  );

  clearExplore();
  clearTarget();
  setLocked("exploreSection", false);
  setLocked("targetSection", false);
}

async function uploadFile(file) {
  hideInline($("uploadError"));

  if (!file.name.toLowerCase().endsWith(".csv")) {
    showInline($("uploadError"), "Only .csv files are accepted.");
    return;
  }

  setReading(true);
  const formData = new FormData();
  formData.append("file", file);

  const { ok, data } = await request("/upload", { method: "POST", body: formData });
  setReading(false);
  fileInput.value = "";

  if (!ok) {
    showInline($("uploadError"), data.error || "Something went wrong reading that file.");
    return;
  }
  onUploaded(data);
}

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) uploadFile(fileInput.files[0]);
});
["dragenter", "dragover"].forEach((name) =>
  dropzone.addEventListener(name, (e) => {
    e.preventDefault();
    dropzone.classList.add("is-dragover");
  })
);
["dragleave", "drop"].forEach((name) =>
  dropzone.addEventListener(name, (e) => {
    e.preventDefault();
    dropzone.classList.remove("is-dragover");
  })
);
dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length > 0) uploadFile(e.dataTransfer.files[0]);
});
$("resetBtn").addEventListener("click", resetAll);

/* ============================================================
   02 Explore
   ============================================================ */

function renderExplore(data) {
  $("exploreStats").replaceChildren(
    statCard(num(data.rows), "rows"),
    statCard(num(data.column_count), "columns"),
    statCard(num(data.missing_cells), "missing cells", data.missing_cells ? "warn" : null)
  );

  $("exploreColumns").replaceChildren(
    dataTable(
      [
        h("th", { text: "Column" }),
        h("th", { text: "Type" }),
        h("th", { class: "num", text: "Missing" }),
        h("th", { class: "num", text: "Missing %" }),
        h("th", { class: "num", text: "Unique values" }),
      ],
      data.columns.map((c) =>
        h(
          "tr",
          {},
          h("td", { title: c.name, text: c.name }),
          h("td", { class: "type-tag", text: c.dtype }),
          h("td", { class: c.missing ? "num cell-warn" : "num", text: num(c.missing) }),
          h("td", { class: c.missing ? "num cell-warn" : "num", text: `${c.missing_pct}%` }),
          h("td", { class: "num", text: num(c.unique) })
        )
      )
    )
  );

  const sample = data.sample;
  $("exploreSample").replaceChildren(
    dataTable(
      [h("th", { class: "num", text: "#" }), sample.columns.map((name) => h("th", { text: name }))],
      sample.rows.map((row) =>
        h(
          "tr",
          {},
          h("td", { class: "row-num", text: row.row_number }),
          row.cells.map((cell) =>
            cell === null
              ? h("td", { class: "cell-missing", text: "NaN" })
              : h("td", { title: cell, text: cell })
          )
        )
      )
    )
  );
  $("exploreSampleNote").textContent =
    `A random sample of ${sample.rows.length} of ${num(data.rows)} rows. ` +
    "Click Explore dataset again for a different sample.";

  $("exploreResult").hidden = false;
}

$("exploreBtn").addEventListener("click", () =>
  run(
    $("exploreBtn"),
    async () => {
      hideInline($("exploreError"));
      const { ok, data } = await postJSON("/explore");
      if (!ok) return showInline($("exploreError"), errorsOf(data)[0]);
      renderExplore(data);
    },
    (message) => showInline($("exploreError"), message)
  )
);

/* ============================================================
   03 Target and features
   ============================================================ */

$("targetSelect").addEventListener("change", () => {
  $("analyzeBtn").disabled = !$("targetSelect").value;
  clearTarget();
});

function featureGroup(name, features, chipClass) {
  return h(
    "div",
    { class: "feature-group" },
    h("span", { class: "feature-group__name", text: `${name} (${features.length})` }),
    features.length
      ? h(
          "div",
          { class: "chips" },
          features.map((f) => h("span", { class: `chip ${chipClass}`, text: f }))
        )
      : h("span", { class: "none", text: "None" })
  );
}

function renderTarget(data) {
  renderNotices($("targetNotices"), {
    errors: data.errors,
    warnings: data.warnings,
    heading: data.errors.length ? "This target can't be used to train a model yet." : null,
  });

  $("targetStats").replaceChildren(
    statCard(num(data.usable_records), `usable records (of ${num(data.total_records)})`),
    statCard(num(data.n_features), "features"),
    statCard(num(data.numeric_features.length), "numeric features"),
    statCard(num(data.categorical_features.length), "categorical features", "info")
  );

  $("distributionTitle").textContent =
    `Target distribution: ${data.target} (${plural(data.n_classes, "class", "classes")})`;
  fill(
    $("distributionBars"),
    data.class_distribution.map((d) =>
      barRow(d.label, d.pct / 100, `${num(d.count)}  (${d.pct}%)`, false)
    )
  );
  const tooMany = data.n_classes > 0 && data.class_distribution.length === 0;
  $("distributionNote").hidden = !tooMany;
  if (tooMany) {
    $("distributionNote").textContent =
      `This column has ${num(data.n_classes)} distinct values, too many to chart as classes.`;
  }

  $("featureLists").replaceChildren(
    featureGroup("Numeric", data.numeric_features, "chip--num"),
    featureGroup("Categorical", data.categorical_features, "chip--cat")
  );

  $("excludedBlock").hidden = data.excluded_columns.length === 0;
  fill(
    $("excludedList"),
    data.excluded_columns.map((c) =>
      h(
        "li",
        {},
        h("span", { class: "chip", text: c.name }),
        h("span", { class: "excluded-list__reason", text: c.reason })
      )
    )
  );

  $("targetResult").hidden = false;
}

$("analyzeBtn").addEventListener("click", () =>
  run(
    $("analyzeBtn"),
    async () => {
      hideInline($("targetError"));
      clearTarget();
      const { ok, data } = await postJSON("/target", { target: $("targetSelect").value });
      if (!ok) return showInline($("targetError"), errorsOf(data)[0]);
      targetInfo = data;
      renderTarget(data);
      setLocked("trainSection", !data.ready);
      if (data.ready) selectDefaultKs();
    },
    (message) => showInline($("targetError"), message)
  )
);

/* ============================================================
   04 Train and evaluate: shared pieces
   ============================================================ */

function confusionMatrix(labels, matrix) {
  const max = Math.max(1, ...matrix.flat());
  return dataTable(
    [h("th", { class: "cm__corner", text: "actual \\ predicted" }), labels.map((l) => h("th", { text: l }))],
    matrix.map((row, i) =>
      h(
        "tr",
        {},
        h("th", { text: labels[i] }),
        row.map((value, j) => {
          const ratio = value / max;
          const rgb = i === j ? "31, 111, 84" : "181, 118, 43";
          return h("td", {
            class: value === 0 ? "cm__zero" : null,
            style:
              value === 0
                ? null
                : `background: rgba(${rgb}, ${(0.12 + 0.7 * ratio).toFixed(2)}); color: ${ratio > 0.5 ? "#fff" : "inherit"}`,
            text: value,
          });
        })
      )
    ),
    "cm"
  );
}

// Metric cards, per-class table and confusion matrix. Used for Naive Bayes
// and for whichever k is selected in the KNN results.
function renderPerformance({ title, metrics, baseline, perClass, labels, matrix }) {
  $("performanceTitle").textContent = title;

  const metric = (label, value, sub) =>
    h(
      "div",
      { class: "metric" },
      h("span", { class: "metric__value", text: pct(value) }),
      h("span", { class: "metric__label", text: label }),
      h("span", { class: "metric__sub", text: sub })
    );
  $("metricCards").replaceChildren(
    metric("Accuracy", metrics.accuracy, "share predicted correctly"),
    metric("Precision", metrics.precision, "weighted average"),
    metric("Recall", metrics.recall, "weighted average"),
    metric("F1 score", metrics.f1, "weighted average")
  );
  $("metricNote").textContent =
    "Precision, recall and F1 are averaged across classes, weighted by how many test records each class has. " +
    `A model that always predicted the most common class would score ${pct(baseline)} accuracy on this test split.`;

  $("perClassTable").replaceChildren(
    dataTable(
      [
        h("th", { text: "Class" }),
        h("th", { class: "num", text: "Precision" }),
        h("th", { class: "num", text: "Recall" }),
        h("th", { class: "num", text: "F1" }),
        h("th", { class: "num", text: "Test records" }),
      ],
      perClass.map((c) =>
        h(
          "tr",
          {},
          h("td", { title: c.label, text: c.label }),
          h("td", { class: "num", text: pct(c.precision) }),
          h("td", { class: "num", text: c.support ? pct(c.recall) : "n/a" }),
          h("td", { class: "num", text: c.support ? pct(c.f1) : "n/a" }),
          h("td", { class: "num", text: num(c.support) })
        )
      )
    )
  );

  $("confusionMatrix").replaceChildren(confusionMatrix(labels, matrix));
}

function splitNote(data, lead) {
  return (
    `${lead} ${Math.round(data.test_size * 100)}% of records held out for testing, ` +
    `${data.stratified ? "split stratified by class" : "split not stratified"}, random seed ${data.random_state}.`
  );
}

/* ---------- algorithm choice and the k grid ---------- */

const K_MAX = parseInt($("kGrid").dataset.maxK, 10) || 40;

function kInputs() {
  return Array.from($("kGrid").querySelectorAll("input"));
}

function selectedKs() {
  return kInputs()
    .filter((input) => input.checked)
    .map((input) => parseInt(input.value, 10));
}

// The largest k this split allows: you can't have more neighbors than training records.
function knnMaxK() {
  if (!targetInfo) return K_MAX;
  const n = targetInfo.usable_records;
  const nTest = Math.ceil(parseFloat($("testSizeSelect").value) * n);
  return Math.max(1, Math.min(K_MAX, n - nTest));
}

function updateKSummary() {
  const chosen = selectedKs().length;
  $("kSummary").textContent =
    `${chosen ? plural(chosen, "value") : "No values"} of k selected. This split allows k up to ${knnMaxK()}.`;
}

function syncKGrid() {
  const maxK = knnMaxK();
  const locked = $("trainSection").classList.contains("is-locked");
  kInputs().forEach((input) => {
    const k = parseInt(input.value, 10);
    input.disabled = locked || k > maxK;
    if (k > maxK) input.checked = false;
  });
  updateKSummary();
}

function applyKPreset(name) {
  const maxK = knnMaxK();
  kInputs().forEach((input) => {
    const k = parseInt(input.value, 10);
    input.checked = k <= maxK && (name === "all" || (name === "odd" && k % 2 === 1));
  });
  updateKSummary();
}

function selectDefaultKs() {
  const cap = Math.min(knnMaxK(), 25);
  kInputs().forEach((input) => {
    const k = parseInt(input.value, 10);
    input.checked = k <= cap && k % 2 === 1;
  });
  updateKSummary();
}

fill(
  $("kGrid"),
  Array.from({ length: K_MAX }, (_, i) =>
    h(
      "label",
      { class: "k-chip" },
      h("input", { type: "checkbox", value: i + 1, "aria-label": `k = ${i + 1}` }),
      h("span", { text: i + 1 })
    )
  )
);
$("kGrid").addEventListener("change", updateKSummary);
document.querySelectorAll("[data-k-preset]").forEach((button) =>
  button.addEventListener("click", () => applyKPreset(button.dataset.kPreset))
);

function onAlgorithmChange() {
  const isKnn = $("algorithmSelect").value === "knn";
  $("knnOptions").hidden = !isKnn;
  $("trainBtnLabel").textContent = isKnn ? "Run KNN" : "Train model";
  $("trainBtnSub").textContent = isKnn
    ? "Record the test error rate for each selected k"
    : "Fit on the training split, then evaluate on the test split";
}

$("algorithmSelect").addEventListener("change", () => {
  onAlgorithmChange();
  clearTrain();
});
$("testSizeSelect").addEventListener("change", () => {
  syncKGrid();
  clearTrain();
});
onAlgorithmChange();
syncKGrid();

/* ============================================================
   04 Naive Bayes
   ============================================================ */

function renderTrain(data) {
  activeAlgorithm = "naive_bayes";
  knnData = null;
  $("knnBlocks").hidden = true;
  $("knnPlotResult").hidden = true;
  $("detailKField").hidden = true;

  renderNotices($("trainNotices"), { warnings: data.warnings });
  $("trainStats").replaceChildren(
    statCard(num(data.n_train), "training samples"),
    statCard(num(data.n_test), "testing samples", "info"),
    statCard(num(data.n_features), "features used")
  );
  $("trainSplitNote").textContent = splitNote(data, `${data.algorithm}.`);

  renderPerformance({
    title: "Performance on the test split",
    metrics: data.metrics,
    baseline: data.baseline_accuracy,
    perClass: data.per_class,
    labels: data.labels,
    matrix: data.confusion_matrix,
  });
  $("trainResult").hidden = false;

  ensurePredictForm(data);
  setLocked("predictSection", false);
}

/* ============================================================
   04 K-nearest neighbors
   ============================================================ */

function kNote(k, data) {
  const notes = [];
  if (k === data.optimal_k) notes.push("lowest error");
  if (k === data.elbow_k) notes.push("elbow");
  return notes.length ? ` (${notes.join(", ")})` : "";
}

function kTags(k, data) {
  return [
    k === data.optimal_k ? h("span", { class: "chip chip--num", text: "lowest error" }) : null,
    k === data.elbow_k ? h("span", { class: "chip chip--cat", text: "elbow" }) : null,
  ];
}

function highlightClass(k, data) {
  if (k === data.elbow_k) return "row-elbow";
  if (k === data.optimal_k) return "row-optimal";
  return null;
}

function renderKnnRecords(data) {
  $("knnRecords").replaceChildren(
    dataTable(
      [
        h("th", { text: "k" }),
        h("th", { class: "num", text: "Error rate" }),
        h("th", { class: "num", text: "Misclassified" }),
        h("th", { class: "num", text: "Accuracy" }),
      ],
      data.records.map((r) =>
        h(
          "tr",
          { class: highlightClass(r.k, data) },
          h("td", { class: "k-cell" }, String(r.k), kTags(r.k, data)),
          h("td", { class: "num", text: pct(r.error_rate) }),
          h("td", { class: "num", text: `${num(r.errors)} of ${num(data.n_test)}` }),
          h("td", { class: "num", text: pct(r.accuracy) })
        )
      )
    )
  );

  const notes = [
    `Error rate is the share of the ${num(data.n_test)} test records that were misclassified.`,
  ];
  const tiedOthers = data.tied_ks.filter((k) => k !== data.optimal_k);
  if (tiedOthers.length) {
    notes.push(`Also at the lowest error rate: k = ${tiedOthers.join(", ")}. The smallest k is the one marked.`);
  }
  if (data.records.length < 3) {
    notes.push("With fewer than three recorded values there is no bend to find, so the elbow is the lowest-error k.");
  }
  $("knnRecordsNote").textContent = notes.join(" ");

  const canPlot = data.records.length >= 2;
  $("plotBtn").disabled = !canPlot;
  $("plotBtnSub").textContent = canPlot
    ? "Marks the lowest-error k and the elbow, and compares every recorded k with the elbow"
    : "Record at least two values of k first";
}

function renderKnnDetails() {
  const k = parseInt($("detailK").value, 10);
  const record = knnData.records.find((r) => r.k === k);
  const details = knnData.details[String(k)];
  renderPerformance({
    title: `Performance at k = ${k}`,
    metrics: record,
    baseline: knnData.baseline_accuracy,
    perClass: details.per_class,
    labels: knnData.labels,
    matrix: details.confusion_matrix,
  });
}

function renderKnn(data) {
  activeAlgorithm = "knn";
  knnData = data;

  renderNotices($("trainNotices"), { warnings: data.warnings });
  $("trainStats").replaceChildren(
    statCard(num(data.n_train), "training samples"),
    statCard(num(data.n_test), "testing samples", "info"),
    statCard(num(data.n_features), "features used"),
    statCard(num(data.n_dimensions), "dimensions after encoding")
  );
  $("trainSplitNote").textContent = splitNote(
    data,
    `${data.algorithm}. Numeric features are standardized and categorical features one-hot encoded, ` +
      "then records are compared by Euclidean distance."
  );

  $("knnBlocks").hidden = false;
  renderKnnRecords(data);

  // k selector for the performance block: keep the previous choice if it is still recorded
  const previous = $("detailK").value;
  fill(
    $("detailK"),
    data.records.map((r) => h("option", { value: r.k, text: `k = ${r.k}${kNote(r.k, data)}` }))
  );
  $("detailK").value = data.records.some((r) => String(r.k) === previous)
    ? previous
    : String(data.optimal_k);
  $("detailKField").hidden = false;
  renderKnnDetails();

  $("trainResult").hidden = false;
  ensurePredictForm(data);
  setLocked("predictSection", false);
}

$("detailK").addEventListener("change", () => {
  if (knnData) renderKnnDetails();
});

/* ---------- the k-against-error chart (inline SVG) ---------- */

// Round-number tick positions covering [lo, hi].
function niceTicks(lo, hi, target = 5) {
  const rough = (hi - lo) / target;
  const magnitude = Math.pow(10, Math.floor(Math.log10(rough)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((v) => v >= rough);
  const first = Math.floor(lo / step + 1e-9) * step;
  const last = Math.ceil(hi / step - 1e-9) * step;
  const ticks = [];
  for (let i = 0; first + i * step <= last + step / 2; i++) {
    ticks.push(Number((first + i * step).toFixed(6)));
  }
  return ticks;
}

function errorCurveChart(data) {
  const W = 820;
  const H = 400;
  const M = { top: 40, right: 28, bottom: 56, left: 62 };
  const plotW = W - M.left - M.right;
  const plotH = H - M.top - M.bottom;

  const points = data.points.map((p) => ({ k: p.k, e: p.error_rate * 100, errors: p.errors }));
  const eMin = Math.min(...points.map((p) => p.e));
  const eMax = Math.max(...points.map((p) => p.e));
  const pad = Math.max(eMax - eMin, 1) * 0.2;
  const yTicks = niceTicks(Math.max(0, eMin - pad), eMax + pad);
  const yLo = yTicks[0];
  const yHi = yTicks[yTicks.length - 1];

  const kMin = points[0].k;
  const kMax = points[points.length - 1].k;
  const inset = 16;
  const x = (k) =>
    kMax === kMin
      ? M.left + plotW / 2
      : M.left + inset + ((k - kMin) / (kMax - kMin)) * (plotW - 2 * inset);
  const y = (e) => M.top + plotH - ((e - yLo) / (yHi - yLo)) * plotH;
  const fmtTick = (v) => `${parseFloat(v.toFixed(2))}%`;

  const xTickKs =
    points.length <= 20
      ? points.map((p) => p.k)
      : points.map((p) => p.k).filter((k) => k === kMin || k % 5 === 0);

  const optimal = points.find((p) => p.k === data.optimal_k);
  const elbow = points.find((p) => p.k === data.elbow_k);
  const same = data.elbow_is_optimal;

  const optimalText = `${same ? "Lowest error and elbow" : "Lowest error"}: k = ${optimal.k} (${pct(data.optimal_error_rate)})`;
  const elbowText = `Elbow: k = ${elbow.k} (${pct(data.elbow_error_rate)})`;

  // Callouts: try six positions around a marker and keep the one that covers the
  // curve the least, stays inside the plot, and doesn't collide with another label.
  const curvePixels = [];
  points.forEach((p, i) => {
    const x0 = x(p.k);
    const y0 = y(p.e);
    curvePixels.push([x0, y0]);
    if (i + 1 < points.length) {
      const x1 = x(points[i + 1].k);
      const y1 = y(points[i + 1].e);
      const steps = Math.ceil(Math.hypot(x1 - x0, y1 - y0) / 4);
      for (let t = 1; t < steps; t++) {
        curvePixels.push([x0 + ((x1 - x0) * t) / steps, y0 + ((y1 - y0) * t) / steps]);
      }
    }
  });
  const overlaps = (a, b) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
  const placed = [];
  const markers = [optimal, elbow].map((p) => ({ x: x(p.k) - 16, y: y(p.e) - 16, w: 32, h: 32 }));

  function placeCallout(point, text, preference) {
    const cx = x(point.k);
    const cy = y(point.e);
    const w = text.length * 6.9;
    const options = {
      above: { anchor: "middle", tx: cx, ty: cy - 18 },
      below: { anchor: "middle", tx: cx, ty: cy + 30 },
      aboveRight: { anchor: "start", tx: cx + 14, ty: cy - 12 },
      belowRight: { anchor: "start", tx: cx + 14, ty: cy + 26 },
      aboveLeft: { anchor: "end", tx: cx - 14, ty: cy - 12 },
      belowLeft: { anchor: "end", tx: cx - 14, ty: cy + 26 },
    };
    let best = null;
    preference.forEach((name, order) => {
      const o = options[name];
      const rect = {
        x: o.anchor === "start" ? o.tx : o.anchor === "end" ? o.tx - w : o.tx - w / 2,
        y: o.ty - 11,
        w,
        h: 15,
      };
      let score = order * 0.5; // earlier preference wins ties
      score += curvePixels.filter(
        ([px, py]) => px > rect.x - 4 && px < rect.x + rect.w + 4 && py > rect.y - 4 && py < rect.y + rect.h + 4
      ).length;
      if (rect.x < M.left + 2 || rect.x + rect.w > W - M.right - 2 || rect.y < 8 || rect.y + rect.h > M.top + plotH - 4) {
        score += 1000;
      }
      if (placed.some((r) => overlaps(rect, r))) score += 500;
      if (markers.some((m) => overlaps(rect, m))) score += 500;
      if (!best || score < best.score) best = { score, rect, ...o };
    });
    placed.push(best.rect);
    return best;
  }

  const optimalPlace = placeCallout(optimal, optimalText, ["below", "belowRight", "belowLeft", "aboveRight", "aboveLeft", "above"]);
  const elbowPlace = same
    ? null
    : placeCallout(elbow, elbowText, ["above", "aboveRight", "aboveLeft", "belowRight", "belowLeft", "below"]);

  const summary =
    `Error rate against k. Lowest error rate ${pct(data.optimal_error_rate)} at k = ${optimal.k}. ` +
    (same ? "The elbow is at the same k." : `Elbow at k = ${elbow.k} with ${pct(data.elbow_error_rate)}.`);

  const path = points.map((p, i) => `${i ? "L" : "M"}${x(p.k).toFixed(1)} ${y(p.e).toFixed(1)}`).join(" ");

  return s(
    "svg",
    { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": summary },

    // grid + y axis labels
    yTicks.map((v) =>
      s(
        "g",
        {},
        s("line", { class: "grid-line", x1: M.left, x2: W - M.right, y1: y(v), y2: y(v) }),
        s("text", { x: M.left - 10, y: y(v) + 4, "text-anchor": "end", text: fmtTick(v) })
      )
    ),
    s("line", { class: "axis-line", x1: M.left, x2: W - M.right, y1: M.top + plotH, y2: M.top + plotH }),
    s("line", { class: "axis-line", x1: M.left, x2: M.left, y1: M.top, y2: M.top + plotH }),

    // x axis labels and titles
    xTickKs.map((k) => s("text", { x: x(k), y: M.top + plotH + 20, "text-anchor": "middle", text: k })),
    s("text", {
      class: "axis-title",
      x: M.left + plotW / 2,
      y: H - 12,
      "text-anchor": "middle",
      text: "k (number of neighbors)",
    }),
    s("text", {
      class: "axis-title",
      transform: `translate(16 ${M.top + plotH / 2}) rotate(-90)`,
      "text-anchor": "middle",
      text: "Error rate on the test split",
    }),

    // guide lines from the highlighted points down to the axis
    same
      ? null
      : s("line", {
          class: "guide guide--elbow",
          x1: x(elbow.k), x2: x(elbow.k), y1: y(elbow.e), y2: M.top + plotH,
        }),
    s("line", {
      class: "guide guide--optimal",
      x1: x(optimal.k), x2: x(optimal.k), y1: y(optimal.e), y2: M.top + plotH,
    }),

    // the curve and its points (each with a tooltip)
    s("path", { class: "series", d: path }),
    points.map((p) =>
      s(
        "circle",
        { class: "point", cx: x(p.k), cy: y(p.e), r: 3.5 },
        s("title", {
          text: `k = ${p.k}: error rate ${pct(p.e / 100)} (${p.errors} of ${data.n_test} test records misclassified)`,
        })
      )
    ),

    // highlighted points: elbow (diamond) and lowest error (large green dot)
    same
      ? null
      : s(
          "rect",
          {
            class: "elbow",
            "data-role": "elbow",
            x: -6, y: -6, width: 12, height: 12,
            transform: `translate(${x(elbow.k)} ${y(elbow.e)}) rotate(45)`,
          },
          s("title", { text: elbowText })
        ),
    s("circle", { class: "optimal-halo", cx: x(optimal.k), cy: y(optimal.e), r: 14 }),
    s(
      "circle",
      { class: "optimal", "data-role": "optimal", cx: x(optimal.k), cy: y(optimal.e), r: 7 },
      s("title", { text: optimalText })
    ),

    // callouts
    s("text", {
      class: "callout callout--optimal",
      x: optimalPlace.tx, y: optimalPlace.ty, "text-anchor": optimalPlace.anchor, text: optimalText,
    }),
    elbowPlace
      ? s("text", {
          class: "callout callout--elbow",
          x: elbowPlace.tx, y: elbowPlace.ty, "text-anchor": elbowPlace.anchor, text: elbowText,
        })
      : null
  );
}

function deltaText(row) {
  if (row.verdict === "elbow") return "reference";
  if (row.verdict === "same") return "same";
  const size = Math.abs(row.delta_pts);
  const digits = size < 0.05 ? 2 : 1;
  return `${row.delta_pts > 0 ? "+" : "\u2212"}${size.toFixed(digits)} pts`;
}

function renderCurve(data) {
  fill($("knnChart"), errorCurveChart(data));

  fill(
    $("knnLegend"),
    h(
      "li",
      {},
      h("span", { class: "legend__mark legend__mark--optimal" }),
      data.elbow_is_optimal
        ? `Lowest error rate and elbow (k = ${data.optimal_k})`
        : `Lowest error rate (k = ${data.optimal_k})`
    ),
    data.elbow_is_optimal
      ? null
      : h(
          "li",
          {},
          h("span", { class: "legend__mark legend__mark--elbow" }),
          `Elbow (k = ${data.elbow_k})`
        ),
    h("li", {}, h("span", { class: "legend__mark legend__mark--point" }), "Other recorded values of k")
  );

  const notes = [];
  if (data.elbow_is_optimal) {
    notes.push(
      "The elbow and the lowest-error k are the same here: the curve has no earlier bend, so no smaller " +
        "k captures most of the improvement."
    );
  } else {
    notes.push(
      `The elbow is k = ${data.elbow_k} (${pct(data.elbow_error_rate)}). Among the values up to the ` +
        `lowest-error k (${data.optimal_k}, ${pct(data.optimal_error_rate)}), it is where the curve bends ` +
        "most sharply and the error stops improving quickly. It is a rule of thumb, so judge it against the plot."
    );
  }
  const tiedOthers = data.tied_ks.filter((k) => k !== data.optimal_k);
  if (tiedOthers.length) {
    notes.push(`Also at the lowest error rate: k = ${tiedOthers.join(", ")}. The smallest k is the one marked.`);
  }
  notes.push(
    "k is picked by looking at error on the test split, so the lowest error rate is slightly optimistic for new data."
  );
  $("knnElbowNote").textContent = notes.join(" ");

  $("compareTitle").textContent = `Comparison with the elbow point (k = ${data.elbow_k})`;
  $("compareTally").replaceChildren(
    statCard(num(data.tally.better), "values of k with a lower error than the elbow"),
    statCard(num(data.tally.same), "values with the same error"),
    statCard(num(data.tally.worse), "values with a higher error", "warn")
  );
  $("knnCompare").replaceChildren(
    dataTable(
      [
        h("th", { text: "k" }),
        h("th", { class: "num", text: "Error rate" }),
        h("th", { class: "num", text: "Precision" }),
        h("th", { class: "num", text: "Recall" }),
        h("th", { class: "num", text: "F1" }),
        h("th", { class: "num", text: "Error change vs elbow" }),
      ],
      data.comparison.map((row) =>
        h(
          "tr",
          { class: highlightClass(row.k, data) },
          h("td", { class: "k-cell" }, String(row.k), kTags(row.k, data)),
          h("td", { class: "num", text: pct(row.error_rate) }),
          h("td", { class: "num", text: pct(row.precision) }),
          h("td", { class: "num", text: pct(row.recall) }),
          h("td", { class: "num", text: pct(row.f1) }),
          h("td", { class: `num delta--${row.verdict}`, text: deltaText(row) })
        )
      )
    )
  );

  $("knnPlotResult").hidden = false;
}

async function loadCurve() {
  hideInline($("plotError"));
  const { ok, data } = await postJSON("/knn/curve");
  if (!ok) {
    $("knnPlotResult").hidden = true;
    showInline($("plotError"), errorsOf(data)[0]);
    return;
  }
  renderCurve(data);
}

$("plotBtn").addEventListener("click", () =>
  run($("plotBtn"), loadCurve, (message) => showInline($("plotError"), message))
);

$("knnClearBtn").addEventListener("click", () =>
  run(
    $("knnClearBtn"),
    async () => {
      await postJSON("/knn/clear");
      clearTrain();
    },
    (message) => renderNotices($("trainNotices"), { errors: [message] })
  )
);

/* ============================================================
   04 Train button (both algorithms)
   ============================================================ */

$("trainBtn").addEventListener("click", () =>
  run(
    $("trainBtn"),
    async () => {
      renderNotices($("trainNotices"));
      const algorithm = $("algorithmSelect").value;
      const payload = { algorithm, test_size: parseFloat($("testSizeSelect").value) };

      if (algorithm === "knn") {
        const ks = selectedKs();
        if (ks.length === 0) {
          renderNotices($("trainNotices"), { errors: ["Select at least one value of k."] });
          return;
        }
        payload.params = { ks };
      }

      const { ok, data } = await postJSON("/train", payload);
      if (!ok) {
        renderNotices($("trainNotices"), {
          errors: errorsOf(data),
          warnings: data.warnings || [],
          heading: "The model could not be trained.",
        });
        return;
      }

      if (data.algorithm_key === "knn") {
        renderKnn(data);
        // keep an open plot in step with the newly recorded values
        if (!$("knnPlotResult").hidden) await loadCurve();
      } else {
        renderTrain(data);
      }
    },
    (message) => renderNotices($("trainNotices"), { errors: [message] })
  )
);

/* ============================================================
   05 Predict a new instance
   ============================================================ */

function buildPredictForm(schema) {
  fill(
    $("predictFields"),
    schema.map((field, i) => {
      const id = `predict-${i}`;
      let control;
      let hint = null;
      if (field.kind === "numeric") {
        control = h("input", {
          class: "control",
          id,
          type: "number",
          step: "any",
          inputmode: "decimal",
          placeholder: field.median === null ? "" : `median ${field.median}`,
          "data-feature": field.name,
        });
        hint = field.min === null ? null : `Training range ${field.min} to ${field.max}.`;
      } else {
        control = h(
          "select",
          { class: "control", id, "data-feature": field.name },
          h("option", { value: "", text: "Blank (most common)" }),
          field.categories.map((c) => h("option", { value: c, text: c }))
        );
        hint = `${plural(field.categories.length, "category", "categories")} seen in training.`;
      }
      return h(
        "div",
        { class: "field" },
        h("label", { class: "field__label", for: id, text: field.name }),
        control,
        hint ? h("span", { class: "field__hint", text: hint }) : null
      );
    })
  );
}

// Builds the feature form once per trained model, so values typed into it
// survive when more values of k are recorded for the same KNN model.
function ensurePredictForm(data) {
  if (predictModelId !== data.model_id) {
    buildPredictForm(data.schema);
    predictModelId = data.model_id;
    $("predictResult").hidden = true;
    renderNotices($("predictErrors"));
  }

  const isKnn = data.algorithm_key === "knn";
  $("predictKRow").hidden = !isKnn;
  if (isKnn) {
    const select = $("predictK");
    const previous = select.value;
    fill(
      select,
      data.records.map((r) => h("option", { value: r.k, text: `k = ${r.k}${kNote(r.k, data)}` }))
    );
    select.value = data.records.some((r) => String(r.k) === previous)
      ? previous
      : String(data.optimal_k);
  }
}

function renderPrediction(data) {
  $("predictValue").textContent = data.prediction;
  fill(
    $("predictBars"),
    data.probabilities.map((p, i) =>
      barRow(
        p.label,
        p.probability,
        data.k ? `${p.votes} of ${data.k}  (${pctProb(p.probability)})` : pctProb(p.probability),
        i === 0
      )
    )
  );
  $("predictExplain").textContent = data.explanation;

  const filled = data.filled_columns;
  $("predictNote").hidden = filled.length === 0;
  if (filled.length) {
    const shown = filled.slice(0, 6).join(", ");
    const more = filled.length > 6 ? ` and ${filled.length - 6} more` : "";
    $("predictNote").textContent = `Blank fields were filled in from the training data: ${shown}${more}.`;
  }
  $("predictResult").hidden = false;
}

$("predictBtn").addEventListener("click", () =>
  run(
    $("predictBtn"),
    async () => {
      const values = {};
      $("predictFields")
        .querySelectorAll("[data-feature]")
        .forEach((el) => {
          values[el.dataset.feature] = el.value;
        });

      const body = { values };
      if (activeAlgorithm === "knn") body.k = parseInt($("predictK").value, 10);

      const { ok, data } = await postJSON("/predict", body);
      if (!ok) {
        $("predictResult").hidden = true;
        renderNotices($("predictErrors"), { errors: errorsOf(data) });
        return;
      }
      renderNotices($("predictErrors"));
      renderPrediction(data);
    },
    (message) => renderNotices($("predictErrors"), { errors: [message] })
  )
);
