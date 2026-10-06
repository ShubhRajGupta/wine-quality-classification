/**
 * Wine Quality Classification - Client-Side Prediction Engine & Project Interface
 * 
 * Machine Learning Architecture:
 * - Implements a calibrated Logistic Regression classification model
 *   trained on physicochemical properties from the UCI Wine Quality dataset.
 * - Normalized feature inputs are passed through a linear decision function
 *   and a sigmoid activation to yield a probability score P(Good) in [0, 1].
 * - Decision boundary threshold is set at 0.50 (50%):
 *     - Score >= 0.50 -> "Good"
 *     - Score <  0.50 -> "Bad"
 * - Multi-page navigation seamlessly transitions across 5 project views:
 *   Prediction, Dataset, Model, Results, and Project.
 */

// ============================================================================
// 1. Model Parameters & Calibration Constants (Standardized UCI Dataset)
// ============================================================================

/**
 * Feature normalization parameters (Mean and Standard Deviation)
 * derived from the UCI Red Wine Quality dataset (Cortez et al.).
 */
const FEATURE_SCALER = {
  fixed_acidity:        { mean: 8.32,    std: 1.74 },
  volatile_acidity:     { mean: 0.528,   std: 0.179 },
  citric_acid:          { mean: 0.271,   std: 0.195 },
  residual_sugar:       { mean: 2.539,   std: 1.410 },
  chlorides:            { mean: 0.0875,  std: 0.0471 },
  free_sulfur_dioxide:  { mean: 15.87,   std: 10.46 },
  total_sulfur_dioxide: { mean: 46.47,   std: 32.90 },
  density:              { mean: 0.99675, std: 0.00189 },
  ph:                   { mean: 3.311,   std: 0.154 },
  sulphates:            { mean: 0.658,   std: 0.170 },
  alcohol:              { mean: 10.423,  std: 1.066 }
};

/**
 * Calibrated logistic regression weights and bias.
 * Sign and magnitude reflect empirical statistical feature importance:
 * - Alcohol (+): Strongest positive predictor of wine quality.
 * - Volatile Acidity (-): Acetic acid spoils flavor; lower is better.
 * - Sulphates (+): Preserves freshness and wine stability.
 * - Total SO2 (-): High levels indicate oxidation or unpleasant pungency.
 */
const MODEL_WEIGHTS = {
  fixed_acidity:        0.12,
  volatile_acidity:    -0.88,
  citric_acid:          0.22,
  residual_sugar:       0.08,
  chlorides:           -0.24,
  free_sulfur_dioxide:  0.14,
  total_sulfur_dioxide:-0.36,
  density:             -0.20,
  ph:                  -0.12,
  sulphates:            0.58,
  alcohol:              1.12
};

const MODEL_INTERCEPT = -0.15; // Calibrated base log-odds

/**
 * Valid laboratory physiological bounds for sanity checking inputs.
 */
const PARAM_LIMITS = {
  fixed_acidity:        { min: 3.0,   max: 16.0,  name: "Fixed Acidity" },
  volatile_acidity:     { min: 0.10,  max: 1.60,  name: "Volatile Acidity" },
  citric_acid:          { min: 0.00,  max: 1.00,  name: "Citric Acid" },
  residual_sugar:       { min: 0.5,   max: 20.0,  name: "Residual Sugar" },
  chlorides:            { min: 0.010, max: 0.600, name: "Chlorides" },
  free_sulfur_dioxide:  { min: 1,     max: 75,    name: "Free Sulfur Dioxide" },
  total_sulfur_dioxide: { min: 5,     max: 300,   name: "Total Sulfur Dioxide" },
  density:              { min: 0.985, max: 1.005, name: "Density" },
  ph:                   { min: 2.70,  max: 4.10,  name: "pH" },
  sulphates:            { min: 0.30,  max: 2.00,  name: "Sulphates" },
  alcohol:              { min: 8.0,   max: 15.5,  name: "Alcohol" }
};

// ============================================================================
// 2. Realistic Dataset Presets for Evaluation
// ============================================================================

const SAMPLE_PRESETS = {
  good: {
    fixed_acidity: 8.5,
    volatile_acidity: 0.28,
    citric_acid: 0.49,
    residual_sugar: 2.6,
    chlorides: 0.075,
    free_sulfur_dioxide: 17,
    total_sulfur_dioxide: 34,
    density: 0.9956,
    ph: 3.30,
    sulphates: 0.86,
    alcohol: 12.8
  },
  bad: {
    fixed_acidity: 7.4,
    volatile_acidity: 0.70,
    citric_acid: 0.00,
    residual_sugar: 1.9,
    chlorides: 0.092,
    free_sulfur_dioxide: 11,
    total_sulfur_dioxide: 68,
    density: 0.9978,
    ph: 3.51,
    sulphates: 0.49,
    alcohol: 9.4
  }
};

// ============================================================================
// 3. Classification Algorithm & Inference Function
// ============================================================================

/**
 * Logistic Sigmoid Activation Function
 * @param {number} z - Linear log-odds
 * @returns {number} Probability in range (0, 1)
 */
function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

/**
 * Predict Wine Quality based on input feature values.
 * 
 * @param {Object} rawValues - Object mapping parameter keys to numerical values
 * @returns {Object} Prediction result containing category, confidence score, and primary drivers
 */
function predictWineQuality(rawValues) {
  let logOdds = MODEL_INTERCEPT;
  const normalizedContributions = [];

  for (const [key, weight] of Object.entries(MODEL_WEIGHTS)) {
    const rawVal = rawValues[key] !== undefined ? rawValues[key] : rawValues[key.toLowerCase()];
    const scaler = FEATURE_SCALER[key] || FEATURE_SCALER[key.toLowerCase()] || FEATURE_SCALER["ph"];
    const { mean, std } = scaler;

    // Standardize: z = (x - mean) / std
    const zScore = (rawVal - mean) / std;
    const contribution = zScore * weight;
    logOdds += contribution;

    normalizedContributions.push({
      key,
      contribution,
      rawVal
    });
  }

  // Calculate probability score via sigmoid function
  const probability = sigmoid(logOdds);
  const scorePercent = Math.round(probability * 100);

  // Classification Threshold
  const isGood = scorePercent >= 50;
  const category = isGood ? "Good" : "Bad";

  // Identify top influencing factor for transparency
  normalizedContributions.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  const topFactor = normalizedContributions[0];

  let factorSummary = "";
  if (topFactor.key === "alcohol") {
    factorSummary = `Alcohol content (${rawValues.alcohol}%)`;
  } else if (topFactor.key === "volatile_acidity") {
    factorSummary = `Volatile Acidity (${rawValues.volatile_acidity} g/dm³)`;
  } else if (topFactor.key === "sulphates") {
    factorSummary = `Sulphates (${rawValues.sulphates} g/dm³)`;
  } else {
    factorSummary = `${topFactor.key.replace(/_/g, " ")} (${topFactor.rawVal})`;
  }

  return {
    category,
    confidence: scorePercent,
    isGood,
    primaryDriver: factorSummary,
    details: {
      logOdds,
      rawValues
    }
  };
}

// ============================================================================
// 4. DOM Elements & State Management
// ============================================================================

const form = document.getElementById("wine-form");
const loadGoodBtn = document.getElementById("load-good-btn");
const loadBadBtn = document.getElementById("load-bad-btn");
const resetBtn = document.getElementById("reset-btn");

const resultEmptyState = document.getElementById("result-empty");
const resultCard = document.getElementById("result-card");
const resultBadge = document.getElementById("result-badge");
const badgeText = document.getElementById("badge-text");
const scorePercentage = document.getElementById("score-percentage");
const progressBarFill = document.getElementById("progress-bar-fill");
const progressBarBg = document.getElementById("quality-progress-bar");
const detailCategoryLabel = document.getElementById("detail-category-label");
const detailStatus = document.getElementById("detail-status");
const detailDrivers = document.getElementById("detail-drivers");
const formAlert = document.getElementById("form-alert");

// Navigation elements
const navTabs = document.querySelectorAll(".nav-item, .nav-tab, .nav-link");
const pageViews = document.querySelectorAll(".page-view");
const brandLink = document.getElementById("brand-link");

// All input field IDs
const inputIds = [
  "fixed_acidity",
  "volatile_acidity",
  "citric_acid",
  "residual_sugar",
  "chlorides",
  "free_sulfur_dioxide",
  "total_sulfur_dioxide",
  "density",
  "ph",
  "sulphates",
  "alcohol"
];

// ============================================================================
// 5. Form Validation & Helpers
// ============================================================================

/**
 * Clear individual field error state
 */
function clearFieldError(id) {
  const group = document.getElementById(`group-${id}`);
  const errEl = document.getElementById(`err-${id}`);
  if (group) group.classList.remove("has-error");
  if (errEl) {
    errEl.textContent = "";
    errEl.style.display = "none";
  }
}

/**
 * Display an error message for an individual input field
 */
function setFieldError(id, message) {
  const group = document.getElementById(`group-${id}`);
  const errEl = document.getElementById(`err-${id}`);
  if (group) group.classList.add("has-error");
  if (errEl) {
    errEl.textContent = message;
    errEl.style.display = "block";
  }
}

/**
 * Validate all form inputs
 * @returns {Object|null} Validated feature map if valid, null otherwise
 */
function validateAndGetFormValues() {
  let hasErrors = false;
  const values = {};
  hideFormAlert();

  for (const id of inputIds) {
    const input = document.getElementById(id);
    const rawVal = String(input.value !== undefined && input.value !== null ? input.value : "").trim();
    const limits = PARAM_LIMITS[id];

    // Check empty
    if (rawVal === "") {
      setFieldError(id, "Value is required.");
      hasErrors = true;
      continue;
    }

    const num = parseFloat(rawVal);

    // Check valid number
    if (isNaN(num)) {
      setFieldError(id, "Please enter a valid number.");
      hasErrors = true;
      continue;
    }

    // Check bounds
    if (num < limits.min || num > limits.max) {
      setFieldError(id, `Must be between ${limits.min} and ${limits.max}.`);
      hasErrors = true;
      continue;
    }

    clearFieldError(id);
    values[id] = num;
  }

  // Cross-parameter consistency validation
  if (!hasErrors) {
    if (values.free_sulfur_dioxide > values.total_sulfur_dioxide) {
      setFieldError("free_sulfur_dioxide", "Free SO₂ cannot exceed Total SO₂.");
      hasErrors = true;
    }
  }

  if (hasErrors) {
    showFormAlert("Please review the highlighted input parameters before predicting.");
    return null;
  }

  return values;
}

function showFormAlert(message) {
  if (formAlert) {
    formAlert.textContent = message;
    formAlert.style.display = "block";
    formAlert.classList.remove("alert-exit");
    formAlert.classList.add("alert-enter");
  }
}

function hideFormAlert() {
  if (formAlert && formAlert.style.display !== "none") {
    formAlert.classList.remove("alert-enter");
    formAlert.classList.add("alert-exit");
    setTimeout(() => {
      formAlert.textContent = "";
      formAlert.style.display = "none";
      formAlert.classList.remove("alert-exit");
    }, 160);
  }
}

// ============================================================================
// 6. UI Rendering & Interactions
// ============================================================================

/**
 * Display prediction results in the result card with incoming/outgoing animations
 */
function renderPredictionResult(result) {
  const showResultCard = () => {
    // Category badge
    if (badgeText) badgeText.textContent = result.category;

    if (result.isGood) {
      if (resultBadge) {
        resultBadge.className = "result-badge badge-good badge-pop";
      }
      if (progressBarFill) {
        progressBarFill.className = "progress-fill fill-good";
      }
      if (detailCategoryLabel) {
        detailCategoryLabel.textContent = "High Quality Wine";
      }
      if (detailStatus) {
        detailStatus.textContent = "Meets High Quality Benchmark";
      }
    } else {
      if (resultBadge) {
        resultBadge.className = "result-badge badge-bad badge-pop";
      }
      if (progressBarFill) {
        progressBarFill.className = "progress-fill fill-bad";
      }
      if (detailCategoryLabel) {
        detailCategoryLabel.textContent = "Poor / Low Quality Wine";
      }
      if (detailStatus) {
        detailStatus.textContent = "Below Quality Benchmark";
      }
    }

    // Progress Bar & Confidence Score
    if (scorePercentage) scorePercentage.textContent = `${result.confidence}%`;
    if (progressBarFill) {
      progressBarFill.style.width = "0%";
      setTimeout(() => {
        progressBarFill.style.width = `${result.confidence}%`;
      }, 50);
    }
    if (progressBarBg) progressBarBg.setAttribute("aria-valuenow", result.confidence);

    // Decisive Factor
    if (detailDrivers) detailDrivers.textContent = result.primaryDriver;

    // Smooth scroll into view on mobile
    if (window.innerWidth < 768 && resultCard) {
      resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  };

  // State Transition: Empty -> Populated
  if (resultEmptyState && resultEmptyState.style.display !== "none") {
    resultEmptyState.classList.add("state-exit");
    setTimeout(() => {
      resultEmptyState.style.display = "none";
      resultEmptyState.classList.remove("state-exit");

      if (resultCard) {
        resultCard.style.display = "block";
        resultCard.classList.remove("state-exit");
        resultCard.classList.add("state-enter");
      }
      showResultCard();
    }, 140);
  } else {
    // Already populated: trigger re-calculation pulse
    if (resultCard) {
      resultCard.classList.remove("state-pulse");
      void resultCard.offsetWidth; // Trigger reflow for animation restart
      resultCard.classList.add("state-pulse");
    }
    showResultCard();
  }
}

/**
 * Fill form fields with preset sample data and highlight inputs
 */
function loadSampleData(sample) {
  hideFormAlert();
  for (const [key, value] of Object.entries(sample)) {
    const input = document.getElementById(key);
    if (input) {
      input.value = value;
      clearFieldError(key);
      input.classList.remove("input-pulse");
      void input.offsetWidth; // Restart pulse
      input.classList.add("input-pulse");
    }
  }
}

/**
 * Reset form and prediction display with outgoing animation
 */
function resetForm() {
  if (form) form.reset();
  hideFormAlert();
  for (const id of inputIds) {
    clearFieldError(id);
  }

  if (resultCard && resultCard.style.display !== "none") {
    resultCard.classList.add("state-exit");
    setTimeout(() => {
      resultCard.style.display = "none";
      resultCard.classList.remove("state-exit");

      if (resultEmptyState) {
        resultEmptyState.style.display = "block";
        resultEmptyState.classList.remove("state-exit");
        resultEmptyState.classList.add("state-enter");
      }
    }, 140);
  }
}

// ============================================================================
// 7. Multi-Page Navigation Controller
// ============================================================================

/**
 * Switch active page view with incoming and outgoing animations
 * @param {string} targetPageId - Page identifier ('prediction', 'dataset', 'model', 'results', 'project')
 */
function switchPage(targetPageId) {
  const validPages = ["prediction", "dataset", "model", "results", "project"];
  if (!validPages.includes(targetPageId)) {
    targetPageId = "prediction";
  }

  const targetView = document.getElementById(`page-${targetPageId}`);
  let currentActive = null;
  pageViews.forEach((view) => {
    if (view.classList.contains("active")) {
      currentActive = view;
    }
  });

  if (currentActive === targetView) return;

  // Update nav tabs immediately
  navTabs.forEach((tab) => {
    const isTarget = tab.dataset.page === targetPageId;
    if (isTarget) {
      tab.classList.add("active");
      tab.setAttribute("aria-selected", "true");
    } else {
      tab.classList.remove("active");
      tab.setAttribute("aria-selected", "false");
    }
  });

  // Outgoing animation for current view, incoming for target view
  if (currentActive) {
    currentActive.classList.add("page-exit");
    currentActive.classList.remove("page-enter");

    setTimeout(() => {
      currentActive.classList.remove("active", "page-exit");

      if (targetView) {
        targetView.classList.add("active", "page-enter");
      }
      window.scrollTo({ top: 0, behavior: "smooth" });
    }, 140);
  } else if (targetView) {
    targetView.classList.add("active", "page-enter");
  }

  // Sync browser URL hash without causing viewport jump
  if (window.location.hash !== `#${targetPageId}`) {
    history.replaceState(null, "", `#${targetPageId}`);
  }
}

// ============================================================================
// 8. Event Listeners
// ============================================================================

// Form Submission -> Predict
if (form) {
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const values = validateAndGetFormValues();
    if (values) {
      const result = predictWineQuality(values);
      renderPredictionResult(result);
    }
  });
}

// Load Presets
if (loadGoodBtn) {
  loadGoodBtn.addEventListener("click", () => {
    loadSampleData(SAMPLE_PRESETS.good);
    const values = validateAndGetFormValues();
    if (values) {
      renderPredictionResult(predictWineQuality(values));
    }
  });
}

if (loadBadBtn) {
  loadBadBtn.addEventListener("click", () => {
    loadSampleData(SAMPLE_PRESETS.bad);
    const values = validateAndGetFormValues();
    if (values) {
      renderPredictionResult(predictWineQuality(values));
    }
  });
}

// Reset Button
if (resetBtn) {
  resetBtn.addEventListener("click", resetForm);
}

// Clear field errors as user modifies inputs
inputIds.forEach((id) => {
  const input = document.getElementById(id);
  if (input) {
    input.addEventListener("input", () => {
      clearFieldError(id);
      hideFormAlert();
    });
  }
});

// Tab Navigation click handlers
navTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const targetPage = tab.dataset.page;
    switchPage(targetPage);
  });
});

if (brandLink) {
  brandLink.addEventListener("click", (e) => {
    e.preventDefault();
    switchPage("prediction");
  });
}

// Handle browser Back/Forward or direct bookmark hash
window.addEventListener("hashchange", () => {
  const hash = window.location.hash.replace("#", "");
  if (hash) {
    switchPage(hash);
  }
});

// Startup Initialization
document.addEventListener("DOMContentLoaded", () => {
  // Check initial URL hash
  const initialHash = window.location.hash.replace("#", "");
  if (initialHash) {
    switchPage(initialHash);
  } else {
    switchPage("prediction");
  }

  // Pre-fill realistic good wine sample so evaluation is immediately ready
  loadSampleData(SAMPLE_PRESETS.good);

  // Initialize animated waving dotted canvas background
  initWaveBackground();
});

// If script runs after DOMContentLoaded already fired
if (document.readyState === "complete" || document.readyState === "interactive") {
  initWaveBackground();
}

// ============================================================================
// 8. Animated Waving Dotted Canvas Background
// ============================================================================

/**
 * Initializes and runs the ambient waving dotted canvas mesh.
 * Renders an undulating grid of particles in harmonic sine waves
 * with subtle wine/bordeaux highlights and cursor ripple interactivity.
 */
function initWaveBackground() {
  const canvas = document.getElementById("bg-wave-canvas");
  if (!canvas || typeof canvas.getContext !== "function") return;

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  let width = 0;
  let height = 0;
  let dpr = 1;
  let animId = null;
  let startTime = performance.now();

  // Mouse interaction state
  const mouse = {
    x: -9999,
    y: -9999,
    targetX: -9999,
    targetY: -9999,
    active: false,
    decayTimeout: null
  };

  // Color layers from trough (subtle slate) to crest (rich Bordeaux wine)
  const layerColors = [
    "rgba(148, 163, 184, 0.20)", // Trough: Slate
    "rgba(140, 100, 120, 0.25)", // Mid-low
    "rgba(130, 55, 75, 0.32)",   // Mid
    "rgba(118, 35, 55, 0.40)",   // Mid-high: Wine
    "rgba(107, 29, 47, 0.50)"    // Crest: Rich Bordeaux
  ];

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    width = window.innerWidth;
    height = window.innerHeight;

    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = width + "px";
    canvas.style.height = height + "px";

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);
  }

  // Handle cursor movement
  window.addEventListener("mousemove", (e) => {
    mouse.targetX = e.clientX;
    mouse.targetY = e.clientY;
    mouse.active = true;

    clearTimeout(mouse.decayTimeout);
    mouse.decayTimeout = setTimeout(() => {
      mouse.active = false;
    }, 2500);
  }, { passive: true });

  window.addEventListener("mouseleave", () => {
    mouse.active = false;
  });

  // Debounced resize listener
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(resize, 100);
  }, { passive: true });

  // Initial sizing
  resize();

  // Animation render loop
  function render(now) {
    const time = (now - startTime) * 0.001;

    // Smooth mouse interpolation
    if (mouse.active) {
      mouse.x += (mouse.targetX - mouse.x) * 0.12;
      mouse.y += (mouse.targetY - mouse.y) * 0.12;
    } else {
      mouse.x += (-9999 - mouse.x) * 0.05;
      mouse.y += (-9999 - mouse.y) * 0.05;
    }

    ctx.clearRect(0, 0, width, height);

    // Spacing between dots
    const spacing = width < 768 ? 32 : 28;
    const cols = Math.ceil(width / spacing) + 2;
    const rows = Math.ceil(height / spacing) + 2;

    // 5 buckets to batch arc calls by elevation color
    const buckets = [[], [], [], [], []];

    for (let i = -1; i < cols; i++) {
      const baseX = i * spacing;

      for (let j = -1; j < rows; j++) {
        const baseY = j * spacing;

        // Multi-frequency wave harmonic functions
        const w1 = Math.sin(baseX * 0.0032 + baseY * 0.0022 - time * 1.35);
        const w2 = Math.cos(baseX * 0.0025 - baseY * 0.0030 + time * 0.95);
        const w3 = Math.sin((baseX + baseY) * 0.0048 - time * 1.65) * 0.35;

        // Normalized wave elevation in [-1, 1]
        const wave = (w1 + w2 + w3) / 2.35;

        // Displacements
        let dispY = wave * 13;
        let dispX = Math.cos(baseX * 0.0028 - time * 0.75) * 4;

        let renderX = baseX + dispX;
        let renderY = baseY + dispY;

        // Interactive mouse wave perturbation
        if (mouse.active) {
          const dx = renderX - mouse.x;
          const dy = renderY - mouse.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          const maxDist = 150;

          if (dist < maxDist && dist > 1) {
            const influence = (1 - dist / maxDist);
            const ripple = Math.sin(dist * 0.06 - time * 4.0) * 8.0 * influence;
            renderX += (dx / dist) * (4.0 * influence);
            renderY += (dy / dist) * (4.0 * influence) + ripple;
          }
        }

        // Elevation factor [0, 1] for sizing & color layer
        const norm = Math.max(0, Math.min(1, (wave + 1) * 0.5));
        const radius = 1.1 + norm * 1.5;

        // Assign to elevation bucket
        const bIndex = Math.min(4, Math.floor(norm * 5));
        buckets[bIndex].push(renderX, renderY, radius);
      }
    }

    // Render batch paths per layer
    for (let b = 0; b < 5; b++) {
      const pts = buckets[b];
      if (pts.length === 0) continue;

      ctx.fillStyle = layerColors[b];
      ctx.beginPath();
      for (let k = 0; k < pts.length; k += 3) {
        ctx.moveTo(pts[k] + pts[k + 2], pts[k + 1]);
        ctx.arc(pts[k], pts[k + 1], pts[k + 2], 0, Math.PI * 2);
      }
      ctx.fill();
    }

    animId = requestAnimationFrame(render);
  }

  // Respect page visibility to save resources
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      if (animId) cancelAnimationFrame(animId);
      animId = null;
    } else if (!animId) {
      startTime = performance.now();
      animId = requestAnimationFrame(render);
    }
  });

  animId = requestAnimationFrame(render);
}
