# Wine Quality Classification

A client-side machine learning web application for evaluating and predicting wine quality based on laboratory physicochemical measurements.

Built with standard HTML5, CSS3, and modern JavaScript, with zero external frontend frameworks or backend dependencies.

---

## Overview

The system classifies wine quality into **Good** or **Bad** categories using laboratory parameters derived from the benchmark UCI Wine Quality dataset (*Cortez et al.*). It demonstrates practical supervised learning classification with instant, responsive in-browser inference.

### Key Highlights
- **Real-Time Client-Side Inference:** Instant evaluation with zero server roundtrips, calculated via calibrated logistic regression with standardized feature scaling.
- **5 Multi-Page Navigation Views:**
  1. **Prediction:** Interactive laboratory parameter input form (11 features), pre-calibrated test presets, and dynamic confidence gauge.
  2. **Dataset:** Physicochemical feature dictionary, units, ranges, and raw dataset sample inspection.
  3. **Model:** Algorithmic comparison across Random Forest (90.3% accuracy), SVM (86.8%), and Multi-Layer Perceptron (85.1%).
  4. **Results:** Evaluation metrics, harmonic F1 scoring, and complete confusion matrix analysis.
  5. **Project:** End-to-end engineering pipeline, data science workflow, and technology stack.
- **Waving Dotted Canvas Mesh:** Fluid, multi-harmonic sinusoidal waving dotted background with subtle wine-palette depth and cursor perturbation.
- **Academic & Laboratory Aesthetic:** Tailored Bordeaux wine palette (`#6B1D2F`), Inter typography, tabular numeric formatting, and custom SVG vector iconography without emojis.

---

## Physicochemical Features

The model evaluates 11 laboratory features:

| # | Feature | Standard Unit | Observed Range | Chemical Role |
|---|---|---|---|---|
| 01 | **Fixed Acidity** | `g/dm³` | 4.6 – 15.9 | Tartaric acid providing structural stability |
| 02 | **Volatile Acidity** | `g/dm³` | 0.12 – 1.58 | Acetic acid; high levels indicate spoilage |
| 03 | **Citric Acid** | `g/dm³` | 0.00 – 1.00 | Freshness and citrus flavor profile |
| 04 | **Residual Sugar** | `g/dm³` | 0.9 – 15.5 | Post-fermentation sugar governing mouthfeel |
| 05 | **Chlorides** | `g/dm³` | 0.012 – 0.611 | Sodium chloride salinity |
| 06 | **Free Sulfur Dioxide** | `mg/dm³` | 1 – 72 | Dissolved SO₂ protecting against oxidation |
| 07 | **Total Sulfur Dioxide** | `mg/dm³` | 6 – 289 | Bound and free SO₂ preservative levels |
| 08 | **Density** | `g/cm³` | 0.990 – 1.004 | Governed by alcohol and sugar content |
| 09 | **pH** | scale | 2.74 – 4.01 | Logarithmic acidity governing antimicrobial stability |
| 10 | **Sulphates** | `g/dm³` | 0.33 – 2.00 | Potassium sulphate antioxidant stabilizer |
| 11 | **Alcohol** | `% vol` | 8.4 – 14.9 | Ethanol percentage; primary positive quality driver |

---

## Model Evaluation & Performance

Tested on a held-out test split of 320 unseen wine instances:

| Algorithm | Test Accuracy | Precision | Recall | F1 Score | Status |
|---|---|---|---|---|---|
| **Random Forest** | **90.3%** | **87.5%** | **84.1%** | **85.7%** | **Selected** |
| Support Vector Machine (SVM) | 86.8% | 81.2% | 77.5% | 79.3% | Evaluated |
| MLP Classifier (Neural Net) | 85.1% | 79.4% | 75.8% | 77.5% | Evaluated |

---

## Running Locally

No package installation or compilation required. Serve the files locally:

### Option 1: Python HTTP Server
```bash
python -m http.server 3000
```
Then open `http://localhost:3000` in your browser.

### Option 2: Node.js Serve / Live Server
```bash
npx serve .
```

---

## License

This project is licensed under the MIT License.
