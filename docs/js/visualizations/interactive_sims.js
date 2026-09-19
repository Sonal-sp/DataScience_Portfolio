/**
 * Interactive Simulation Controllers
 * Bridges UI sliders with live mathematical calculations and Chart.js re-renders
 */

const SimController = {
  // 1. Threshold Classifier Simulation (ER Triage, Cardiac, Phishing, Audio Skip)
  updateThreshold(project, newThresh) {
    const tau = parseFloat(newThresh);
    const cfg = project.simConfig || {
      totalPos: 384, totalNeg: 616, costFN: 50000, costFP: 1200
    };

    // Calculate dynamic confusion matrix
    const sens = Math.min(0.99, Math.max(0.15, 1.0 - Math.pow(tau, 1.9)));
    const spec = Math.min(0.99, Math.max(0.20, Math.pow(tau, 0.45)));

    const tp = Math.round(cfg.totalPos * sens);
    const fn = cfg.totalPos - tp;
    const tn = Math.round(cfg.totalNeg * spec);
    const fp = cfg.totalNeg - tn;

    const prec = tp / (tp + fp + 1e-8);
    const f1 = (2 * prec * sens) / (prec + sens + 1e-8);
    const totalCost = fn * cfg.costFN + fp * cfg.costFP;

    // Update DOM Metrics
    const elThreshVal = document.getElementById('thresh-display-val');
    if (elThreshVal) elThreshVal.innerText = tau.toFixed(2);

    const elSens = document.getElementById('metric-sens');
    if (elSens) elSens.innerText = (sens * 100).toFixed(1) + '%';

    const elSpec = document.getElementById('metric-spec');
    if (elSpec) elSpec.innerText = (spec * 100).toFixed(1) + '%';

    const elPrec = document.getElementById('metric-prec');
    if (elPrec) elPrec.innerText = (prec * 100).toFixed(1) + '%';

    const elCost = document.getElementById('metric-cost');
    if (elCost) elCost.innerText = '$' + totalCost.toLocaleString();

    // Update Confusion Matrix Grid
    const elTP = document.getElementById('cm-tp');
    if (elTP) elTP.innerText = tp;
    const elFP = document.getElementById('cm-fp');
    if (elFP) elFP.innerText = fp;
    const elFN = document.getElementById('cm-fn');
    if (elFN) elFN.innerText = fn;
    const elTN = document.getElementById('cm-tn');
    if (elTN) elTN.innerText = tn;

    // Re-render chart operating point
    ChartEngine.renderROC('sim-chart-canvas', tau);
  },

  // 2. Pricing Elasticity Simulation
  updatePrice(project, newPrice) {
    const p = parseFloat(newPrice);
    const cfg = project.simConfig || { defaultCost: 25.0, defaultEta: -1.75, alpha: 6.8 };

    const q = Math.exp(cfg.alpha + cfg.defaultEta * Math.log(p));
    const revenue = p * q;
    const profit = (p - cfg.defaultCost) * q;
    const margin = (profit / revenue) * 100;

    const elPriceVal = document.getElementById('price-display-val');
    if (elPriceVal) elPriceVal.innerText = '$' + p.toFixed(2);

    const elDemand = document.getElementById('metric-demand');
    if (elDemand) elDemand.innerText = Math.round(q).toLocaleString() + ' units';

    const elProfit = document.getElementById('metric-profit');
    if (elProfit) elProfit.innerText = '$' + Math.round(profit).toLocaleString();

    const elMargin = document.getElementById('metric-margin');
    if (elMargin) elMargin.innerText = margin.toFixed(1) + '%';

    ChartEngine.renderPricingCurves('sim-chart-canvas', cfg.defaultCost, cfg.defaultEta, cfg.alpha);
  },

  // 3. Safety Stock & Supply Chain Simulation
  updateSafetyStock(project, newBuffer) {
    const buffer = parseInt(newBuffer, 10);
    const elBufferVal = document.getElementById('buffer-display-val');
    if (elBufferVal) elBufferVal.innerText = buffer + ' units';

    const serviceLevel = Math.min(99.9, (90 + buffer * 0.22)).toFixed(1);
    const elServ = document.getElementById('metric-service-level');
    if (elServ) elServ.innerText = serviceLevel + '%';

    const holdingCost = Math.round(buffer * 18.5);
    const elHold = document.getElementById('metric-holding-cost');
    if (elHold) elHold.innerText = '$' + holdingCost.toLocaleString();

    ChartEngine.renderInventoryForecast('sim-chart-canvas', buffer);
  },

  // 4. Volatility Sentiment Shock Simulator
  updateSentimentShock(project, newSentiment) {
    const s = parseFloat(newSentiment);
    const elSentVal = document.getElementById('sent-display-val');
    if (elSentVal) elSentVal.innerText = (s > 0 ? '+' : '') + s.toFixed(2);

    const projVol = Math.max(0.08, -s * 0.12 + 0.15);
    const elVol = document.getElementById('metric-proj-vol');
    if (elVol) elVol.innerText = (projVol * 100).toFixed(1) + '%';

    const regime = projVol > 0.22 ? 'HIGH STRESS' : (projVol > 0.15 ? 'NORMAL' : 'CALM');
    const elRegime = document.getElementById('metric-vol-regime');
    if (elRegime) elRegime.innerText = regime;

    ChartEngine.renderVolatilitySentiment('sim-chart-canvas', s);
  }
};
