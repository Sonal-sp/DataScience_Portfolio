/**
 * High-Precision Chart.js Visualization Engine
 * Developer dark-mode aesthetics: zinc-950 gridlines, emerald/cyan data series
 */

const ChartEngine = {
  activeCharts: {},

  destroy(id) {
    if (this.activeCharts[id]) {
      this.activeCharts[id].destroy();
      delete this.activeCharts[id];
    }
  },

  // Base Dark Theme Options
  getBaseOptions(titleText) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      plugins: {
        legend: {
          labels: { color: '#a1a1aa', font: { family: "'JetBrains Mono', monospace", size: 11 } }
        },
        title: {
          display: !!titleText,
          text: titleText,
          color: '#f4f4f5',
          font: { family: "'JetBrains Mono', monospace", size: 12, weight: '600' }
        },
        tooltip: {
          backgroundColor: '#18181b',
          titleColor: '#fff',
          bodyColor: '#a1a1aa',
          borderColor: '#27272a',
          borderWidth: 1,
          padding: 10,
          displayColors: true
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(39, 39, 42, 0.6)' },
          ticks: { color: '#71717a', font: { family: "'JetBrains Mono', monospace", size: 10 } }
        },
        y: {
          grid: { color: 'rgba(39, 39, 42, 0.6)' },
          ticks: { color: '#71717a', font: { family: "'JetBrains Mono', monospace", size: 10 } }
        }
      }
    };
  },

  // 1. ROC Curve with dynamic operating threshold point
  renderROC(canvasId, currentThreshold = 0.35) {
    this.destroy(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    // Simulated high-fidelity ROC curve
    const fprPoints = [0.0, 0.02, 0.05, 0.10, 0.18, 0.28, 0.42, 0.60, 0.80, 1.0];
    const tprPoints = [0.0, 0.45, 0.68, 0.82, 0.91, 0.95, 0.97, 0.99, 1.0, 1.0];

    // Compute operating point based on current threshold
    // As threshold decreases, sensitivity (TPR) and FPR increase
    const opFpr = Math.min(1.0, Math.max(0.01, (1.0 - currentThreshold) * 0.45)).toFixed(3);
    const opTpr = Math.min(1.0, Math.max(0.10, 1.0 - Math.pow(currentThreshold, 1.8))).toFixed(3);

    const options = this.getBaseOptions("Receiver Operating Characteristic (ROC) & Dynamic Operating Cutoff");
    options.scales.x.title = { display: true, text: "False Positive Rate (1 - Specificity)", color: '#71717a' };
    options.scales.y.title = { display: true, text: "True Positive Rate (Sensitivity)", color: '#71717a' };

    this.activeCharts[canvasId] = new Chart(ctx, {
      type: 'line',
      data: {
        labels: fprPoints,
        datasets: [
          {
            label: 'Random Forest Champion (AUC = 0.908)',
            data: tprPoints,
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            fill: true,
            tension: 0.3,
            borderWidth: 2,
            pointRadius: 2
          },
          {
            label: 'Random Chance Baseline',
            data: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
            borderColor: '#52525b',
            borderDash: [5, 5],
            borderWidth: 1.5,
            fill: false,
            pointRadius: 0
          },
          {
            label: `Operating Point (tau = ${Number(currentThreshold).toFixed(2)})`,
            data: [{ x: Number(opFpr), y: Number(opTpr) }],
            type: 'scatter',
            backgroundColor: '#06b6d4',
            borderColor: '#fff',
            borderWidth: 2,
            pointRadius: 8,
            pointHoverRadius: 10
          }
        ]
      },
      options: options
    });
  },

  // 2. Demand Elasticity & Profit Maximization Curves
  renderPricingCurves(canvasId, unitCost = 25.0, eta = -1.75, alpha = 6.8) {
    this.destroy(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const prices = [];
    const revenues = [];
    const profits = [];
    const demands = [];

    for (let p = 26; p <= 120; p += 4) {
      const q = Math.exp(alpha + eta * Math.log(p));
      const rev = (p * q) / 1000; // in $K
      const prof = ((p - unitCost) * q) / 1000;
      prices.push(`$${p}`);
      revenues.push(rev.toFixed(2));
      profits.push(prof.toFixed(2));
      demands.push(q.toFixed(0));
    }

    const options = this.getBaseOptions("Price vs Gross Profit & Revenue Response Surface");
    options.scales.x.title = { display: true, text: "Offered Unit Price ($)", color: '#71717a' };
    options.scales.y.title = { display: true, text: "Projected Value ($ in Thousands)", color: '#71717a' };

    this.activeCharts[canvasId] = new Chart(ctx, {
      type: 'line',
      data: {
        labels: prices,
        datasets: [
          {
            label: 'Gross Profit ($K) - Peak at Optimal Price',
            data: profits,
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.15)',
            fill: true,
            tension: 0.3,
            borderWidth: 2.5
          },
          {
            label: 'Topline Revenue ($K)',
            data: revenues,
            borderColor: '#06b6d4',
            borderDash: [4, 4],
            fill: false,
            tension: 0.3,
            borderWidth: 2
          }
        ]
      },
      options: options
    });
  },

  // 3. Time Series Multi-Step Demand & Safety Stock Buffer
  renderInventoryForecast(canvasId, safetyBuffer = 38) {
    this.destroy(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const days = Array.from({ length: 30 }, (_, i) => `Day ${i + 1}`);
    const actualDemand = [
      82, 85, 90, 78, 115, 130, 88, 84, 89, 93, 120, 140, 86, 82, 91,
      95, 118, 135, 87, 83, 90, 96, 122, 138, 89, 85, 92, 98, 125, 142
    ];
    const forecastedDemand = actualDemand.map(v => v + Math.round((Math.random() - 0.5) * 8));
    const safetyLine = forecastedDemand.map(v => v + Number(safetyBuffer));

    const options = this.getBaseOptions("Multi-Store SKU Demand Forecast & Safety Buffer");
    options.scales.y.title = { display: true, text: "Units Demanded", color: '#71717a' };

    this.activeCharts[canvasId] = new Chart(ctx, {
      type: 'line',
      data: {
        labels: days,
        datasets: [
          {
            label: 'Actual In-Store Demand',
            data: actualDemand,
            borderColor: '#f4f4f5',
            borderWidth: 1.5,
            pointRadius: 2,
            tension: 0.2
          },
          {
            label: 'LightGBM Forecast (WAPE = 11.4%)',
            data: forecastedDemand,
            borderColor: '#06b6d4',
            borderWidth: 2,
            pointRadius: 2,
            tension: 0.2
          },
          {
            label: `Dynamic Safety Stock Buffer (+${safetyBuffer} units)`,
            data: safetyLine,
            borderColor: '#f59e0b',
            borderDash: [6, 4],
            borderWidth: 2,
            fill: false,
            pointRadius: 0
          }
        ]
      },
      options: options
    });
  },

  // 4. Volatility & Sentiment Correlation Chart
  renderVolatilitySentiment(canvasId, sentimentShock = -0.5) {
    this.destroy(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const days = Array.from({ length: 25 }, (_, i) => `T-${25 - i}`);
    const historicalVol = [0.14, 0.15, 0.13, 0.16, 0.14, 0.15, 0.17, 0.16, 0.18, 0.19, 0.17, 0.18, 0.20, 0.19, 0.21, 0.22, 0.20, 0.23, 0.22, 0.25, 0.24, 0.26, 0.25, 0.28, 0.27];
    
    // Impact of current sentiment shock on projected next-day volatility
    const predictedVol = historicalVol.slice();
    const shockImpact = Math.max(0.08, -sentimentShock * 0.12 + 0.15);
    predictedVol.push(shockImpact.toFixed(3));
    days.push("Tomorrow (T+1)");

    const options = this.getBaseOptions("Realized Market Volatility & Sentiment Shock Response");
    options.scales.y.title = { display: true, text: "Annualized Volatility (sigma)", color: '#71717a' };

    this.activeCharts[canvasId] = new Chart(ctx, {
      type: 'line',
      data: {
        labels: days,
        datasets: [
          {
            label: 'Realized Volatility',
            data: predictedVol,
            borderColor: '#f43f5e',
            backgroundColor: 'rgba(244, 63, 94, 0.1)',
            fill: true,
            tension: 0.3,
            borderWidth: 2
          }
        ]
      },
      options: options
    });
  },

  // 5. IoT Sensor Run-to-Failure Trajectory
  renderTurbofanRUL(canvasId, currentCycle = 150) {
    this.destroy(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const cycles = Array.from({ length: 220 }, (_, i) => i + 1);
    const tempSensor = cycles.map(c => {
      const wear = Math.pow(c / 220, 2.5) * 22;
      return (642.0 + wear + (Math.random() - 0.5) * 1.5).toFixed(1);
    });

    const options = this.getBaseOptions("Turbofan Sensor 2 Temperature Wear Degradation Trajectory");
    options.scales.x.title = { display: true, text: "Operating Cycle", color: '#71717a' };
    options.scales.y.title = { display: true, text: "Exhaust Gas Temp (°C)", color: '#71717a' };

    this.activeCharts[canvasId] = new Chart(ctx, {
      type: 'line',
      data: {
        labels: cycles,
        datasets: [
          {
            label: 'Sensor 2 Temperature Trajectory',
            data: tempSensor,
            borderColor: '#06b6d4',
            borderWidth: 1.5,
            pointRadius: 0
          }
        ]
      },
      options: options
    });
  }
};
