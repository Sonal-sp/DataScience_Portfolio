/**
 * Project Detail View Component
 * Multi-tab interactive workspace: Simulator, Algorithm Deconstructed, Benchmarks, Code
 */

const ProjectDetailComponent = {
  currentTab: 'visualizer', // 'visualizer' | 'algorithms' | 'benchmarks' | 'code'

  render(project) {
    const container = document.getElementById('viewport-container');
    if (!container) return;

    container.innerHTML = `
      <div class="workspace-viewport">
        <!-- Breadcrumb & Header -->
        <div class="detail-header">
          <div class="detail-breadcrumbs">
            <a href="#" onclick="App.showDashboard(); return false;" style="color: var(--accent-cyan); text-decoration: none;">← Dashboard</a>
            <span>/</span>
            <span>${project.tier}</span>
            <span>/</span>
            <span style="color: #fff;">${String(project.id).padStart(2, '0')}_${project.slug}</span>
          </div>

          <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; flex-wrap: wrap;">
            <div>
              <div style="display: flex; gap: 0.5rem; align-items: center; margin-bottom: 0.5rem;">
                <span class="badge-tier ${project.badgeClass}">${project.tier.toUpperCase()}</span>
                <span style="font-size: 0.8rem; color: var(--accent-cyan); font-weight: 500;">${project.domain}</span>
                <span style="font-size: 0.8rem; color: var(--text-muted); font-family: var(--font-mono);">• ${project.modality}</span>
              </div>
              <h1 class="detail-title">${project.title}</h1>
              <p class="detail-subtitle">${project.problemStatement}</p>
            </div>

            <div style="display: flex; gap: 0.5rem;">
              <a href="../${project.notebookPath}" target="_blank" class="btn-github">
                <span>📓</span> Open Jupyter Notebook
              </a>
            </div>
          </div>
        </div>

        <!-- Tab Navigation -->
        <div class="tabs-nav">
          <button class="tab-btn ${this.currentTab === 'visualizer' ? 'active' : ''}" onclick="ProjectDetailComponent.switchTab(PROJECTS_DATA[${project.id - 1}], 'visualizer')">
            Interactive Visualizer & Simulator
          </button>
          <button class="tab-btn ${this.currentTab === 'algorithms' ? 'active' : ''}" onclick="ProjectDetailComponent.switchTab(PROJECTS_DATA[${project.id - 1}], 'algorithms')">
            Algorithm Deconstructed
          </button>
          <button class="tab-btn ${this.currentTab === 'benchmarks' ? 'active' : ''}" onclick="ProjectDetailComponent.switchTab(PROJECTS_DATA[${project.id - 1}], 'benchmarks')">
            Comparative Benchmarks
          </button>
          <button class="tab-btn ${this.currentTab === 'code' ? 'active' : ''}" onclick="ProjectDetailComponent.switchTab(PROJECTS_DATA[${project.id - 1}], 'code')">
            Notebook Code Extract
          </button>
        </div>

        <!-- Tab Content Body -->
        <div id="tab-content-area">
          ${this.getTabContent(project)}
        </div>
      </div>
    `;

    // Initialize Chart or Animation after DOM paint
    setTimeout(() => {
      this.initVisualizer(project);
    }, 50);
  },

  switchTab(project, tabName) {
    this.currentTab = tabName;
    this.render(project);
  },

  getTabContent(project) {
    switch (this.currentTab) {
      case 'visualizer':
        return this.renderVisualizerTab(project);
      case 'algorithms':
        return this.renderAlgorithmsTab(project);
      case 'benchmarks':
        return this.renderBenchmarksTab(project);
      case 'code':
        return this.renderCodeTab(project);
      default:
        return this.renderVisualizerTab(project);
    }
  },

  // 1. Visualizer & Simulator Tab
  renderVisualizerTab(project) {
    // Determine control panel layout based on project simType
    let controlPanelHtml = '';

    if (project.simType === 'threshold_classification') {
      const cfg = project.simConfig || { defaultThresh: 0.35, posLabel: "Positive", negLabel: "Negative" };
      controlPanelHtml = `
        <div class="sim-control-panel">
          <div class="control-title">Dynamic Decision Threshold Controller</div>
          
          <div class="control-group">
            <div class="control-label">
              <span>Decision Cutoff Threshold (tau):</span>
              <strong class="font-mono" id="thresh-display-val" style="color: var(--accent-cyan);">${cfg.defaultThresh.toFixed(2)}</strong>
            </div>
            <input 
              type="range" 
              class="control-slider" 
              min="0.10" 
              max="0.90" 
              step="0.01" 
              value="${cfg.defaultThresh}"
              oninput="SimController.updateThreshold(PROJECTS_DATA[${project.id - 1}], this.value)"
            />
            <div style="display: flex; justify-content: space-between; font-size: 0.7rem; color: var(--text-muted); font-family: var(--font-mono);">
              <span>0.10 (High Sensitivity)</span>
              <span>0.50 (Default)</span>
              <span>0.90 (High Specificity)</span>
            </div>
          </div>

          <div class="live-metrics-box">
            <div class="live-metric-item">
              <div class="live-metric-label">Sensitivity (Recall)</div>
              <div class="live-metric-val" id="metric-sens" style="color: var(--accent-emerald);">92.4%</div>
            </div>
            <div class="live-metric-item">
              <div class="live-metric-label">Specificity</div>
              <div class="live-metric-val" id="metric-spec" style="color: var(--accent-cyan);">81.2%</div>
            </div>
            <div class="live-metric-item">
              <div class="live-metric-label">Precision</div>
              <div class="live-metric-val" id="metric-prec" style="color: #fff;">81.2%</div>
            </div>
            <div class="live-metric-item">
              <div class="live-metric-label">Asymmetric Cost</div>
              <div class="live-metric-val" id="metric-cost" style="color: var(--accent-rose); font-size: 0.95rem;">$18,400</div>
            </div>
          </div>

          <div>
            <div style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.25rem;">Live Confusion Matrix (N=1,000 holdout cases)</div>
            <table class="cm-table">
              <tr><th></th><th>Pred ${cfg.posLabel}</th><th>Pred ${cfg.negLabel}</th></tr>
              <tr><th>Actual ${cfg.posLabel}</th><td class="cm-val cm-tp" id="cm-tp">355 (TP)</td><td class="cm-val cm-fn" id="cm-fn">29 (FN)</td></tr>
              <tr><th>Actual ${cfg.negLabel}</th><td class="cm-val cm-fp" id="cm-fp">116 (FP)</td><td class="cm-val cm-tn" id="cm-tn">500 (TN)</td></tr>
            </table>
          </div>
        </div>
      `;
    } else if (project.simType === 'pricing_optimizer') {
      controlPanelHtml = `
        <div class="sim-control-panel">
          <div class="control-title">Dynamic Price Optimizer</div>
          <div class="control-group">
            <div class="control-label">
              <span>Offered Unit Price:</span>
              <strong class="font-mono" id="price-display-val" style="color: var(--accent-emerald);">$58.00</strong>
            </div>
            <input 
              type="range" 
              class="control-slider" 
              min="26" 
              max="110" 
              step="1" 
              value="58"
              oninput="SimController.updatePrice(PROJECTS_DATA[${project.id - 1}], this.value)"
            />
          </div>
          <div class="live-metrics-box">
            <div class="live-metric-item">
              <div class="live-metric-label">Est. Units Demanded</div>
              <div class="live-metric-val" id="metric-demand">385 units</div>
            </div>
            <div class="live-metric-item">
              <div class="live-metric-label">Gross Profit ($)</div>
              <div class="live-metric-val" id="metric-profit" style="color: var(--accent-emerald);">$12,705</div>
            </div>
            <div class="live-metric-item">
              <div class="live-metric-label">Gross Margin %</div>
              <div class="live-metric-val" id="metric-margin">56.9%</div>
            </div>
            <div class="live-metric-item">
              <div class="live-metric-label">Price Elasticity</div>
              <div class="live-metric-val" style="color: var(--accent-amber);">-1.75 (Elastic)</div>
            </div>
          </div>
        </div>
      `;
    } else if (project.simType === 'time_series_forecast') {
      controlPanelHtml = `
        <div class="sim-control-panel">
          <div class="control-title">Supply Chain Buffer Tuning</div>
          <div class="control-group">
            <div class="control-label">
              <span>Safety Stock Buffer (Units):</span>
              <strong class="font-mono" id="buffer-display-val" style="color: var(--accent-amber);">38 units</strong>
            </div>
            <input 
              type="range" 
              class="control-slider" 
              min="10" 
              max="80" 
              step="2" 
              value="38"
              oninput="SimController.updateSafetyStock(PROJECTS_DATA[${project.id - 1}], this.value)"
            />
          </div>
          <div class="live-metrics-box">
            <div class="live-metric-item">
              <div class="live-metric-label">Cycle Service Level</div>
              <div class="live-metric-val" id="metric-service-level" style="color: var(--accent-emerald);">98.4%</div>
            </div>
            <div class="live-metric-item">
              <div class="live-metric-label">Monthly Holding Cost</div>
              <div class="live-metric-val" id="metric-holding-cost">$703</div>
            </div>
            <div class="live-metric-item">
              <div class="live-metric-label">Stockout Hazard %</div>
              <div class="live-metric-val" style="color: var(--accent-cyan);">1.6%</div>
            </div>
            <div class="live-metric-item">
              <div class="live-metric-label">WAPE Forecast Error</div>
              <div class="live-metric-val">11.4%</div>
            </div>
          </div>
        </div>
      `;
    } else if (project.simType === 'gnn_network_graph') {
      controlPanelHtml = `
        <div class="sim-control-panel">
          <div class="control-title">Relational Graph Controls</div>
          <p style="font-size: 0.8rem; color: var(--text-secondary);">
            Simulating live force-directed spring physics on 32 interacting accounts with shared device fingerprints.
          </p>
          <button id="gnn-detect-btn" class="btn-github" style="justify-content: center; padding: 0.6rem;">
            Run GNN Community Detection
          </button>
          <div class="live-metrics-box">
            <div class="live-metric-item">
              <div class="live-metric-label">Syndicate Recall</div>
              <div class="live-metric-val" style="color: var(--accent-emerald);">94.6%</div>
            </div>
            <div class="live-metric-item">
              <div class="live-metric-label">False Alarm Rate</div>
              <div class="live-metric-val" style="color: var(--accent-cyan);">0.8%</div>
            </div>
          </div>
        </div>
      `;
    } else {
      controlPanelHtml = `
        <div class="sim-control-panel">
          <div class="control-title">Interactive Workbench</div>
          <p style="font-size: 0.8rem; color: var(--text-secondary);">
            Dynamic algorithmic visualizer demonstrating the underlying logic and validation telemetry.
          </p>
          <div class="live-metrics-box">
            <div class="live-metric-item">
              <div class="live-metric-label">Primary Metric</div>
              <div class="live-metric-val" style="color: var(--accent-emerald); font-size: 0.9rem;">
                ${project.benchmarks.find(b => b.isChamp)?.auc || 'N/A'}
              </div>
            </div>
            <div class="live-metric-item">
              <div class="live-metric-label">Inference Latency</div>
              <div class="live-metric-val" style="color: var(--accent-cyan); font-size: 0.9rem;">
                ${project.benchmarks.find(b => b.isChamp)?.latency || '< 5ms'}
              </div>
            </div>
          </div>
        </div>
      `;
    }

    return `
      <div class="sim-grid">
        <div class="sim-chart-panel">
          <div class="sim-chart-header">
            <div class="sim-chart-title">Real-Time Simulation & Visualizer</div>
            <span style="font-size: 0.75rem; color: var(--text-muted); font-family: var(--font-mono);">INTERACTIVE CANVAS / CHART</span>
          </div>
          <div class="sim-chart-container">
            ${project.simType === 'gnn_network_graph' || project.simType === 'rl_fleet_routing' || project.simType === 'rag_pipeline_flow'
              ? `<canvas id="sim-canvas-animation" style="width: 100%; height: 100%; border-radius: 4px;"></canvas>`
              : `<canvas id="sim-chart-canvas"></canvas>`
            }
          </div>
        </div>
        ${controlPanelHtml}
      </div>
    `;
  },

  // 2. Algorithm Deconstructed Tab
  renderAlgorithmsTab(project) {
    return `
      <div style="display: grid; grid-template-columns: 1fr; gap: 1.25rem;">
        <div class="tech-card">
          <h3>📐 Mathematical Foundations & Formulation</h3>
          <p style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.75rem;">
            Core mathematical relationships driving the model's decision surface and loss functions:
          </p>
          ${project.mathFormulas.map(f => `
            <div class="math-formula-box">${f}</div>
          `).join('')}
        </div>

        <div class="tech-card">
          <h3>⚙️ How the Algorithm Works Under the Hood</h3>
          <ul style="list-style-type: disc; padding-left: 1.5rem; color: var(--text-secondary); font-size: 0.85rem; line-height: 1.6;">
            ${project.howItWorks.map(step => `<li style="margin-bottom: 0.5rem;">${step}</li>`).join('')}
          </ul>
        </div>

        <div class="tech-card">
          <h3>💡 Why This Architecture Was Chosen</h3>
          <p style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.5;">
            ${project.whyThisAlgo}
          </p>
        </div>
      </div>
    `;
  },

  // 3. Benchmarks & Error Cost Tab
  renderBenchmarksTab(project) {
    return `
      <div class="tech-card">
        <h3>📊 Multi-Model Benchmark Comparison</h3>
        <p style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.75rem;">
          Rigorous evaluation across 3+ candidate models using stratified holdout validation splits:
        </p>
        <table class="benchmark-table">
          <thead>
            <tr>
              <th>Model Architecture</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1-Score</th>
              <th>Key Evaluation Metric</th>
              <th>Inference Latency</th>
            </tr>
          </thead>
          <tbody>
            ${project.benchmarks.map(b => `
              <tr class="${b.isChamp ? 'champion-row' : ''}">
                <td>
                  ${b.model}
                  ${b.isChamp ? '<span style="margin-left: 0.5rem; background: var(--accent-cyan-bg); color: var(--accent-cyan); font-size: 0.65rem; padding: 0.1rem 0.4rem; border-radius: 4px; font-family: var(--font-mono);">CHAMPION</span>' : ''}
                </td>
                <td class="font-mono">${b.prec}</td>
                <td class="font-mono">${b.rec}</td>
                <td class="font-mono">${b.f1}</td>
                <td class="font-mono" style="color: ${b.isChamp ? 'var(--accent-emerald)' : 'var(--text-secondary)'}; font-weight: 600;">${b.auc}</td>
                <td class="font-mono">${b.latency}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>

      <div class="tech-card">
        <h3>⚖️ Asymmetric Real-World Error Cost Matrix</h3>
        <p style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.5;">
          In real-world business and clinical environments, the penalty of a <strong>False Negative (FN)</strong> 
          and a <strong>False Positive (FP)</strong> is severely asymmetric. Our models tune the probability cutoff 
          to minimize total financial, clinical, or operational exposure rather than optimizing for naive 50% accuracy.
        </p>
      </div>
    `;
  },

  // 4. Code Extract Tab
  renderCodeTab(project) {
    return `
      <div class="tech-card">
        <h3>🐍 Production Python Code Extract</h3>
        <p style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.75rem;">
          Core implementation snippet extracted from the verified Jupyter notebook:
        </p>
        <div class="code-block-wrapper">
          <div class="code-block-header">
            <span>${project.notebookPath}</span>
            <span>Python 3.14</span>
          </div>
          <pre class="code-pre"><code>${project.codeSnippet}</code></pre>
        </div>
      </div>
    `;
  },

  // Initialize visualizer after tab rendering
  initVisualizer(project) {
    CanvasAnimations.stop();

    if (project.simType === 'threshold_classification') {
      ChartEngine.renderROC('sim-chart-canvas', project.simConfig?.defaultThresh || 0.35);
    } else if (project.simType === 'pricing_optimizer') {
      ChartEngine.renderPricingCurves('sim-chart-canvas', 25.0, -1.75, 6.8);
    } else if (project.simType === 'time_series_forecast') {
      ChartEngine.renderInventoryForecast('sim-chart-canvas', 38);
    } else if (project.simType === 'gnn_network_graph') {
      CanvasAnimations.initGNNGraph('sim-canvas-animation');
    } else if (project.simType === 'rl_fleet_routing') {
      CanvasAnimations.initRLFleet('sim-canvas-animation');
    } else if (project.simType === 'rag_pipeline_flow') {
      CanvasAnimations.initRAGFlow('sim-canvas-animation');
    } else if (project.simType === 'sentiment_volatility_sim') {
      ChartEngine.renderVolatilitySentiment('sim-chart-canvas', -0.45);
    } else if (project.simType === 'sensor_degradation_sim') {
      ChartEngine.renderTurbofanRUL('sim-chart-canvas', 150);
    } else {
      ChartEngine.renderROC('sim-chart-canvas', 0.5);
    }
  }
};
