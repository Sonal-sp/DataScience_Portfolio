/**
 * Master Dashboard Overview Component
 * Renders executive metrics, domain filter pills, and responsive project grid
 */

const DashboardComponent = {
  currentDomainFilter: "All",

  render(projects) {
    const container = document.getElementById('viewport-container');
    if (!container) return;

    // Filter by domain
    const filtered = projects.filter(p => {
      if (this.currentDomainFilter === "All") return true;
      return p.domain.toLowerCase().includes(this.currentDomainFilter.toLowerCase());
    });

    const domains = ["All", "Healthcare", "Fintech", "E-Commerce", "Logistics", "Energy", "Cybersecurity", "Vision", "NLP"];

    container.innerHTML = `
      <div class="workspace-viewport">
        <!-- Hero Header -->
        <div style="margin-bottom: 2rem;">
          <div class="brand-badge">PORTFOLIO WORKBENCH</div>
          <h1 style="font-size: 2rem; font-weight: 700; color: #fff; letter-spacing: -0.025em; margin-bottom: 0.5rem;">
            Data Science & Machine Learning Engineering Catalog
          </h1>
          <p style="color: var(--text-secondary); max-width: 800px; font-size: 0.95rem;">
            Explore 20 production-grade data science systems. Every project features self-contained datasets, 
            mathematical deconstructions, interactive logic simulators, and multi-model benchmark comparisons.
          </p>
        </div>

        <!-- Metric Stat Strip -->
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem;">
          <div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); padding: 1.25rem; border-radius: 8px;">
            <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase;">Total Projects</div>
            <div style="font-size: 1.75rem; font-weight: 700; color: #fff; font-family: var(--font-mono);">20</div>
            <div style="font-size: 0.75rem; color: var(--accent-emerald);">Beginner • Intermediate • Advanced</div>
          </div>
          <div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); padding: 1.25rem; border-radius: 8px;">
            <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase;">Industry Verticals</div>
            <div style="font-size: 1.75rem; font-weight: 700; color: #fff; font-family: var(--font-mono);">10+</div>
            <div style="font-size: 0.75rem; color: var(--accent-cyan);">Healthcare, Fintech, Vision, NLP, IoT</div>
          </div>
          <div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); padding: 1.25rem; border-radius: 8px;">
            <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase;">Verification Suite</div>
            <div style="font-size: 1.75rem; font-weight: 700; color: var(--accent-emerald); font-family: var(--font-mono);">100%</div>
            <div style="font-size: 0.75rem; color: var(--text-secondary);">Automated JSON schema & data checks</div>
          </div>
          <div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); padding: 1.25rem; border-radius: 8px;">
            <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase;">Notebook Integration</div>
            <div style="font-size: 1.75rem; font-weight: 700; color: var(--accent-amber); font-family: var(--font-mono);">20 .ipynb</div>
            <div style="font-size: 0.75rem; color: var(--text-secondary);">End-to-end runnable in Jupyter</div>
          </div>
        </div>

        <!-- Domain Filter Chips -->
        <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1.5rem; align-items: center;">
          <span style="font-size: 0.8rem; color: var(--text-muted); margin-right: 0.5rem;">Filter Domain:</span>
          ${domains.map(d => `
            <button 
              onclick="DashboardComponent.setFilter('${d}')"
              style="
                background: ${this.currentDomainFilter === d ? 'var(--text-primary)' : 'var(--bg-surface-elevated)'};
                color: ${this.currentDomainFilter === d ? 'var(--bg-base)' : 'var(--text-secondary)'};
                border: 1px solid var(--border-subtle);
                padding: 0.35rem 0.75rem;
                border-radius: 6px;
                font-size: 0.8rem;
                cursor: pointer;
                transition: all 0.15s;
                font-weight: ${this.currentDomainFilter === d ? '600' : '400'};
              "
            >
              ${d}
            </button>
          `).join('')}
        </div>

        <!-- Project Cards Grid -->
        <div class="dashboard-grid">
          ${filtered.map(p => `
            <div class="project-card" onclick="App.selectProject(${p.id})">
              <div class="project-card-header">
                <span class="badge-tier ${p.badgeClass}">${p.tier.toUpperCase()}</span>
                <span style="font-size: 0.75rem; color: var(--text-muted); font-family: var(--font-mono);">${String(p.id).padStart(2, '0')}</span>
              </div>
              <div class="project-card-title">${p.title}</div>
              <div style="font-size: 0.75rem; color: var(--accent-cyan); margin-bottom: 0.5rem; font-weight: 500;">
                ${p.domain}
              </div>
              <div class="project-card-desc">
                ${p.description}
              </div>
              <div style="display: flex; gap: 0.4rem; flex-wrap: wrap; margin-bottom: 1rem;">
                ${p.algorithms.slice(0, 2).map(a => `
                  <span style="background: var(--bg-surface-elevated); border: 1px solid var(--border-subtle); color: var(--text-secondary); font-size: 0.7rem; padding: 0.15rem 0.4rem; border-radius: 4px; font-family: var(--font-mono);">
                    ${a}
                  </span>
                `).join('')}
              </div>
              <div class="project-card-footer">
                <span>Modality: <strong>${p.modality.split(' ')[0]}</strong></span>
                <span style="color: var(--accent-emerald); font-weight: 500;">Launch Simulation →</span>
              </div>
            </div>
          `).join('')}
        </div>
      </div>
    `;
  },

  setFilter(domain) {
    this.currentDomainFilter = domain;
    this.render(PROJECTS_DATA);
  }
};
