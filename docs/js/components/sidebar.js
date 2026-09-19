/**
 * Sidebar Navigation Component
 * Renders categorized project list with real-time multi-attribute search
 */

const SidebarComponent = {
  render(projects, activeId = null, searchQuery = "") {
    const container = document.getElementById('project-nav-list');
    if (!container) return;

    // Filter projects based on search query
    const filtered = projects.filter(p => {
      if (!searchQuery) return true;
      const q = searchQuery.toLowerCase();
      return (
        p.title.toLowerCase().includes(q) ||
        p.domain.toLowerCase().includes(q) ||
        p.tier.toLowerCase().includes(q) ||
        p.algorithms.some(a => a.toLowerCase().includes(q))
      );
    });

    const tiers = [
      { name: "🟢 Level 1: Beginner", filter: "Beginner" },
      { name: "🟡 Level 2: Intermediate", filter: "Intermediate" },
      { name: "🔴 Level 3: Advanced", filter: "Advanced" }
    ];

    let html = "";

    tiers.forEach(tier => {
      const tierProjects = filtered.filter(p => p.tier === tier.filter);
      if (tierProjects.length === 0) return;

      html += `
        <div class="nav-tier-group">
          <div class="nav-tier-title">
            <span>${tier.name}</span>
            <span class="font-mono">${tierProjects.length}</span>
          </div>
          ${tierProjects.map(p => `
            <div class="nav-item ${p.id === activeId ? 'active' : ''}" onclick="App.selectProject(${p.id})">
              <span class="nav-item-num">${String(p.id).padStart(2, '0')}</span>
              <div class="nav-item-content">
                <div class="nav-item-title">${p.title}</div>
                <div class="nav-item-domain">${p.domain.split('&')[0]}</div>
              </div>
            </div>
          `).join('')}
        </div>
      `;
    });

    if (filtered.length === 0) {
      html = `<div style="padding: 1.5rem; text-align: center; color: #71717a; font-size: 0.85rem;">No projects found matching "${searchQuery}"</div>`;
    }

    container.innerHTML = html;
  }
};
