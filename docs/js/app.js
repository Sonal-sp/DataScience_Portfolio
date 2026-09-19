/**
 * Application State Manager & Single-Page Router
 * Coordinates Sidebar, Dashboard, Project Detail view, URL hashes & Mobile Drawer
 */

const App = {
  currentProjectId: null,
  searchQuery: "",

  init() {
    console.log("Initializing Data Science Portfolio Visualizer...");

    // Setup search listener
    const searchInput = document.getElementById('search-input');
    if (searchInput) {
      searchInput.addEventListener('input', (e) => {
        this.searchQuery = e.target.value.trim();
        SidebarComponent.render(PROJECTS_DATA, this.currentProjectId, this.searchQuery);
      });
    }

    // Handle initial routing from URL hash (e.g. #project-5)
    this.handleHashRoute();
    window.addEventListener('hashchange', () => this.handleHashRoute());

    // Render initial sidebar
    SidebarComponent.render(PROJECTS_DATA, this.currentProjectId);
  },

  handleHashRoute() {
    const hash = window.location.hash;
    if (hash.startsWith('#project-')) {
      const id = parseInt(hash.replace('#project-', ''), 10);
      const project = PROJECTS_DATA.find(p => p.id === id);
      if (project) {
        this.selectProject(id, false);
        return;
      }
    }
    this.showDashboard(false);
  },

  selectProject(id, updateHash = true) {
    this.currentProjectId = id;
    const project = PROJECTS_DATA.find(p => p.id === id);
    if (!project) return;

    if (updateHash) {
      window.location.hash = `#project-${id}`;
    }

    // Render Detail View
    ProjectDetailComponent.render(project);
    SidebarComponent.render(PROJECTS_DATA, id, this.searchQuery);

    // Scroll to top
    const mainArea = document.getElementById('main-content-scroll');
    if (mainArea) mainArea.scrollTop = 0;

    // Close mobile menu if open
    this.closeMobileMenu();
  },

  showDashboard(updateHash = true) {
    this.currentProjectId = null;
    if (updateHash) {
      window.location.hash = '#dashboard';
    }

    DashboardComponent.render(PROJECTS_DATA);
    SidebarComponent.render(PROJECTS_DATA, null, this.searchQuery);

    const mainArea = document.getElementById('main-content-scroll');
    if (mainArea) mainArea.scrollTop = 0;

    this.closeMobileMenu();
  },

  toggleMobileMenu() {
    const sidebar = document.getElementById('sidebar');
    if (sidebar) {
      sidebar.classList.toggle('open');
    }
  },

  closeMobileMenu() {
    const sidebar = document.getElementById('sidebar');
    if (sidebar && sidebar.classList.contains('open')) {
      sidebar.classList.remove('open');
    }
  }
};

// Bootstrap on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
  App.init();
});
