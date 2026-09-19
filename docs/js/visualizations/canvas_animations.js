/**
 * High-Performance HTML5 Canvas Interactive Animations
 * Handles GNN Network Graph Physics, RL Dynamic Vehicle Routing & RAG Pipelines
 */

const CanvasAnimations = {
  currentAnimId: null,

  stop() {
    if (this.currentAnimId) {
      cancelAnimationFrame(this.currentAnimId);
      this.currentAnimId = null;
    }
  },

  // 1. Force-Directed Graph Physics for Project 15 (GNN Fraud Rings)
  initGNNGraph(canvasId) {
    this.stop();
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    // Resize canvas properly
    canvas.width = canvas.parentElement.clientWidth || 600;
    canvas.height = 360;

    const nNodes = 32;
    const nodes = [];
    for (let i = 0; i < nNodes; i++) {
      const isSyndicate = i >= 8 && i <= 16;
      nodes.push({
        id: i,
        x: canvas.width / 2 + (Math.random() - 0.5) * 200,
        y: canvas.height / 2 + (Math.random() - 0.5) * 150,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
        radius: isSyndicate ? 7 : 5,
        isSyndicate: isSyndicate,
        label: isSyndicate ? `FRAUD-${i}` : `USR-${i}`
      });
    }

    // Edges
    const edges = [];
    // Dense connections among syndicate
    for (let i = 8; i <= 16; i++) {
      for (let j = i + 1; j <= 16; j++) {
        if (Math.random() < 0.65) edges.push([i, j, true]);
      }
    }
    // Sparse connections among benign
    for (let i = 0; i < 25; i++) {
      const a = Math.floor(Math.random() * nNodes);
      const b = Math.floor(Math.random() * nNodes);
      if (a !== b && !(a >= 8 && a <= 16 && b >= 8 && b <= 16)) {
        edges.push([a, b, false]);
      }
    }

    let detected = false;
    const btn = document.getElementById('gnn-detect-btn');
    if (btn) {
      btn.onclick = () => {
        detected = !detected;
        btn.innerText = detected ? "Reset Network View" : "Run GNN Community Detection";
        btn.className = detected ? "btn-github" : "btn-github";
      };
    }

    const animate = () => {
      ctx.fillStyle = '#09090b';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Simple repulsion physics
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[j].x - nodes[i].x;
          const dy = nodes[j].y - nodes[i].y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          if (dist < 100) {
            const force = (100 - dist) / 100 * 0.05;
            nodes[i].vx -= (dx / dist) * force;
            nodes[i].vy -= (dy / dist) * force;
            nodes[j].vx += (dx / dist) * force;
            nodes[j].vy += (dy / dist) * force;
          }
        }
      }

      // Spring attraction along edges
      for (const [a, b, isSyn] of edges) {
        const dx = nodes[b].x - nodes[a].x;
        const dy = nodes[b].y - nodes[a].y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const targetDist = isSyn ? 35 : 75;
        const force = (dist - targetDist) * 0.008;
        nodes[a].vx += (dx / dist) * force;
        nodes[a].vy += (dy / dist) * force;
        nodes[b].vx -= (dx / dist) * force;
        nodes[b].vy -= (dy / dist) * force;
      }

      // Draw Edges
      for (const [a, b, isSyn] of edges) {
        ctx.beginPath();
        ctx.moveTo(nodes[a].x, nodes[a].y);
        ctx.lineTo(nodes[b].x, nodes[b].y);
        if (detected && isSyn) {
          ctx.strokeStyle = 'rgba(244, 63, 94, 0.8)';
          ctx.lineWidth = 2;
        } else {
          ctx.strokeStyle = 'rgba(39, 39, 42, 0.5)';
          ctx.lineWidth = 1;
        }
        ctx.stroke();
      }

      // Draw Nodes
      for (const node of nodes) {
        // Friction & bounds
        node.vx *= 0.94;
        node.vy *= 0.94;
        node.x += node.vx;
        node.y += node.vy;

        // Boundary containment
        node.x = Math.max(20, Math.min(canvas.width - 20, node.x));
        node.y = Math.max(20, Math.min(canvas.height - 20, node.y));

        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        if (detected) {
          ctx.fillStyle = node.isSyndicate ? '#f43f5e' : '#3f3f46';
        } else {
          ctx.fillStyle = '#06b6d4';
        }
        ctx.fill();
        ctx.strokeStyle = '#18181b';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      // Status text
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.fillStyle = detected ? '#f43f5e' : '#71717a';
      ctx.fillText(detected ? "SYNTHETIC FRAUD RING DETECTED (High Density Cluster)" : "Heterogeneous Transaction Graph (Raw State)", 15, 25);

      CanvasAnimations.currentAnimId = requestAnimationFrame(animate);
    };

    animate();
  },

  // 2. Dynamic Vehicle Routing Animation for Project 19 (RL Fleet Dispatch)
  initRLFleet(canvasId) {
    this.stop();
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.parentElement.clientWidth || 600;
    canvas.height = 360;

    const depot = { x: canvas.width / 2, y: canvas.height / 2 };
    const dropoffs = [
      { x: 100, y: 80, done: false },
      { x: 180, y: 280, done: false },
      { x: 420, y: 70, done: false },
      { x: 500, y: 260, done: false },
      { x: 300, y: 60, done: false },
      { x: 120, y: 190, done: false }
    ];

    const vehicle = { x: depot.x, y: depot.y, targetIdx: 0, speed: 1.8 };

    const animate = () => {
      ctx.fillStyle = '#09090b';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Grid Lines
      ctx.strokeStyle = 'rgba(39, 39, 42, 0.3)';
      ctx.lineWidth = 1;
      for (let x = 0; x < canvas.width; x += 40) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
      }
      for (let y = 0; y < canvas.height; y += 40) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
      }

      // Draw Depot
      ctx.fillStyle = '#6366f1';
      ctx.fillRect(depot.x - 10, depot.y - 10, 20, 20);
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.fillStyle = '#a1a1aa';
      ctx.fillText("DEPOT", depot.x - 16, depot.y + 24);

      // Draw Dropoff Targets
      dropoffs.forEach((d, idx) => {
        ctx.beginPath();
        ctx.arc(d.x, d.y, 6, 0, Math.PI * 2);
        ctx.fillStyle = d.done ? '#10b981' : '#f59e0b';
        ctx.fill();
        ctx.strokeStyle = '#18181b';
        ctx.stroke();
        ctx.fillStyle = '#71717a';
        ctx.fillText(`P-${idx+1}`, d.x - 8, d.y - 10);
      });

      // Move Vehicle towards target
      const target = dropoffs[vehicle.targetIdx];
      if (target) {
        const dx = target.x - vehicle.x;
        const dy = target.y - vehicle.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        // Path trail
        ctx.beginPath();
        ctx.moveTo(vehicle.x, vehicle.y);
        ctx.lineTo(target.x, target.y);
        ctx.strokeStyle = 'rgba(6, 182, 212, 0.4)';
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.setLineDash([]);

        if (dist < 4) {
          target.done = true;
          vehicle.targetIdx = (vehicle.targetIdx + 1) % dropoffs.length;
          if (vehicle.targetIdx === 0) {
            dropoffs.forEach(d => d.done = false);
          }
        } else {
          vehicle.x += (dx / dist) * vehicle.speed;
          vehicle.y += (dy / dist) * vehicle.speed;
        }
      }

      // Draw Vehicle
      ctx.beginPath();
      ctx.arc(vehicle.x, vehicle.y, 8, 0, Math.PI * 2);
      ctx.fillStyle = '#06b6d4';
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.fillStyle = '#10b981';
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.fillText("DQN Agent: Real-Time Dynamic Fleet Dispatching (Active Route)", 15, 25);

      CanvasAnimations.currentAnimId = requestAnimationFrame(animate);
    };

    animate();
  },

  // 3. RAG Retrieval Pipeline Flow Animation for Project 16
  initRAGFlow(canvasId) {
    this.stop();
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.parentElement.clientWidth || 600;
    canvas.height = 360;

    let step = 0;
    const animate = () => {
      ctx.fillStyle = '#09090b';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      step = (step + 0.5) % 360;

      // Draw Stages: Query -> Dense/BM25 Index -> Cross-Encoder -> Hallucination Auditor
      const stages = [
        { label: "1. User Query", x: 60, y: 180, color: "#6366f1" },
        { label: "2. Hybrid Search", x: 200, y: 180, color: "#06b6d4" },
        { label: "3. Re-Ranker", x: 350, y: 180, color: "#10b981" },
        { label: "4. Fact Auditor", x: 500, y: 180, color: "#f59e0b" }
      ];

      // Draw Connector Lines
      for (let i = 0; i < stages.length - 1; i++) {
        ctx.beginPath();
        ctx.moveTo(stages[i].x + 35, stages[i].y);
        ctx.lineTo(stages[i + 1].x - 35, stages[i + 1].y);
        ctx.strokeStyle = '#27272a';
        ctx.lineWidth = 3;
        ctx.stroke();

        // Pulsing token
        const pulseX = stages[i].x + 35 + ((step * 2) % 100) / 100 * (stages[i + 1].x - stages[i].x - 70);
        ctx.beginPath();
        ctx.arc(pulseX, stages[i].y, 4, 0, Math.PI * 2);
        ctx.fillStyle = '#fff';
        ctx.fill();
      }

      // Draw stage boxes
      stages.forEach(s => {
        ctx.fillStyle = '#18181b';
        ctx.strokeStyle = s.color;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.roundRect(s.x - 45, s.y - 30, 90, 60, 8);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = '#fff';
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText(s.label, s.x, s.y + 4);
      });

      ctx.textAlign = 'left';
      ctx.fillStyle = '#10b981';
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.fillText("Automated SEC 10-K RAG Verification: Faithfulness Score 0.978 (Zero Hallucination)", 15, 25);

      CanvasAnimations.currentAnimId = requestAnimationFrame(animate);
    };

    animate();
  }
};
