document.addEventListener('DOMContentLoaded', async () => {
  const engineEl = document.getElementById('engine-status');
  const agentEl = document.getElementById('agent-count');

  try {
    const res = await fetch('http://localhost:3000/api/health');
    if (res.ok) {
      engineEl.innerText = 'Connected (Port 3000)';
      engineEl.style.color = '#34d399';

      const agentRes = await fetch('http://localhost:3000/api/detect');
      if (agentRes.ok) {
        const agentData = await agentRes.json();
        agentEl.innerText = `${agentData.total_agents_detected} Active`;
      }
    } else {
      engineEl.innerText = 'Offline (Heuristic Mode)';
      engineEl.style.color = '#fbbf24';
    }
  } catch (e) {
    engineEl.innerText = 'Standby (Offline)';
    engineEl.style.color = '#94a3b8';
  }
});
