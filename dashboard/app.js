document.addEventListener('DOMContentLoaded', () => {
  const runBtn = document.getElementById('run-pipeline-btn');
  const promptInput = document.getElementById('prompt-input');
  const logOutput = document.getElementById('log-output');
  const finalCard = document.getElementById('final-output-card');

  runBtn.addEventListener('click', async () => {
    const promptText = promptInput.value.trim();
    if (!promptText) return;

    logOutput.textContent = `[System] Initializing nexus-ai-core pipeline...\n[System] Mission Objective: "${promptText}"\n`;
    finalCard.innerHTML = `<span class="placeholder-text">Executing autonomous agents...</span>`;

    runBtn.disabled = true;
    runBtn.textContent = '⏳ Agent Pipeline Running...';

    // Simulate Agent Step 1
    await sleep(800);
    logOutput.textContent += `\n[Agent: ResearchAgent] Querying web search tools...`;
    logOutput.textContent += `\n[Tool: web_search] Output -> Found 2 relevant papers on AI modular workflows.`;

    // Simulate Agent Step 2
    await sleep(1000);
    logOutput.textContent += `\n\n[Agent: WriterAgent] Synthesizing findings into structured Markdown documentation...`;
    logOutput.textContent += `\n[Workflow: Finished] Pipeline completed successfully in 1.8s.`;

    finalCard.innerHTML = `
      <h3># Executive Summary</h3>
      <p>Modern autonomous AI agent systems rely on decoupled tool registries, sliding-window conversation memory, and flexible multi-agent pipelines.</p>
      <hr style="margin: 1rem 0; border: 0; border-top: 1px solid var(--border-color);" />
      <h4>Key Architectural Pillars:</h4>
      <ul>
        <li><strong>Tool Registry:</strong> Safe function execution & parameter schemas.</li>
        <li><strong>State Memory:</strong> Contextual message retention across turns.</li>
        <li><strong>Workflow Orchestrator:</strong> Sequential & Parallel execution graphs.</li>
      </ul>
    `;

    runBtn.disabled = false;
    runBtn.textContent = '⚡ Execute Agent Workflow';
  });

  function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
});
