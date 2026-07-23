/**
 * Multi-Agent Workflow Orchestrator for nexus-ai-core.
 */
export class Workflow {
  /**
   * @param {Object} config
   * @param {string} config.name - Name of workflow pipeline
   * @param {string} [config.description=''] - Purpose of pipeline
   */
  constructor({ name, description = '' }) {
    this.name = name;
    this.description = description;
    this.steps = [];
  }

  /**
   * Adds a sequential agent step to the workflow.
   * @param {Agent} agent 
   * @param {Function} [transformInput] - Transform preceding output to next agent prompt
   */
  addSequentialStep(agent, transformInput = (input) => input) {
    this.steps.push({
      type: 'sequential',
      agent,
      transformInput
    });
    return this;
  }

  /**
   * Adds parallel agent execution step.
   * @param {Array<Agent>} agents 
   */
  addParallelStep(agents) {
    this.steps.push({
      type: 'parallel',
      agents
    });
    return this;
  }

  /**
   * Runs the workflow pipeline with an initial input.
   * @param {string} initialInput 
   * @returns {Promise<Object>} Execution report
   */
  async execute(initialInput) {
    let currentData = initialInput;
    const executionLogs = [];

    for (let i = 0; i < this.steps.length; i++) {
      const step = this.steps[i];
      const stepIndex = i + 1;

      if (step.type === 'sequential') {
        const prompt = step.transformInput(currentData);
        const result = await step.agent.run(prompt);
        currentData = result.finalResponse;
        executionLogs.push({
          step: stepIndex,
          type: 'sequential',
          agent: step.agent.name,
          inputPrompt: prompt,
          output: result
        });
      } else if (step.type === 'parallel') {
        const results = await Promise.all(
          step.agents.map(a => a.run(currentData))
        );
        currentData = results.map(r => `[${r.agent}]: ${r.finalResponse}`).join('\n\n');
        executionLogs.push({
          step: stepIndex,
          type: 'parallel',
          agents: step.agents.map(a => a.name),
          outputs: results
        });
      }
    }

    return {
      workflowName: this.name,
      completedAt: new Date().toISOString(),
      finalOutput: currentData,
      logs: executionLogs
    };
  }
}
