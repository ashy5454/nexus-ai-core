import { Memory } from './memory.js';
import { ToolRegistry } from './tool.js';
import { MockProvider } from './providers.js';

/**
 * Autonomous AI Agent runner in nexus-ai-core.
 */
export class Agent {
  /**
   * @param {Object} config
   * @param {string} config.name - Agent name
   * @param {string} [config.role='AI Assistant'] - Agent role description
   * @param {string} [config.instructions=''] - Base system instructions
   * @param {BaseProvider} [config.provider] - LLM provider adapter
   * @param {ToolRegistry} [config.tools] - Tool registry instance
   * @param {Memory} [config.memory] - Conversation memory
   * @param {number} [config.maxSteps=5] - Maximum execution loops per turn
   */
  constructor({
    name,
    role = 'AI Assistant',
    instructions = '',
    provider = new MockProvider(),
    tools = new ToolRegistry(),
    memory = null,
    maxSteps = 5
  }) {
    if (!name) throw new Error('Agent name is required.');

    this.name = name;
    this.role = role;
    this.instructions = instructions;
    this.provider = provider;
    this.tools = tools;
    this.maxSteps = maxSteps;
    this.listeners = new Map();

    const basePrompt = `You are ${name}, a ${role}.\n${instructions}`;
    this.memory = memory || new Memory({ systemPrompt: basePrompt });
    if (!memory && instructions) {
      this.memory.setSystemPrompt(basePrompt);
    }
  }

  on(event, handler) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(handler);
  }

  emit(event, data) {
    const handlers = this.listeners.get(event) || [];
    for (const fn of handlers) {
      fn(data);
    }
  }

  registerTool(tool) {
    this.tools.register(tool);
  }

  /**
   * Executes a prompt through the agent reasoning and tool loop.
   * @param {string} userPrompt 
   * @returns {Promise<Object>} Execution summary
   */
  async run(userPrompt) {
    this.memory.addUserMessage(userPrompt);
    let stepsTaken = 0;
    const historySteps = [];

    while (stepsTaken < this.maxSteps) {
      stepsTaken++;
      const messages = this.memory.getFormattedHistory();
      const availableTools = this.tools.list();

      this.emit('step:start', { step: stepsTaken, agent: this.name });

      const response = await this.provider.generateResponse(messages, availableTools);

      if (response.content) {
        this.memory.addAssistantMessage(response.content);
      }

      historySteps.push({
        step: stepsTaken,
        response
      });

      this.emit('step:finish', { step: stepsTaken, response });

      // If agent initiated tool calls, execute them
      if (response.toolCalls && response.toolCalls.length > 0) {
        for (const call of response.toolCalls) {
          this.emit('tool:start', { toolName: call.name, args: call.args });
          const toolResult = await this.tools.executeTool(call.name, call.args);
          this.memory.addToolMessage(call.name, toolResult);
          this.emit('tool:finish', { toolName: call.name, result: toolResult });
        }
      } else {
        // Agent finished turn (no more tool calls)
        break;
      }
    }

    const finalOutput = {
      agent: this.name,
      stepsTaken,
      finalResponse: this.memory.messages[this.memory.messages.length - 1]?.content || '',
      history: historySteps
    };

    this.emit('finish', finalOutput);
    return finalOutput;
  }
}
