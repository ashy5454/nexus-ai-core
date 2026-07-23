/**
 * Tool representation in nexus-ai-core
 */
export class Tool {
  /**
   * @param {Object} config
   * @param {string} config.name - Unique identifier for the tool
   * @param {string} config.description - Clear instructions on when to use this tool
   * @param {Function} config.execute - Async or sync execution function
   * @param {Object} [config.parameters] - Expected JSON schema or parameter definitions
   */
  constructor({ name, description, execute, parameters = {} }) {
    if (!name || typeof name !== 'string') {
      throw new Error('Tool must have a valid string name.');
    }
    if (!description || typeof description !== 'string') {
      throw new Error('Tool must have a valid description.');
    }
    if (typeof execute !== 'function') {
      throw new Error('Tool execution handler must be a function.');
    }

    this.name = name;
    this.description = description;
    this.execute = execute;
    this.parameters = parameters;
    this.callCount = 0;
  }

  /**
   * Executes tool with arguments and records invocation metrics.
   * @param {Object} args 
   * @returns {Promise<any>}
   */
  async run(args = {}) {
    this.callCount++;
    try {
      const result = await this.execute(args);
      return {
        status: 'success',
        toolName: this.name,
        output: result,
        timestamp: new Date().toISOString()
      };
    } catch (err) {
      return {
        status: 'error',
        toolName: this.name,
        error: err.message || String(err),
        timestamp: new Date().toISOString()
      };
    }
  }

  /**
   * Formats tool definition for LLM prompt context.
   */
  toJSON() {
    return {
      name: this.name,
      description: this.description,
      parameters: this.parameters
    };
  }
}

/**
 * Registry managing available tools for agents.
 */
export class ToolRegistry {
  constructor() {
    this.tools = new Map();
  }

  register(tool) {
    if (!(tool instanceof Tool)) {
      throw new Error('Can only register instances of Tool class.');
    }
    this.tools.set(tool.name, tool);
  }

  get(name) {
    return this.tools.get(name);
  }

  has(name) {
    return this.tools.has(name);
  }

  list() {
    return Array.from(this.tools.values()).map(t => t.toJSON());
  }

  async executeTool(name, args) {
    const tool = this.get(name);
    if (!tool) {
      throw new Error(`Tool '${name}' is not registered in registry.`);
    }
    return await tool.run(args);
  }
}
