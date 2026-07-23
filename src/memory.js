/**
 * Conversation memory and state manager for nexus-ai-core agents.
 */
export class Memory {
  /**
   * @param {Object} [options]
   * @param {number} [options.maxMessages=50] - Maximum messages before window trims
   * @param {string} [options.systemPrompt=''] - Base system instructions
   */
  constructor({ maxMessages = 50, systemPrompt = '' } = {}) {
    this.maxMessages = maxMessages;
    this.systemPrompt = systemPrompt;
    this.messages = [];
    this.state = new Map();
  }

  setSystemPrompt(prompt) {
    this.systemPrompt = prompt;
  }

  addMessage(role, content, metadata = {}) {
    const msg = {
      role,
      content,
      metadata,
      timestamp: new Date().toISOString()
    };
    this.messages.push(msg);
    this._trim();
    return msg;
  }

  addUserMessage(content, metadata) {
    return this.addMessage('user', content, metadata);
  }

  addAssistantMessage(content, metadata) {
    return this.addMessage('assistant', content, metadata);
  }

  addToolMessage(toolName, output, metadata = {}) {
    return this.addMessage('tool', typeof output === 'string' ? output : JSON.stringify(output), {
      toolName,
      ...metadata
    });
  }

  /**
   * Trims message buffer keeping recent messages within maxMessages limit.
   * @private
   */
  _trim() {
    if (this.messages.length > this.maxMessages) {
      this.messages = this.messages.slice(this.messages.length - this.maxMessages);
    }
  }

  getFormattedHistory() {
    const history = [];
    if (this.systemPrompt) {
      history.push({ role: 'system', content: this.systemPrompt });
    }
    return history.concat(this.messages);
  }

  setState(key, value) {
    this.state.set(key, value);
  }

  getState(key) {
    return this.state.get(key);
  }

  clear() {
    this.messages = [];
    this.state.clear();
  }
}
