/**
 * Provider Adapters for LLM API integration.
 */

export class BaseProvider {
  constructor(options = {}) {
    this.options = options;
  }

  async generateResponse(messages, tools = []) {
    throw new Error('generateResponse must be implemented by provider subclass.');
  }
}

/**
 * Deterministic / Simulated Mock Provider for testing and local demos.
 */
export class MockProvider extends BaseProvider {
  /**
   * @param {Object} [options]
   * @param {Array<Object>} [options.mockResponses] - Custom queued responses
   */
  constructor(options = {}) {
    super(options);
    this.mockResponses = options.mockResponses || [];
    this.responseIndex = 0;
  }

  addMockResponse(response) {
    this.mockResponses.push(response);
  }

  async generateResponse(messages, tools = []) {
    if (this.responseIndex < this.mockResponses.length) {
      const resp = this.mockResponses[this.responseIndex++];
      return typeof resp === 'function' ? await resp(messages, tools) : resp;
    }

    const lastMessage = messages[messages.length - 1];
    const userText = lastMessage?.content || '';

    // Auto-detect tool call intent in mock mode
    if (tools.length > 0 && userText.toLowerCase().includes('search')) {
      const searchTool = tools.find(t => t.name.includes('search'));
      if (searchTool) {
        return {
          content: 'I need to use the search tool to find the answer.',
          toolCalls: [
            {
              id: 'call_' + Math.random().toString(36).substring(2, 9),
              name: searchTool.name,
              args: { query: userText }
            }
          ]
        };
      }
    }

    return {
      content: `[Mock AI Response] Processed prompt: "${userText.substring(0, 80)}"`,
      toolCalls: []
    };
  }
}

/**
 * OpenAI-compatible HTTP Provider
 */
export class OpenAIProvider extends BaseProvider {
  constructor(options = {}) {
    super(options);
    this.apiKey = options.apiKey || process.env.OPENAI_API_KEY;
    this.baseUrl = options.baseUrl || 'https://api.openai.com/v1';
    this.model = options.model || 'gpt-4o-mini';
  }

  async generateResponse(messages, tools = []) {
    if (!this.apiKey) {
      throw new Error('OpenAI API key is missing.');
    }

    const payload = {
      model: this.model,
      messages: messages.map(m => ({ role: m.role, content: m.content })),
    };

    if (tools.length > 0) {
      payload.tools = tools.map(t => ({
        type: 'function',
        function: {
          name: t.name,
          description: t.description,
          parameters: t.parameters
        }
      }));
    }

    const res = await fetch(`${this.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`
      },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const err = await res.text();
      throw new Error(`OpenAI API error (${res.status}): ${err}`);
    }

    const data = await res.json();
    const choice = data.choices[0].message;

    const toolCalls = choice.tool_calls ? choice.tool_calls.map(tc => ({
      id: tc.id,
      name: tc.function.name,
      args: JSON.parse(tc.function.arguments || '{}')
    })) : [];

    return {
      content: choice.content || '',
      toolCalls
    };
  }
}
