# ⚡ nexus-ai-core

> A lightweight, modular JavaScript/TypeScript framework for building autonomous AI agents, tool-calling pipelines, sliding-window memory management, and multi-agent workflow orchestrations.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Node.js CI](https://img.shields.io/badge/node-%3E%3D%2018.0.0-blue.svg)](https://nodejs.org/)
[![GitHub](https://img.shields.io/badge/GitHub-ashy5454%2Fnexus--ai--core-brightgreen)](https://github.com/ashy5454/nexus-ai-core)

---

## ✨ Features

- 🤖 **Autonomous Agent Loops**: State-driven execution loops with prompt composition and event system.
- 🛠️ **Tool Registry**: Type-checked tool definitions, execution wrappers, and schema exporters.
- 🧠 **Context & Memory Manager**: Sliding-window conversation memory with role management (`system`, `user`, `assistant`, `tool`).
- 🔀 **Multi-Agent Workflows**: Compose complex pipelines with **Sequential** steps and **Parallel** execution branches.
- 🔌 **Provider Adapters**: Out-of-the-box support for `OpenAIProvider`, `MockProvider` (offline testing), and extensible custom endpoints.
- 💻 **Interactive Playground**: Built-in dark-mode web dashboard visualizer for pipeline inspection.

---

## 🚀 Quickstart

### Installation

```bash
git clone https://github.com/ashy5454/nexus-ai-core.git
cd nexus-ai-core
npm install
```

### Basic Agent Example

```js
import { Agent, Tool, MockProvider } from './src/index.js';

// 1. Define custom tool
const calculator = new Tool({
  name: 'calculator',
  description: 'Adds two numbers together',
  execute: async ({ a, b }) => a + b
});

// 2. Instantiate Agent
const agent = new Agent({
  name: 'MathBot',
  role: 'Mathematical Assistant',
  provider: new MockProvider()
});

agent.registerTool(calculator);

// 3. Execute prompt
const response = await agent.run('Calculate 5 + 10');
console.log(response.finalResponse);
```

### Multi-Agent Workflow Example

```js
import { Agent, Workflow, MockProvider } from './src/index.js';

const researcher = new Agent({ name: 'Researcher', provider: new MockProvider() });
const writer = new Agent({ name: 'Writer', provider: new MockProvider() });

const pipeline = new Workflow({ name: 'ResearchPipeline' });
pipeline
  .addSequentialStep(researcher)
  .addSequentialStep(writer, (res) => `Draft article based on: ${res}`);

const result = await pipeline.execute('Analyze AI trends');
console.log(result.finalOutput);
```

---

## 🧪 Testing

Run the automated unit test suite:

```bash
npm test
```

Run the multi-agent researcher example:

```bash
npm run example
```

---

## 🌐 Web Visualizer Playground

Launch the interactive dashboard locally by opening `dashboard/index.html` in any modern web browser or hosting via static server:

```bash
npx serve dashboard
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
