import { Agent, Tool, Workflow, MockProvider } from '../src/index.js';

console.log('🚀 Running Multi-Agent Autonomous Research Workflow Example...\n');

// Step 1: Create Search Tool
const searchTool = new Tool({
  name: 'web_search',
  description: 'Searches public web databases for scientific papers and tech articles',
  execute: async ({ query }) => {
    return [
      `Paper 1: Next-Gen Autonomous AI Agents (2026)`,
      `Paper 2: Multi-Agent Workflow Systems & Parallel Orchestration`
    ];
  }
});

// Step 2: Create Researcher Agent
const researcherProvider = new MockProvider();
researcherProvider.addMockResponse({
  content: 'Initiating web query for autonomous AI frameworks.',
  toolCalls: [
    { id: 'c1', name: 'web_search', args: { query: 'autonomous AI frameworks' } }
  ]
});
researcherProvider.addMockResponse({
  content: 'Found relevant publications on Multi-Agent Workflow Systems and Autonomous AI Agents.',
  toolCalls: []
});

const researcher = new Agent({
  name: 'ResearchAgent',
  role: 'Academic Researcher',
  instructions: 'Gather facts and summarize technical literature.',
  provider: researcherProvider
});
researcher.registerTool(searchTool);

// Step 3: Create Technical Writer Agent
const writerProvider = new MockProvider({
  mockResponses: [
    {
      content: `# Technical Report: The Rise of Autonomous AI Frameworks\n\nModern AI agent frameworks leverage modular tool registries and multi-agent workflow orchestration. By separating memory, execution loops, and provider adapters, systems achieve robust agent collaboration.`,
      toolCalls: []
    }
  ]
});

const writer = new Agent({
  name: 'WriterAgent',
  role: 'Technical Publisher',
  instructions: 'Synthesize research data into structured Markdown documentation.',
  provider: writerProvider
});

// Step 4: Build Pipeline
const pipeline = new Workflow({
  name: 'Autonomous Research & Documentation Pipeline'
});

pipeline
  .addSequentialStep(researcher)
  .addSequentialStep(writer, (researchSummary) => `Write a technical report based on these findings:\n${researchSummary}`);

// Step 5: Run Workflow
const report = await pipeline.execute('Investigate 2026 state of AI agent frameworks');

console.log('--- Workflow Output ---');
console.log(report.finalOutput);
console.log('\n--- Execution Pipeline Logs ---');
console.dir(report.logs, { depth: null });
