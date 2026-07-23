import assert from 'node:assert/strict';
import { Agent, Tool, ToolRegistry, Memory, Workflow, MockProvider } from '../src/index.js';

async function runTests() {
  console.log('🧪 Starting nexus-ai-core unit tests...\n');

  // Test 1: Tool & Registry
  console.log('Test 1: Tool & ToolRegistry');
  const registry = new ToolRegistry();
  const mathTool = new Tool({
    name: 'calculator',
    description: 'Performs simple addition',
    execute: async ({ a, b }) => a + b
  });

  registry.register(mathTool);
  assert.equal(registry.has('calculator'), true);
  const result = await registry.executeTool('calculator', { a: 5, b: 10 });
  assert.equal(result.status, 'success');
  assert.equal(result.output, 15);
  console.log('  ✓ Tool registration and execution verified');

  // Test 2: Memory
  console.log('\nTest 2: Memory & Context');
  const memory = new Memory({ systemPrompt: 'System initialization' });
  memory.addUserMessage('Hello Agent');
  memory.addAssistantMessage('Hello User');
  const history = memory.getFormattedHistory();
  assert.equal(history.length, 3);
  assert.equal(history[0].role, 'system');
  assert.equal(history[1].role, 'user');
  console.log('  ✓ Memory tracking verified');

  // Test 3: Agent Execution
  console.log('\nTest 3: Agent Execution');
  const mockProvider = new MockProvider();
  mockProvider.addMockResponse({
    content: 'Calculated math result.',
    toolCalls: []
  });

  const agent = new Agent({
    name: 'TestAgent',
    role: 'Tester',
    provider: mockProvider,
    memory
  });

  const runResult = await agent.run('Run test calculation');
  assert.equal(runResult.agent, 'TestAgent');
  assert.equal(runResult.finalResponse, 'Calculated math result.');
  console.log('  ✓ Agent execution loop verified');

  // Test 4: Multi-Agent Workflow
  console.log('\nTest 4: Workflow Orchestrator');
  const researcher = new Agent({
    name: 'Researcher',
    provider: new MockProvider({
      mockResponses: [{ content: 'Research summary: AI frameworks are modular.', toolCalls: [] }]
    })
  });

  const writer = new Agent({
    name: 'Writer',
    provider: new MockProvider({
      mockResponses: [{ content: 'Final Article: AI frameworks enable autonomous workflows.', toolCalls: [] }]
    })
  });

  const pipeline = new Workflow({ name: 'ContentPipeline' });
  pipeline.addSequentialStep(researcher);
  pipeline.addSequentialStep(writer, (input) => `Turn this into an article: ${input}`);

  const wfResult = await pipeline.execute('Research modern AI frameworks');
  assert.equal(wfResult.logs.length, 2);
  assert.equal(wfResult.finalOutput.includes('Final Article'), true);
  console.log('  ✓ Multi-agent pipeline workflow verified');

  console.log('\n✅ All unit tests passed successfully!');
}

runTests().catch(err => {
  console.error('❌ Test suite failed:', err);
  process.exit(1);
});
