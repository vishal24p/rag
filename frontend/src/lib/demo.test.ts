import assert from "node:assert/strict";
import { createDemoAnswer, validPrompt } from "./demo.ts";

assert.equal(validPrompt("   "), false);
assert.equal(validPrompt("Summarize the documents"), true);
assert.match(createDemoAnswer("What are the risks?"), /dependencies/i);
assert.match(createDemoAnswer("Compare customer research"), /strategy/i);
assert.match(createDemoAnswer("Summarize"), /launch/i);
console.log("Demo prompt and response selection checks passed.");
