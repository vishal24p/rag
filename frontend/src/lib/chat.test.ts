import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { randomUUID } from "node:crypto";
import vm from "node:vm";
import ts from "typescript";

// Exercise the real handlers without adding a production state abstraction.
const source = readFileSync(new URL("../App.tsx", import.meta.url), "utf8");
const tree = ts.createSourceFile("App.tsx", source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
const handlers: string[] = [];
function collect(node: ts.Node) {
  if (ts.isFunctionDeclaration(node) && ["submit", "stop"].includes(node.name?.text ?? ""))
    handlers.push(node.getText(tree));
  ts.forEachChild(node, collect);
}
collect(tree);
assert.equal(handlers.length, 2);
const code = ts.transpileModule(handlers.join("\n"), {
  compilerOptions: { target: ts.ScriptTarget.ES2022 },
}).outputText;

for (const [oldOutcome, answer, expected] of [
  ["resolve", "new answer", "new answer"],
  ["reject", "assistant : new answer", "new answer"],
  ["resolve", " \nAsSiStAnT\t:  new answer", "new answer"],
  ["reject", "Text mentions assistant: internally.", "Text mentions assistant: internally."],
]) {
  const requests: { resolve: (value: unknown) => void; reject: (error: Error) => void }[] = [];
  let threads = [
    { id: "first", title: "first", messages: [] as { text: string; stopped?: boolean }[] },
    { id: "second", title: "second", messages: [] as { text: string; stopped?: boolean }[] },
  ];
  let pending: unknown;
  const context = {
    requestRef: { current: null }, activeId: "first", input: "old query",
    crypto: { randomUUID }, AbortController,
    validPrompt: () => true,
    sendQuery: () => new Promise((resolve, reject) => requests.push({ resolve, reject })),
    setInput: () => {},
    setPending: (value: unknown) => { pending = value; },
    setThreads: (update: (rows: typeof threads) => typeof threads) => { threads = update(threads); },
  };
  const runtime = vm.createContext(context);
  vm.runInContext(code, runtime);
  const oldRun = vm.runInContext("submit()", runtime);
  vm.runInContext("stop()", runtime);
  assert.equal(pending, null);
  context.activeId = "second";
  context.input = "new query";
  const newRun = vm.runInContext("submit()", runtime);
  const newPending = pending;
  if (oldOutcome === "resolve") requests[0].resolve({ answer: "stale answer" });
  else requests[0].reject(new Error("stale failure"));
  await oldRun;
  assert.equal(pending, newPending, "old finally must not clear the new request");
  assert.equal(threads[0].messages[1].text, "");
  assert.equal(threads[0].messages[1].stopped, true);
  requests[1].resolve({ answer, status: "ok" });
  await newRun;
  assert.equal(threads[1].messages[1].text, expected);
  assert.equal(pending, null);
}
console.log("Chat Stop/new-request stale success, failure and finally checks passed.");
