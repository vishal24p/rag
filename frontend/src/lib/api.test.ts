import assert from "node:assert/strict";
import { sendQuery, validPrompt } from "./api.ts";

const originalFetch = globalThis.fetch;
const response = {
  request_id: "request", status: "ok", route: "RAG", block_score: 0.02,
  blocked: false, answer: "Answer [1]", contexts: [
    { chunk_id: 12, content: "Source text", score: 0.6 },
  ], jev_decision_ms: 12, total_request_ms: 40, error: null,
};
const signal = () => new AbortController().signal;
try {
  assert.equal(validPrompt("  "), false);
  assert.equal(validPrompt("a".repeat(10001)), false);
  assert.equal(validPrompt("valid"), true);
  assert.equal(validPrompt("x".repeat(10000)), true);
  globalThis.fetch = async () => { assert.fail("Invalid or cancelled request reached fetch"); };
  for (const query of ["", " \n\t", "x".repeat(10001)]) {
    await assert.rejects(sendQuery(query, signal()), /at most 10,000/);
  }
  const alreadyCancelled = new AbortController();
  alreadyCancelled.abort();
  await assert.rejects(sendQuery("question", alreadyCancelled.signal), error =>
    error instanceof DOMException && error.name === "AbortError");
  globalThis.fetch = async (url, options) => {
    assert.equal(url, "/api/query");
    assert.equal(options?.body, JSON.stringify({ query: " question " }));
    return Response.json(response);
  };
  assert.deepEqual(await sendQuery(" question ", signal()), response);
  for (const [status, body] of [
    [200, { ...response, route: "DIRECT", contexts: [] }],
    [403, { ...response, status: "blocked", blocked: true, contexts: [] }],
    [503, { ...response, status: "error", route: null, answer: "Unavailable", contexts: [], error: "gateway-timeout" }],
    [500, { ...response, status: "error", contexts: [], error: "generation-error" }],
  ] as const) {
    globalThis.fetch = async () => Response.json(body, { status });
    assert.deepEqual(await sendQuery("question", signal()), body);
  }
  for (const reply of [
    Response.json({ detail: "private exception" }, { status: 422 }),
    new Response("not json", { status: 200 }),
    Response.json({ ...response, total_request_ms: "bad" }),
    Response.json({ ...response, contexts: [{ chunk_id: 1 }] }),
    Response.json({ ...response, status: "error" }),
  ]) {
    globalThis.fetch = async () => reply;
    await assert.rejects(sendQuery("question", signal()), error =>
      error instanceof Error && !error.message.includes("private exception"));
  }
  for (const body of [
    null, [], { ...response, status: "unknown" },
    { ...response, route: null }, { ...response, answer: null },
    { ...response, blocked: true }, { ...response, route: "DIRECT" },
    { ...response, contexts: [{ chunk_id: 1.5, content: "text", score: 0.1 }] },
    { ...response, contexts: [{ chunk_id: true, content: "text", score: 0.1 }] },
    { ...response, jev_decision_ms: "12" },
  ]) {
    globalThis.fetch = async () => Response.json(body);
    await assert.rejects(sendQuery("question", signal()), /invalid response/);
  }
  for (const status of [201, 401, 404, 429, 502]) {
    globalThis.fetch = async () => Response.json(response, { status });
    await assert.rejects(sendQuery("question", signal()), /invalid response/);
  }
  for (const [status, body] of [
    [403, { ...response, status: "blocked", blocked: false, contexts: [] }],
    [500, { ...response, status: "error" }],
    [200, { ...response, route: "DIRECT" }],
  ] as const) {
    globalThis.fetch = async () => Response.json(body, { status });
    await assert.rejects(sendQuery("question", signal()), /invalid response/);
  }

  // Cancellation while the body is loading must survive JSON success or failure.
  for (const invalidJson of [false, true]) {
    const reading = new AbortController();
    const reply = Response.json(response);
    reply.json = async () => {
      reading.abort();
      if (invalidJson) throw new SyntaxError("PRIVATE_BODY");
      return response;
    };
    globalThis.fetch = async () => reply;
    await assert.rejects(sendQuery("question", reading.signal), error =>
      error instanceof DOMException && error.name === "AbortError");
  }
  globalThis.fetch = async () => { throw new Error("private transport detail"); };
  await assert.rejects(sendQuery("question", signal()), /Could not reach/);
  const aborted = new DOMException("cancelled", "AbortError");
  globalThis.fetch = async () => { throw aborted; };
  await assert.rejects(sendQuery("question", signal()), error => error === aborted);

  // A late response must stay cancelled even if a transport ignores AbortSignal.
  let resolve!: (value: Response) => void;
  globalThis.fetch = () => new Promise(done => { resolve = done; });
  const controller = new AbortController();
  const pending = sendQuery("question", controller.signal);
  controller.abort();
  resolve(Response.json(response));
  await assert.rejects(pending, error => error instanceof DOMException && error.name === "AbortError");
  console.log("API response, error and late-cancellation checks passed.");
} finally {
  globalThis.fetch = originalFetch;
}
