export type QueryResponse = {
  request_id: string;
  status: "ok" | "blocked" | "error";
  route: "RAG" | "DIRECT" | null;
  block_score: number | null;
  blocked: boolean;
  answer: string | null;
  contexts: { chunk_id: number; content: string; score: number }[];
  jev_decision_ms: number | null;
  total_request_ms: number;
  error: string | null;
};

export function validPrompt(text: string) {
  return text.trim().length > 0 && text.length <= 10000;
}

const finite = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);
const nullableNumber = (value: unknown) => value === null || finite(value);
const nullableString = (value: unknown) => value === null || typeof value === "string";

function isQueryResponse(value: unknown): value is QueryResponse {
  if (!value || typeof value !== "object") return false;
  const row = value as Record<string, unknown>;
  return typeof row.request_id === "string" &&
    ["ok", "blocked", "error"].includes(String(row.status)) &&
    (row.route === null || row.route === "RAG" || row.route === "DIRECT") &&
    nullableNumber(row.block_score) && typeof row.blocked === "boolean" &&
    nullableString(row.answer) && nullableString(row.error) &&
    nullableNumber(row.jev_decision_ms) && finite(row.total_request_ms) &&
    Array.isArray(row.contexts) && row.contexts.every(chunk =>
      chunk && Number.isInteger(chunk.chunk_id) && typeof chunk.content === "string" && finite(chunk.score)) &&
    (row.status !== "ok" || (row.route !== null && typeof row.answer === "string" && !row.blocked)) &&
    (row.status !== "blocked" || row.blocked) &&
    ((row.status === "ok" && row.route === "RAG") || row.contexts.length === 0);
}

export async function sendQuery(query: string, signal: AbortSignal): Promise<QueryResponse> {
  if (!validPrompt(query)) throw new Error("Enter a question of at most 10,000 characters.");
  let response: Response;
  try {
    signal.throwIfAborted();
    response = await fetch("/api/query", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }), signal,
    });
  } catch (error) {
    if (signal.aborted) signal.throwIfAborted();
    if (error instanceof Error && error.name === "AbortError") throw error;
    throw new Error("Could not reach the server. Please try again.");
  }
  signal.throwIfAborted();
  if (response.status === 422) throw new Error("Enter a question of at most 10,000 characters.");
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    signal.throwIfAborted();
    throw new Error("The server returned an invalid response. Please try again.");
  }
  signal.throwIfAborted();
  const expectedStatus = response.status === 200 ? "ok" : response.status === 403 ? "blocked" :
    [500, 503].includes(response.status) ? "error" : null;
  if (!isQueryResponse(body) || body.status !== expectedStatus)
    throw new Error("The server returned an invalid response. Please try again.");
  return body;
}
