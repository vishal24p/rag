# RAG chat UI

Chat layout adapted from [prompt-kit Full Chat App](https://www.prompt-kit.com/c/full-chat-app): history sidebar, 64px header, open transcript, right-aligned user bubbles, and rounded bottom composer. Libraries.dev bot avatar supplies assistant identity. Citation buttons show actual retrieved chunk previews on hover, focus, or tap.

Built with React 19, TypeScript, Tailwind 4, shadcn/ui, and Vercel AI Elements. The project name is RAG; branding uses a text wordmark.

## Run

```sh
npm install
npm run dev
npm test
npm run build
```

Local preview: http://127.0.0.1:5173. Node 22.18+ required for native TypeScript test execution. Start FastAPI in another terminal from the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api:app --app-dir backend --host 127.0.0.1 --port 8000
```

Vite proxies `/api` to port 8000. Backend settings and Zen credentials belong in `backend/.env`; do not put secrets in Vite environment variables. A production build requires a host/reverse proxy that serves the UI and forwards `/api` to the backend; Vite's development proxy is not a production deployment.

## Components

- shadcn Sidebar handles desktop collapse and mobile Sheet navigation.
- shadcn MessageScroller handles transcript scrolling.
- prompt-kit PromptInput uses the Basic composition: textarea and one circular send/stop action. Enter submits, Shift+Enter inserts a newline, and IME composition does not submit.
- AI Elements MessageResponse renders returned markdown.
- prompt-kit Source shows indexed chunk IDs and excerpts with touch and keyboard access. Its Radix overlay retains fade/zoom/slide animation. Reduced motion is respected.
- [BotAvatar](https://libraries.dev/bots) provides assistant identity.

## Scope

Live text chat through `POST /api/query`. Jev chooses refusal, RAG or DIRECT before downstream work. Answers arrive as complete responses; loading state does not simulate token streaming. Only successful RAG answers have sources. Completed requests show `jev_decision_ms` and `total_request_ms` from the backend service; total excludes HTTP queueing, serialization, browser/network time and rendering.

Threads and feedback reset on refresh. Each request contains the current query only, so displayed chat history does not provide model memory. No document uploads or indexing controls are included. Stop cancels the browser request and ignores late results; backend inference already running can continue. API/provider failures show an error rather than a sample answer.

## Verification

`npm test` checks transport responses, validation and cancellation behavior. `npm run build` validates TypeScript and produces a production bundle. Browser checks cover real answers/sources, refusals/errors, Stop, thread switching, theme and mobile layout. Rich markdown libraries may produce a bundle-size advisory.

## Attribution

Layout and Source derived from [ibelick/prompt-kit](https://github.com/ibelick/prompt-kit), MIT license. Copyright (c) 2025 Julien Thibeaut. See PROMPT-KIT-LICENSE for the complete license. Registry primitives remain editable under src/components; visual styles live in src/index.css.

Typography uses self-hosted Geist, restored from the previous design.

The header light/dark toggle remembers an explicit preference locally; first visit follows system appearance. Both themes retain Geist and the same chat layout.
