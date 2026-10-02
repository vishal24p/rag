# RAG chat UI

Chat layout adapted from [prompt-kit Full Chat App](https://www.prompt-kit.com/c/full-chat-app): grouped history sidebar, 64px header, open transcript, right-aligned user bubbles, and rounded bottom composer. Libraries.dev bot avatar is the visual addition. Citation buttons show animated document previews on hover, focus, or tap.

Built with React 19, TypeScript, Tailwind 4, shadcn/ui, and Vercel AI Elements. The project name is RAG; branding uses a text wordmark.

## Run

```sh
npm install
npm run dev
npm test
npm run build
```

Local preview: http://127.0.0.1:5173. Node 22.18+ required for native TypeScript test execution.

## Components

- shadcn Sidebar handles desktop collapse and mobile Sheet navigation.
- shadcn MessageScroller handles transcript scrolling.
- prompt-kit PromptInput uses the Basic composition: textarea and one circular send/stop action. Enter submits, Shift+Enter inserts a newline, and IME composition does not submit.
- AI Elements MessageResponse renders streaming markdown.
- prompt-kit Source is adapted for local sample documents, touch and keyboard access. Its Radix overlay retains fade/zoom/slide animation. Reduced motion is respected.
- [BotAvatar](https://libraries.dev/bots) provides assistant identity.

## Scope

Frontend-only demo. Threads and feedback reset on refresh. Responses and citation passages are sample data; no live model is connected. The composer accepts text prompts. Backend files were not changed.

## Verification

`npm test` checks prompt validation and fixture response selection. `npm run build` validates TypeScript and produces a production bundle. Browser checks cover desktop/mobile overflow, citations using keyboard/touch/Escape, blank sends, multiline input, streaming stop and thread switching. Rich markdown libraries produce a bundle-size advisory.

## Attribution

Layout and Source derived from [ibelick/prompt-kit](https://github.com/ibelick/prompt-kit), MIT license. Copyright (c) 2025 Julien Thibeaut. See PROMPT-KIT-LICENSE for the complete license. Registry primitives remain editable under src/components; visual styles live in src/index.css.

Typography uses self-hosted Geist, restored from the previous design.

The header light/dark toggle remembers an explicit preference locally; first visit follows system appearance. Both themes retain Geist and the same chat layout.
