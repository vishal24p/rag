import { useEffect, useRef, useState } from "react";
import {
  ArrowUp,
  Copy,
  Moon,
  Sun,
  Plus,
  Search,
  Square,
  ThumbsDown,
  ThumbsUp,
} from "lucide-react";
import { BotAvatar } from "bot-avatars";
import { Button } from "@/components/ui/button";
import { Source, SourceContent, SourceTrigger } from "@/components/ui/source";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarProvider,
  SidebarTrigger,
  useSidebar,
} from "@/components/ui/sidebar";
import {
  MessageScrollerProvider,
  MessageScroller,
  MessageScrollerViewport,
  MessageScrollerContent,
  MessageScrollerItem,
  MessageScrollerButton,
} from "@/components/ui/message-scroller";
import {
  MessageResponse,
  MessageAction,
} from "@/components/ai-elements/message";
import {
  PromptInput,
  PromptInputAction,
  PromptInputActions,
  PromptInputTextarea,
} from "@/components/prompt-kit/prompt-input";
import { sendQuery, validPrompt, type QueryResponse } from "@/lib/api";
type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  stopped?: boolean;
  feedback?: "up" | "down";
  result?: QueryResponse;
  error?: boolean;
};
type Thread = { id: string; title: string; messages: ChatMessage[] };
const initialThreads: Thread[] = [{ id: "welcome", title: "New conversation", messages: [] }];
type PendingRequest = { threadId: string; messageId: string; controller: AbortController };
function ChatApp() {
  const { setOpenMobile } = useSidebar();
  const [dark, setDark] = useState(() =>
    document.documentElement.classList.contains("dark")
  );
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    document.documentElement.style.colorScheme = dark ? "dark" : "light";
    document
      .querySelector('meta[name="theme-color"]')
      ?.setAttribute("content", dark ? "#181818" : "#ffffff");
  }, [dark]);
  function toggleTheme() {
    const next = !dark;
    setDark(next);
    try {
      localStorage.setItem("rag-theme", next ? "dark" : "light");
    } catch {
      /* Theme still works when storage is unavailable. */
    }
  }
  const [threads, setThreads] = useState<Thread[]>(initialThreads);
  const [activeId, setActiveId] = useState("welcome");
  const [input, setInput] = useState("");
  const [toast, setToast] = useState("");
  const [search, setSearch] = useState("");
  const [searching, setSearching] = useState(false);
  const [pending, setPending] = useState<PendingRequest | null>(null);
  const requestRef = useRef<PendingRequest | null>(null);
  const composerRef = useRef<HTMLDivElement>(null);
  const active = threads.find(thread => thread.id === activeId)!;

  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(""), 3500);
    return () => clearTimeout(timer);
  }, [toast]);
  useEffect(() => () => {
    requestRef.current?.controller.abort();
    requestRef.current = null;
  }, []);

  function stop() {
    const request = requestRef.current;
    if (!request) return;
    requestRef.current = null;
    request.controller.abort();
    setPending(null);
    setThreads(prev => prev.map(thread => thread.id === request.threadId ? {
      ...thread,
      messages: thread.messages.map(message => message.id === request.messageId
        ? { ...message, stopped: true } : message),
    } : thread));
  }
  function switchThread(id: string) {
    stop();
    setActiveId(id);
    setInput("");
    setOpenMobile(false);
  }
  function newChat() {
    stop();
    const id = crypto.randomUUID();
    setThreads(prev => [
      { id, title: "New conversation", messages: [] },
      ...prev,
    ]);
    setActiveId(id);
    setInput("");
    setOpenMobile(false);
    requestAnimationFrame(() =>
      composerRef.current?.querySelector("textarea")?.focus()
    );
  }
  async function submit() {
    if (!validPrompt(input) || requestRef.current) return;
    const text = input;
    const messageId = crypto.randomUUID();
    const request = { threadId: activeId, messageId, controller: new AbortController() };
    requestRef.current = request;
    setPending(request);
    setThreads(prev => prev.map(thread => thread.id === request.threadId ? {
      ...thread,
      title: thread.messages.length ? thread.title : text.trim().slice(0, 38),
      messages: [...thread.messages,
        { id: crypto.randomUUID(), role: "user", text },
        { id: messageId, role: "assistant", text: "" }],
    } : thread));
    setInput("");
    try {
      const result = await sendQuery(text, request.controller.signal);
      if (requestRef.current !== request || request.controller.signal.aborted) return;
      setThreads(prev => prev.map(thread => thread.id === request.threadId ? {
        ...thread,
        messages: thread.messages.map(message => message.id === request.messageId
          ? { ...message, text: result.answer?.replace(/^\s*assistant\s*:\s*/i, "") ?? "No answer returned.", result } : message),
      } : thread));
    } catch (error) {
      if (requestRef.current !== request || request.controller.signal.aborted) return;
      setThreads(prev => prev.map(thread => thread.id === request.threadId ? {
        ...thread,
        messages: thread.messages.map(message => message.id === request.messageId
          ? { ...message, text: error instanceof Error ? error.message : "The request failed. Please try again.", error: true }
          : message),
      } : thread));
    } finally {
      // An old response must never clear a newer request's loading state.
      if (requestRef.current === request) {
        requestRef.current = null;
        setPending(null);
      }
    }
  }
  function updateFeedback(id: string, feedback: "up" | "down") {
    setThreads(prev =>
      prev.map(thread =>
        thread.id === activeId
          ? {
              ...thread,
              messages: thread.messages.map(message =>
                message.id === id
                  ? {
                      ...message,
                      feedback:
                        message.feedback === feedback ? undefined : feedback,
                    }
                  : message
              ),
            }
          : thread
      )
    );
  }
  async function copy(text: string) {
    try {
      await navigator.clipboard.writeText(text);
      setToast("Answer copied");
    } catch {
      setToast("Clipboard unavailable. Select the answer text to copy.");
    }
  }
  const composer = (
    <div className="composer-wrap" ref={composerRef}>
      <PromptInput
        key={activeId}
        value={input}
        onValueChange={setInput}
        isLoading={!!pending}
        onSubmit={submit}
        className="w-full max-w-(--breakpoint-md)"
      >
        <PromptInputTextarea
          className="dark:bg-transparent"
          placeholder="Ask me anything..."
          aria-label="Ask your knowledge"
          disabled={!!pending}
        />
        <PromptInputActions className="justify-end pt-2">
          <PromptInputAction
            tooltip={pending ? "Cancel waiting" : "Send message"}
          >
            <Button
              type="button"
              variant="default"
              size="icon"
              className="h-8 w-8 rounded-full"
              disabled={!pending && !validPrompt(input)}
              aria-label={pending ? "Cancel waiting" : "Send message"}
              onClick={pending ? stop : submit}
            >
              {pending ? (
                <Square className="size-5 fill-current" />
              ) : (
                <ArrowUp className="size-5" />
              )}
            </Button>
          </PromptInputAction>
        </PromptInputActions>
      </PromptInput>
    </div>
  );

  const recentThreads = threads.filter(thread => thread.messages.length > 0);
  return (
    <>
      <Sidebar>
        <SidebarHeader className="flex flex-row items-center justify-between gap-2 px-2 py-4">
          <div className="flex items-center gap-2 px-2">
            <span className="brand-wordmark">RAG</span>
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="size-8"
            aria-label="Search conversations"
            onClick={() => setSearching(v => !v)}
          >
            <Search className="size-4" />
          </Button>
        </SidebarHeader>
        <SidebarContent className="pt-4">
          <div className="px-4">
            <Button variant="outline" className="mb-4 w-full" onClick={newChat}>
              <Plus className="size-4" />
              New Chat
            </Button>
            {searching && (
              <input
                className="conversation-search"
                aria-label="Find a conversation"
                placeholder="Search chats..."
                autoFocus
                value={search}
                onChange={e => setSearch(e.target.value)}
              />
            )}
          </div>
          {["Today", "Yesterday"].map((period, group) => (
            <SidebarGroup key={period}>
              <SidebarGroupLabel>{period}</SidebarGroupLabel>
              <SidebarMenu>
                {recentThreads
                  .filter(
                    (t, i) =>
                      (group === 0 ? i < 2 : i >= 2) &&
                      t.title.toLowerCase().includes(search.toLowerCase())
                  )
                  .map(t => (
                    <SidebarMenuButton
                      key={t.id}
                      isActive={t.id === activeId}
                      onClick={() => switchThread(t.id)}
                    >
                      <span>{t.title}</span>
                    </SidebarMenuButton>
                  ))}
              </SidebarMenu>
            </SidebarGroup>
          ))}
          {search &&
            !recentThreads.some(t =>
              t.title.toLowerCase().includes(search.toLowerCase())
            ) && (
              <p className="px-4 text-sm text-muted-foreground">
                No conversations found.
              </p>
            )}
        </SidebarContent>
      </Sidebar>
      <SidebarInset>
        <main className="chat-main">
          <header className="chat-header">
            <SidebarTrigger className="-ml-1" />
            <span className="min-w-0 flex-1 truncate">{active.title}</span>
            <Button
              variant="ghost"
              size="icon"
              className="theme-toggle size-8 rounded-full"
              onClick={toggleTheme}
              aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
              title={dark ? "Switch to light mode" : "Switch to dark mode"}
              aria-pressed={dark}
            >
              {dark ? <Sun className="size-4" /> : <Moon className="size-4" />}
            </Button>
          </header>
          <div className="transcript">
            <MessageScrollerProvider
              key={activeId}
              autoScroll={!!pending}
              defaultScrollPosition="start"
            >
              <MessageScroller className="h-full">
                <MessageScrollerViewport>
                  <MessageScrollerContent
                    className="transcript-content"
                    aria-busy={!!pending}
                  >
                    {active.messages.length === 0 && (
                      <div className="empty-chat">
                        <BotAvatar
                          type="clover"
                          size={64}
                          theme={dark ? "dark" : "light"}
                          interactive={false}
                          jumpEvery={0}
                          shading="fabric"
                        />
                        <h1>How can I help?</h1>
                      </div>
                    )}
                    {active.messages.map((message, index) => (
                      <MessageScrollerItem
                        key={message.id}
                        messageId={message.id}
                        className="message-row"
                      >
                        {message.role === "user" ? (
                          <div className="user-message">
                            <p>{message.text}</p>
                          </div>
                        ) : (
                          <div className="assistant-message group">
                            <div className="bot-avatar">
                              <BotAvatar
                                type="clover"
                                paused={pending?.messageId !== message.id}
                                size={32}
                                theme={dark ? "dark" : "light"}
                                interactive={false}
                                jumpEvery={0}
                                shading="fabric"
                              />
                            </div>
                            <div className="assistant-body">
                              {(message.error || message.result?.status === "error") && <p role="alert">Request failed</p>}
                              {message.result?.status === "blocked" && <p role="status">Request refused</p>}
                              {message.text ? (
                                <MessageResponse
                                  isAnimating={false}
                                >
                                  {message.text}
                                </MessageResponse>
                              ) : message.stopped ? (
                                <p>Waiting cancelled.</p>
                              ) : pending?.messageId === message.id ? <p role="status">Thinking...</p> : null}
                              {message.result && (
                                <p className="text-xs text-muted-foreground mt-2">
                                  Route: {message.result.route ?? "unavailable"} · Jev: {message.result.jev_decision_ms === null ? "N/A" : `${message.result.jev_decision_ms.toFixed(0)} ms`} · Total: {message.result.total_request_ms.toFixed(0)} ms
                                </p>
                              )}
                              {message.result?.status === "ok" && message.result.route === "RAG" &&
                                message.result.contexts.length > 0 && (
                                  <div
                                    className="citations"
                                    aria-label="Answer sources"
                                  >
                                    <span>Sources</span>
                                    {message.result.contexts.map((chunk, i) => (
                                      <Source key={chunk.chunk_id}>
                                        <SourceTrigger
                                          label={i + 1}
                                          title={`Chunk ${chunk.chunk_id}`}
                                        />
                                        <SourceContent
                                          title={`Chunk ${chunk.chunk_id}`}
                                          description={chunk.content}
                                          chunkId={chunk.chunk_id}
                                        />
                                      </Source>
                                    ))}
                                  </div>
                                )}
                              {message.stopped && (
                                <p className="stopped-label">
                                  Browser waiting cancelled; server processing may continue.
                                </p>
                              )}
                              {message.result?.status === "ok" && <div
                                className={`message-actions ${index === active.messages.length - 1 ? "last-actions" : ""}`}
                              >
                                <MessageAction
                                  tooltip="Copy"
                                  label="Copy answer"
                                  onClick={() => copy(message.text)}
                                >
                                  <Copy className="size-4" />
                                </MessageAction>
                                <MessageAction
                                  tooltip="Upvote"
                                  label="Helpful answer"
                                  aria-pressed={message.feedback === "up"}
                                  onClick={() =>
                                    updateFeedback(message.id, "up")
                                  }
                                  className={
                                    message.feedback === "up"
                                      ? "feedback-selected"
                                      : ""
                                  }
                                >
                                  <ThumbsUp className="size-4" />
                                </MessageAction>
                                <MessageAction
                                  tooltip="Downvote"
                                  label="Unhelpful answer"
                                  aria-pressed={message.feedback === "down"}
                                  onClick={() =>
                                    updateFeedback(message.id, "down")
                                  }
                                  className={
                                    message.feedback === "down"
                                      ? "feedback-selected"
                                      : ""
                                  }
                                >
                                  <ThumbsDown className="size-4" />
                                </MessageAction>
                              </div>}
                            </div>
                          </div>
                        )}
                      </MessageScrollerItem>
                    ))}
                  </MessageScrollerContent>
                </MessageScrollerViewport>
                <MessageScrollerButton
                  aria-label="Scroll to latest message"
                  className="scroll-latest"
                />
              </MessageScroller>
            </MessageScrollerProvider>
          </div>
          <div className="composer-dock">{composer}</div>
        </main>
      </SidebarInset>
      {toast && (
        <div className="toast" role="status">
          {toast}
        </div>
      )}
    </>
  );
}
export default function App() {
  return (
    <SidebarProvider>
      <ChatApp />
    </SidebarProvider>
  );
}
