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
import {
  documents,
  openingAnswer,
  openingQuestion,
  createDemoAnswer,
  validPrompt,
} from "@/lib/demo";
type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  stopped?: boolean;
  feedback?: "up" | "down";
};
type Thread = { id: string; title: string; messages: ChatMessage[] };
const initialThreads: Thread[] = [
  { id: "welcome", title: "New conversation", messages: [] },
  {
    id: "launch",
    title: "Q4 launch priorities",
    messages: [
      { id: "q1", role: "user", text: openingQuestion },
      { id: "a1", role: "assistant", text: openingAnswer },
    ],
  },
  {
    id: "research",
    title: "What our customers need",
    messages: [
      {
        id: "q2",
        role: "user",
        text: "Compare the strategy with customer research.",
      },
      {
        id: "a2",
        role: "assistant",
        text: createDemoAnswer("compare research"),
      },
    ],
  },
  {
    id: "risks",
    title: "Roadmap & dependencies",
    messages: [
      { id: "q3", role: "user", text: "What are the biggest launch risks?" },
      { id: "a3", role: "assistant", text: createDemoAnswer("risks") },
    ],
  },
];
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
  const [activeId, setActiveId] = useState("launch");
  const [input, setInput] = useState("");
  const [toast, setToast] = useState("");
  const [search, setSearch] = useState("");
  const [searching, setSearching] = useState(false);
  const [stream, setStream] = useState<{
    threadId: string;
    messageId: string;
    text: string;
  } | null>(null);
  const generating = useRef(false);
  const composerRef = useRef<HTMLDivElement>(null);
  const active = threads.find(thread => thread.id === activeId)!;

  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(""), 3500);
    return () => clearTimeout(timer);
  }, [toast]);
  useEffect(() => {
    if (!stream) return;
    let cursor = 0;
    const timer = setInterval(() => {
      cursor = Math.min(cursor + 18, stream.text.length);
      const text = stream.text.slice(0, cursor);
      setThreads(prev =>
        prev.map(thread =>
          thread.id === stream.threadId
            ? {
                ...thread,
                messages: thread.messages.map(message =>
                  message.id === stream.messageId
                    ? { ...message, text }
                    : message
                ),
              }
            : thread
        )
      );
      if (cursor === stream.text.length) {
        clearInterval(timer);
        generating.current = false;
        setStream(null);
      }
    }, 22);
    return () => clearInterval(timer);
  }, [stream]);

  function stop() {
    if (stream)
      setThreads(prev =>
        prev.map(thread =>
          thread.id === stream.threadId
            ? {
                ...thread,
                messages: thread.messages.map(message =>
                  message.id === stream.messageId
                    ? { ...message, stopped: true }
                    : message
                ),
              }
            : thread
        )
      );
    generating.current = false;
    setStream(null);
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
  function submit() {
    if (!validPrompt(input) || generating.current) return;
    generating.current = true;
    const text = input.trim();
    const messageId = crypto.randomUUID();
    const response = createDemoAnswer(text);
    setThreads(prev =>
      prev.map(thread =>
        thread.id === activeId
          ? {
              ...thread,
              title: thread.messages.length ? thread.title : text.slice(0, 38),
              messages: [
                ...thread.messages,
                {
                  id: crypto.randomUUID(),
                  role: "user",
                  text,
                },
                { id: messageId, role: "assistant", text: "" },
              ],
            }
          : thread
      )
    );
    setInput("");
    setStream({ threadId: activeId, messageId, text: response });
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
        isLoading={!!stream}
        onSubmit={submit}
        className="w-full max-w-(--breakpoint-md)"
      >
        <PromptInputTextarea
          className="dark:bg-transparent"
          placeholder="Ask me anything..."
          aria-label="Ask your knowledge"
          disabled={!!stream}
        />
        <PromptInputActions className="justify-end pt-2">
          <PromptInputAction
            tooltip={stream ? "Stop generation" : "Send message"}
          >
            <Button
              type="button"
              variant="default"
              size="icon"
              className="h-8 w-8 rounded-full"
              disabled={!stream && !validPrompt(input)}
              aria-label={stream ? "Stop response" : "Send message"}
              onClick={stream ? stop : submit}
            >
              {stream ? (
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
              autoScroll={!!stream}
              defaultScrollPosition="start"
            >
              <MessageScroller className="h-full">
                <MessageScrollerViewport>
                  <MessageScrollerContent
                    className="transcript-content"
                    aria-busy={!!stream}
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
                                paused={stream?.messageId !== message.id}
                                size={32}
                                theme={dark ? "dark" : "light"}
                                interactive={false}
                                jumpEvery={0}
                                shading="fabric"
                              />
                            </div>
                            <div className="assistant-body">
                              {message.text ? (
                                <MessageResponse
                                  isAnimating={stream?.messageId === message.id}
                                >
                                  {message.text}
                                </MessageResponse>
                              ) : message.stopped ? (
                                <p>No response generated.</p>
                              ) : null}
                              {stream?.messageId !== message.id &&
                                message.text && (
                                  <div
                                    className="citations"
                                    aria-label="Answer sources"
                                  >
                                    <span>Sources</span>
                                    {documents.map((doc, i) => (
                                      <Source key={doc.id}>
                                        <SourceTrigger
                                          label={i + 1}
                                          title={doc.name}
                                        />
                                        <SourceContent
                                          title={doc.name}
                                          description={doc.excerpt}
                                          page={doc.page}
                                        />
                                      </Source>
                                    ))}
                                  </div>
                                )}
                              {message.stopped && (
                                <p className="stopped-label">
                                  Response stopped
                                </p>
                              )}
                              <div
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
                              </div>
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
