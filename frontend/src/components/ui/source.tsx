import {
  createContext,
  useContext,
  useEffect,
  useId,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { FileText } from "lucide-react";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";

// Adapted from prompt-kit Source (MIT) for retrieved passage previews.
const SourceContext = createContext<{
  open: boolean;
  reveal: () => void;
  dismiss: () => void;
  id: string;
} | null>(null);
export function Source({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const dismissed = useRef(false);
  const id = useId();
  function dismiss() {
    dismissed.current = true;
    setOpen(false);
  }
  function reveal() {
    dismissed.current = false;
    setOpen(true);
  }
  useEffect(() => {
    if (!open) return;
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") dismiss();
    };
    document.addEventListener("keydown", escape, true);
    return () => document.removeEventListener("keydown", escape, true);
  }, [open]);
  return (
    <SourceContext.Provider value={{ open, reveal, dismiss, id }}>
      <HoverCard
        open={open}
        onOpenChange={value => {
          if (!value || !dismissed.current) setOpen(value);
        }}
        openDelay={150}
        closeDelay={150}
      >
        {children}
      </HoverCard>
    </SourceContext.Provider>
  );
}
export function SourceTrigger({
  label,
  title,
}: {
  label: number;
  title: string;
}) {
  const context = useContext(SourceContext)!;
  return (
    <HoverCardTrigger asChild>
      <button
        type="button"
        className="citation"
        aria-label={`Source ${label}: ${title}`}
        aria-expanded={context.open}
        aria-describedby={context.open ? context.id : undefined}
        onPointerEnter={context.reveal}
        onClick={context.reveal}
        onTouchStart={context.reveal}
        onFocus={context.reveal}
      >
        {label}
      </button>
    </HoverCardTrigger>
  );
}
export function SourceContent({
  title,
  description,
  chunkId,
}: {
  title: string;
  description: string;
  chunkId: number;
}) {
  const context = useContext(SourceContext)!;
  return (
    <HoverCardContent
      id={context.id}
      onEscapeKeyDown={context.dismiss}
      side="top"
      align="start"
      sideOffset={8}
      className="citation-overlay w-80 max-w-[calc(100vw-32px)] p-4"
    >
      <div className="mb-3 flex items-center gap-2 text-xs text-muted-foreground">
        <FileText className="size-4" />
        <span>Retrieved passage · Chunk {chunkId}</span>
      </div>
      <h3 className="mb-2 text-sm font-medium">{title}</h3>
      <p className="text-sm leading-relaxed text-muted-foreground">
        {description}
      </p>
    </HoverCardContent>
  );
}
