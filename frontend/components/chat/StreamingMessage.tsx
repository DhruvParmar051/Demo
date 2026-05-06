"use client";

import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { oneLight } from "react-syntax-highlighter/dist/esm/styles/prism";
import { useTheme } from "next-themes";
import type { Citation } from "@/lib/types";
import { CitationCard } from "./CitationCard";

interface StreamingMessageProps {
  content: string;
  citations: Citation[];
  isStreaming: boolean;
}

export function StreamingMessage({ content, citations, isStreaming }: StreamingMessageProps) {
  const { theme } = useTheme();
  const isDark = theme !== "light";

  if (!content && isStreaming) {
    return (
      <div className="flex items-center gap-1.5 py-1">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="w-1.5 h-1.5 rounded-full bg-accent/60"
            style={{ animation: `pulse 1.2s ease-in-out ${i * 0.2}s infinite` }}
          />
        ))}
      </div>
    );
  }

  return (
    <div>
      <div className="prose prose-sm max-w-none">
        <ReactMarkdown
          components={{
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            code({ className, children, ...props }: any) {
              const match = /language-(\w+)/.exec(className || "");
              if (match) {
                return (
                  <SyntaxHighlighter
                    style={(isDark ? oneDark : oneLight) as Record<string, React.CSSProperties>}
                    language={match[1]}
                    PreTag="div"
                    customStyle={{
                      background: isDark ? "#0d0d15" : "#f6f7f9",
                      borderRadius: "12px",
                      border: `1px solid var(--glass-border)`,
                      fontSize: "0.8rem",
                      margin: "0.75rem 0",
                    }}
                  >
                    {String(children).replace(/\n$/, "")}
                  </SyntaxHighlighter>
                );
              }
              return <code className={className} {...props}>{children}</code>;
            },
          }}
        >
          {content}
        </ReactMarkdown>
      </div>

      {citations.length > 0 && (
        <div className="flex flex-wrap items-center gap-1 mt-2 pt-2 border-t border-[var(--glass-border)]">
          <span className="text-xs text-[var(--muted)] mr-1">Sources:</span>
          {citations.map((c, i) => (
            <CitationCard key={c.chunk_id || i} citation={c} index={i} />
          ))}
        </div>
      )}

      {isStreaming && (
        <span className="inline-block w-0.5 h-4 bg-accent/80 ml-0.5 animate-pulse align-middle" />
      )}
    </div>
  );
}
