"use client";

import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import type { Citation } from "@/lib/types";
import { CitationCard } from "./CitationCard";

interface StreamingMessageProps {
  content: string;
  citations: Citation[];
  isStreaming: boolean;
}

// Inject citation refs inline after relevant sentences
function injectCitationMarkers(text: string, citations: Citation[]): string {
  if (!citations.length) return text;
  // Simple approach: append all citation refs at the end of the answer
  return text;
}

export function StreamingMessage({ content, citations, isStreaming }: StreamingMessageProps) {
  if (!content && isStreaming) {
    return (
      <div className="flex items-center gap-1.5 py-1">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="w-1.5 h-1.5 rounded-full bg-accent/60"
            style={{
              animation: "typing 1.2s steps(1) infinite",
              animationDelay: `${i * 0.2}s`,
            }}
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
            code({ node, className, children, ...props }) {
              const match = /language-(\w+)/.exec(className || "");
              const isBlock = !!match;
              if (isBlock) {
                return (
                  <SyntaxHighlighter
                    style={oneDark as Record<string, React.CSSProperties>}
                    language={match[1]}
                    PreTag="div"
                    customStyle={{
                      background: "#0d0d15",
                      borderRadius: "12px",
                      border: "1px solid rgba(255,255,255,0.08)",
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

      {/* Citation inline refs */}
      {citations.length > 0 && (
        <div className="flex flex-wrap items-center gap-1 mt-2 pt-2 border-t border-white/[0.06]">
          <span className="text-xs text-muted mr-1">Sources:</span>
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
