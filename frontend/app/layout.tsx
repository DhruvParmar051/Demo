import type { Metadata } from "next";
import "./globals.css";

export const viewport = {
  colorScheme: "dark",
};

export const metadata: Metadata = {
  title: "AegisRAG — Grounded AI Copilot",
  description: "Confidence-gated retrieval-augmented generation for customer support",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark h-full">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="h-full overflow-hidden" style={{ background: "#0a0a0f", color: "#f1f5f9", fontFamily: "'Inter', system-ui, sans-serif" }}>
        {children}
      </body>
    </html>
  );
}
