import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AegisRAG — Grounded AI Copilot",
  description: "Confidence-gated retrieval-augmented generation for customer support",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark h-full">
      <body className="h-full overflow-hidden" style={{ background: "#0a0a0f" }}>
        {children}
      </body>
    </html>
  );
}
