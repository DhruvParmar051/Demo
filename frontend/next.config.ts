import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "X-Frame-Options", value: "DENY" },
          { key: "X-Content-Type-Options", value: "nosniff" },
        ],
      },
    ];
  },
  async rewrites() {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
    return [
      { source: "/api/:path*", destination: `${apiUrl}/:path*` },
    ];
  },
};

export default nextConfig;
