import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone", // для Docker: .next/standalone/server.js без node_modules
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
