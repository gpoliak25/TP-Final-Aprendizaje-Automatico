/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: { ignoreBuildErrors: true },
  eslint: { ignoreDuringBuilds: true },
  images: { unoptimized: true },
  // onnxruntime-web loads WASM from CDN at runtime; exclude from server bundle
  serverExternalPackages: ["onnxruntime-web"],
}

export default nextConfig
