/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: { ignoreBuildErrors: true },
  eslint: { ignoreDuringBuilds: true },
  images: { unoptimized: true },
  // onnxruntime-web is client-only (WASM loaded from CDN); exclude from server bundles
  serverExternalPackages: ["onnxruntime-web"],
  webpack: (config, { isServer }) => {
    config.experiments = { ...config.experiments, asyncWebAssembly: true }
    config.resolve.fallback = { ...config.resolve.fallback, fs: false, path: false }
    if (isServer) {
      config.externals = [...(config.externals || []), "onnxruntime-web"]
    }
    return config
  },
}

export default nextConfig
