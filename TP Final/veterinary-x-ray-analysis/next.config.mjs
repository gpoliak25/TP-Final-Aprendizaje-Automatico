/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: { ignoreBuildErrors: true },
  eslint: { ignoreDuringBuilds: true },
  images: { unoptimized: true },
  // onnxruntime-web is client-only (WASM); keep it out of server bundles
  serverExternalPackages: ["onnxruntime-web"],
  // Declare Turbopack config so Next.js 16 doesn't error on the webpack config below
  turbopack: {},
  // webpack is used locally via `next dev --webpack`; ignored in Vercel Turbopack builds
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
