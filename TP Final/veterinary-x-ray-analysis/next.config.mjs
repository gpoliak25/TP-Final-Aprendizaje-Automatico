/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: { ignoreBuildErrors: true },
  images: { unoptimized: true },
  webpack: (config) => {
    // Required for onnxruntime-web WASM loading
    config.experiments = { ...config.experiments, asyncWebAssembly: true }
    config.resolve.fallback = { ...config.resolve.fallback, fs: false, path: false }
    return config
  },
}

export default nextConfig
