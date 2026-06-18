"use client"

import { useEffect, useRef, useState, type ReactNode } from "react"
import { createPortal } from "react-dom"
import {
  Activity, CheckCircle2, AlertTriangle, ScanLine, Eye,
  Upload, Play, Loader2, Terminal, Clock, Stethoscope, X, Download,
} from "lucide-react"
import { SlideShell, Panel } from "@/components/hud"

// ── Constants ────────────────────────────────────────────────────────────────
const THRESHOLD = 0.30
const MODEL_URL = "/transfer_mobilenetv2.onnx"
const IMG_SIZE  = 224

const CASES = [
  { label: "Caso B", tag: "Falso negativo", xray: "/fn-original.png",  overlay: "/fn-gradcam.png",  actual: "Patológica" as const },
  { label: "Caso C", tag: "Falso positivo", xray: "/fp-original.png",  overlay: "/fp-gradcam.png",  actual: "Normal"     as const },
]

const STEPS = [
  { title: "Exploración",         emoji: "🔍" },
  { title: "Preparación",         emoji: "✂️" },
  { title: "CNN desde Cero",      emoji: "🧱" },
  { title: "Transfer Learning",   emoji: "🔁" },
  { title: "Evaluación",          emoji: "📊" },
  { title: "GradCAM",             emoji: "🗺️" },
]

// ── Types ────────────────────────────────────────────────────────────────────
type ActualLabel = "Patológica" | "Normal"
type PipeRun = "idle" | "running" | "ok" | "error"

type HistoryEntry = {
  id: string
  at: Date
  label: string
  imageSrc: string
  overlayUrl?: string
  actual?: ActualLabel
  prob: number
}

// ── ONNX session (module-level cache) ────────────────────────────────────────
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let _session: any = null

async function getSession() {
  if (_session) return _session
  const ort = await import("onnxruntime-web")
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/"
  ort.env.wasm.numThreads = 1
  _session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] })
  return _session
}

async function runInference(imgEl: HTMLImageElement): Promise<number> {
  const ort     = await import("onnxruntime-web")
  const session = await getSession()

  const canvas = document.createElement("canvas")
  canvas.width = IMG_SIZE; canvas.height = IMG_SIZE
  const ctx = canvas.getContext("2d")!
  ctx.drawImage(imgEl, 0, 0, IMG_SIZE, IMG_SIZE)
  const { data } = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE)

  const f32 = new Float32Array(IMG_SIZE * IMG_SIZE * 3)
  for (let i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
    f32[i * 3]     = data[i * 4]
    f32[i * 3 + 1] = data[i * 4 + 1]
    f32[i * 3 + 2] = data[i * 4 + 2]
  }
  const tensor = new ort.Tensor("float32", f32, [1, IMG_SIZE, IMG_SIZE, 3])
  const out    = await session.run({ [session.inputNames[0]]: tensor })
  return (out[session.outputNames[0]].data as Float32Array)[1]
}

// ── Grad-CAM dinámico por OCLUSIÓN (Zeiler & Fergus, 2014) ───────────────────
// Desliza un parche neutro por la imagen y mide cuánto cae la probabilidad de la
// clase predicha. Las zonas que más bajan la confianza son las más relevantes.
// No requiere gradientes ni modificar el modelo: solo inferencia hacia adelante.
const OCC_GRID = 12  // resolución del mapa (12×12 = 144 inferencias)

function jet(v: number): [number, number, number] {
  const r = Math.min(Math.max(1.5 - Math.abs(4 * v - 3), 0), 1)
  const g = Math.min(Math.max(1.5 - Math.abs(4 * v - 2), 0), 1)
  const b = Math.min(Math.max(1.5 - Math.abs(4 * v - 1), 0), 1)
  return [r * 255, g * 255, b * 255]
}

// Compone el heatmap (grilla GRID×GRID, valores 0..1) sobre la radiografía y
// devuelve un data-URL listo para usar como overlay.
function composeHeatmap(imgEl: HTMLImageElement, heat: Float32Array, grid: number): string {
  // Heatmap a baja resolución; el reescalado bilineal lo suaviza al ampliarlo.
  const small = document.createElement("canvas")
  small.width = grid; small.height = grid
  const sctx = small.getContext("2d")!
  const px   = sctx.createImageData(grid, grid)
  for (let i = 0; i < grid * grid; i++) {
    const v = heat[i]
    const [r, g, b] = jet(v)
    px.data[i * 4]     = r
    px.data[i * 4 + 1] = g
    px.data[i * 4 + 2] = b
    px.data[i * 4 + 3] = Math.round(Math.pow(v, 0.85) * 0.62 * 255)  // alpha ∝ relevancia
  }
  sctx.putImageData(px, 0, 0)

  // Lienzo final con la proporción original de la imagen (cap a 512px).
  const NW = imgEl.naturalWidth || IMG_SIZE
  const NH = imgEl.naturalHeight || IMG_SIZE
  const scale = Math.min(1, 512 / Math.max(NW, NH))
  const W = Math.max(1, Math.round(NW * scale))
  const H = Math.max(1, Math.round(NH * scale))

  const out  = document.createElement("canvas")
  out.width = W; out.height = H
  const octx = out.getContext("2d")!
  octx.drawImage(imgEl, 0, 0, W, H)
  octx.imageSmoothingEnabled = true
  octx.imageSmoothingQuality = "high"
  octx.drawImage(small, 0, 0, W, H)  // upscale bilineal del heatmap
  return out.toDataURL("image/png")
}

async function runOcclusion(
  imgEl: HTMLImageElement,
  baseProb: number,
  predicted: ActualLabel,
  onProgress?: (frac: number) => void,
): Promise<string> {
  const ort     = await import("onnxruntime-web")
  const session = await getSession()

  // Tensor base (NHWC, 0..255) + color neutro = media de la imagen.
  const canvas = document.createElement("canvas")
  canvas.width = IMG_SIZE; canvas.height = IMG_SIZE
  const ctx = canvas.getContext("2d")!
  ctx.drawImage(imgEl, 0, 0, IMG_SIZE, IMG_SIZE)
  const { data } = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE)

  const base = new Float32Array(IMG_SIZE * IMG_SIZE * 3)
  let sum = 0
  for (let i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
    const r = data[i * 4], g = data[i * 4 + 1], b = data[i * 4 + 2]
    base[i * 3] = r; base[i * 3 + 1] = g; base[i * 3 + 2] = b
    sum += r + g + b
  }
  const mean = sum / (IMG_SIZE * IMG_SIZE * 3)

  const inputName  = session.inputNames[0]
  const outputName = session.outputNames[0]
  // Probabilidad de la clase predicha (binario: patológica = p, normal = 1−p).
  const classProb  = (p: number) => (predicted === "Patológica" ? p : 1 - p)
  const base0      = classProb(baseProb)

  const cell = IMG_SIZE / OCC_GRID
  const half = cell * 0.85            // parche ≈ 1.7 celdas → solapado
  const heat = new Float32Array(OCC_GRID * OCC_GRID)

  for (let gy = 0; gy < OCC_GRID; gy++) {
    for (let gx = 0; gx < OCC_GRID; gx++) {
      const cxp = (gx + 0.5) * cell, cyp = (gy + 0.5) * cell
      const x0 = Math.max(0, Math.round(cxp - half)), x1 = Math.min(IMG_SIZE, Math.round(cxp + half))
      const y0 = Math.max(0, Math.round(cyp - half)), y1 = Math.min(IMG_SIZE, Math.round(cyp + half))

      const buf = base.slice()
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const idx = (y * IMG_SIZE + x) * 3
          buf[idx] = mean; buf[idx + 1] = mean; buf[idx + 2] = mean
        }
      }
      const t   = new ort.Tensor("float32", buf, [1, IMG_SIZE, IMG_SIZE, 3])
      const o   = await session.run({ [inputName]: t })
      const p   = (o[outputName].data as Float32Array)[1]
      heat[gy * OCC_GRID + gx] = Math.max(0, base0 - classProb(p))  // caída de confianza
      onProgress?.((gy * OCC_GRID + gx + 1) / (OCC_GRID * OCC_GRID))
    }
  }

  // Normalización a 0..1.
  let max = 0
  for (let i = 0; i < heat.length; i++) if (heat[i] > max) max = heat[i]
  if (max > 0) for (let i = 0; i < heat.length; i++) heat[i] /= max

  return composeHeatmap(imgEl, heat, OCC_GRID)
}

function loadImg(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = document.createElement("img")
    img.crossOrigin = "anonymous"
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = src
  })
}

function fmtTime(d: Date) {
  return d.toLocaleTimeString("es-AR", { hour: "2-digit", minute: "2-digit", second: "2-digit" })
}
function fmtDate(d: Date) {
  return d.toLocaleDateString("es-AR", { day: "2-digit", month: "2-digit" })
}

// Colores rotativos para los títulos de sección (pantalla y PDF).
const SECTION_COLORS = ["var(--hud-cyan)", "var(--hud-amber)", "var(--hud-green)", "#c084fc", "var(--hud-red)"]
const PRINT_SECTION_COLORS = ["#0e7c8c", "#b3781a", "#138a5e", "#7c3aed", "#c0392b"]

// ── Conversión de Markdown → HTML (para el PDF imprimible) ───────────────────
function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c] as string
  ))
}
function inlineToHtml(text: string): string {
  let t = escapeHtml(text)
  t = t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
  t = t.replace(/\*([^*]+)\*/g, "<em>$1</em>")
  t = t.replace(/`([^`]+)`/g, "<code>$1</code>")
  return t
}
function mdToHtml(md: string): string {
  const lines = md.replace(/\r/g, "").split("\n")
  let html = "", listType: "ul" | "ol" | null = null, sec = 0
  let list: string[] = []
  const flush = () => {
    if (!list.length) { listType = null; return }
    const tag = listType === "ol" ? "ol" : "ul"
    html += `<${tag}>` + list.map((it) => `<li>${inlineToHtml(it)}</li>`).join("") + `</${tag}>`
    list = []; listType = null
  }
  for (const raw of lines) {
    const line = raw.trim()
    if (/^---+$/.test(line)) { flush(); html += "<hr/>"; continue }
    if (!line) { flush(); continue }
    const h1 = line.match(/^#\s+(.*)$/)
    if (h1) { flush(); html += `<h1>${inlineToHtml(h1[1])}</h1>`; continue }
    const hN = line.match(/^#{2,4}\s+(.*)$/)
    const bold = line.match(/^\*\*(.+?)\*\*:?\s*$/)
    if (hN || bold) {
      flush()
      const color = PRINT_SECTION_COLORS[sec % PRINT_SECTION_COLORS.length]; sec++
      const title = inlineToHtml((hN ? hN[1] : bold![1]).replace(/:$/, ""))
      html += `<h2 style="color:${color};border-left:4px solid ${color}">${title}</h2>`
      continue
    }
    const ol = line.match(/^\d+\.\s+(.*)$/)
    if (ol) { if (listType === "ul") flush(); listType = "ol"; list.push(ol[1]); continue }
    const ul = line.match(/^[-*]\s+(.*)$/)
    if (ul) { if (listType === "ol") flush(); listType = "ul"; list.push(ul[1]); continue }
    flush(); html += `<p>${inlineToHtml(line)}</p>`
  }
  flush()
  return html
}
function buildReportHtml(
  md: string,
  meta: { title: string; predicted: string | null; prob: number | null; actual?: string | null; label?: string; date: string },
): string {
  const pct = meta.prob !== null ? `${Math.round(meta.prob * 100)}%` : "—"
  const predColor = meta.predicted === "Patológica" ? "#c0392b" : "#0e7c8c"
  return `<!doctype html><html lang="es"><head><meta charset="utf-8"/>
<title>${escapeHtml(meta.title)}</title>
<style>
  @page { margin: 18mm 16mm; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; color: #1a1f29; line-height: 1.55; font-size: 12.5px; margin: 0; }
  .hdr { border-bottom: 2px solid #0e7c8c; padding-bottom: 10px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: flex-end; }
  .hdr .kicker { color: #0e7c8c; font-size: 10px; letter-spacing: .18em; text-transform: uppercase; font-weight: 700; }
  .hdr h0 { display:block; font-size: 18px; font-weight: 800; margin: 2px 0 0; }
  .badge { text-align: right; font-size: 11px; color: #5b6675; }
  .badge b { color: ${predColor}; font-size: 13px; }
  h1 { font-size: 16px; color: #0e7c8c; margin: 4px 0 12px; }
  h2 { font-size: 13px; margin: 18px 0 6px; padding-left: 9px; letter-spacing: .02em; }
  p { margin: 6px 0; }
  ul, ol { margin: 6px 0 6px 4px; padding-left: 20px; }
  li { margin: 3px 0; }
  code { background: #eef1f4; border-radius: 3px; padding: 1px 4px; font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .9em; }
  hr { border: none; border-top: 1px solid #dde2e8; margin: 14px 0; }
  strong { color: #0b0f16; }
  .ftr { margin-top: 22px; padding-top: 10px; border-top: 1px solid #dde2e8; font-size: 10px; color: #8a93a0; }
</style></head><body>
  <div class="hdr">
    <div>
      <div class="kicker">Detección de patologías · CNN · MobileNetV2</div>
      <h0>Diagnóstico Radiológico Veterinario</h0>
    </div>
    <div class="badge">
      <div>Predicción modelo: <b>${escapeHtml(meta.predicted ?? "—")} · ${pct}</b></div>
      ${meta.actual ? `<div>Etiqueta real: ${escapeHtml(meta.actual)}</div>` : ""}
      ${meta.label ? `<div>Imagen: ${escapeHtml(meta.label)}</div>` : ""}
      <div>${escapeHtml(meta.date)}</div>
    </div>
  </div>
  <main>${mdToHtml(md)}</main>
  <div class="ftr">Generado por Claude (IA) · Análisis demostrativo — Trabajo Final, Maestría en Ciencia de Datos, CAECE 2026. No reemplaza el criterio de un profesional veterinario.</div>
  <script>window.onload=function(){window.focus();window.print();};window.onafterprint=function(){window.close();};</script>
</body></html>`
}

// ── Mini-renderer de Markdown (negritas, itálicas, títulos, listas, hr) ──────
function renderInline(text: string, keyBase: string): ReactNode[] {
  const nodes: ReactNode[] = []
  const re = /(\*\*[^*]+\*\*|\*[^*\n]+\*|`[^`]+`)/g
  let last = 0, m: RegExpExecArray | null, i = 0
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) nodes.push(text.slice(last, m.index))
    const tok = m[0]
    if (tok.startsWith("**")) {
      nodes.push(<strong key={`${keyBase}-s${i}`} className="font-semibold text-foreground">{tok.slice(2, -2)}</strong>)
    } else if (tok.startsWith("`")) {
      nodes.push(<code key={`${keyBase}-c${i}`} className="rounded bg-secondary px-1 py-0.5 font-mono text-[0.85em] text-hud-cyan">{tok.slice(1, -1)}</code>)
    } else {
      nodes.push(<em key={`${keyBase}-i${i}`} className="italic text-foreground/75">{tok.slice(1, -1)}</em>)
    }
    last = m.index + tok.length
    i++
  }
  if (last < text.length) nodes.push(text.slice(last))
  return nodes
}

function Markdown({ text }: { text: string }) {
  const lines = text.replace(/\r/g, "").split("\n")
  const blocks: ReactNode[] = []
  let list: string[] = []
  let listType: "ul" | "ol" | null = null
  let sectionIdx = 0

  const flushList = (key: string) => {
    if (!list.length) { listType = null; return }
    const items = list
    blocks.push(
      listType === "ol" ? (
        <ol key={key} className="my-2 ml-5 list-decimal space-y-1.5">
          {items.map((it, i) => <li key={i} className="pl-1 leading-relaxed text-foreground/90">{renderInline(it, `${key}-${i}`)}</li>)}
        </ol>
      ) : (
        <ul key={key} className="my-2 ml-5 list-disc space-y-1.5 marker:text-hud-cyan">
          {items.map((it, i) => <li key={i} className="pl-1 leading-relaxed text-foreground/90">{renderInline(it, `${key}-${i}`)}</li>)}
        </ul>
      ),
    )
    list = []; listType = null
  }

  lines.forEach((raw, idx) => {
    const line = raw.trim()
    const key  = `b-${idx}`

    if (/^---+$/.test(line))      { flushList(key); blocks.push(<hr key={key} className="my-3 border-border/70" />); return }
    if (!line)                    { flushList(key); return }

    // Título principal (#)
    const h1 = line.match(/^#\s+(.*)$/)
    if (h1) {
      flushList(key)
      blocks.push(<p key={key} className="mb-3 mt-1 text-base font-bold" style={{ color: "var(--hud-cyan)" }}>{renderInline(h1[1], key)}</p>)
      return
    }
    // Secciones: "## …", "### …" o una línea completa en negrita "**…**"
    const hN       = line.match(/^#{2,4}\s+(.*)$/)
    const boldLine = line.match(/^\*\*(.+?)\*\*:?\s*$/)
    if (hN || boldLine) {
      flushList(key)
      const color = SECTION_COLORS[sectionIdx % SECTION_COLORS.length]
      sectionIdx++
      const title = (hN ? hN[1] : boldLine![1]).replace(/:$/, "")
      blocks.push(
        <p key={key} className="mb-1.5 mt-4 flex items-center gap-2 text-sm font-bold" style={{ color }}>
          <span className="inline-block h-4 w-1 shrink-0 rounded-full" style={{ background: color }} />
          {renderInline(title, key)}
        </p>,
      )
      return
    }

    const ol = line.match(/^\d+\.\s+(.*)$/)
    if (ol) { if (listType === "ul") flushList(key); listType = "ol"; list.push(ol[1]); return }

    const ul = line.match(/^[-*]\s+(.*)$/)
    if (ul) { if (listType === "ol") flushList(key); listType = "ul"; list.push(ul[1]); return }

    flushList(key)
    blocks.push(<p key={key} className="my-1.5 leading-relaxed text-foreground/90">{renderInline(line, key)}</p>)
  })
  flushList("b-end")

  return <div className="text-sm">{blocks}</div>
}

// ════════════════════════════════════════════════════════════════════════════
export function LiveDemoSlide() {
  // View
  const [mode, setMode]             = useState<"inference" | "pipeline">("inference")

  // Current image
  const [imageSrc, setImageSrc]     = useState<string | null>(null)
  const [imageLabel, setImageLabel] = useState("")
  const [overlayUrl, setOverlayUrl] = useState<string | null>(null)
  const [actualLabel, setActualLabel] = useState<ActualLabel | null>(null)
  const [showGradcam, setShowGradcam] = useState(false)

  // Inference
  const [inferRunning, setInferRunning] = useState(false)
  const [prob, setProb]               = useState<number | null>(null)
  const [inferError, setInferError]   = useState<string | null>(null)

  // Grad-CAM dinámico (oclusión) para imágenes subidas
  const [gradcamLoading, setGradcamLoading]   = useState(false)
  const [gradcamProgress, setGradcamProgress] = useState(0)

  // History
  const [history, setHistory]         = useState<HistoryEntry[]>([])

  // Pipeline
  const [pipeRun, setPipeRun]         = useState<PipeRun>("idle")
  const [pipeLog, setPipeLog]         = useState<string[]>([])
  const [pipeStep, setPipeStep]       = useState(-1)

  // Diagnosis modal
  const [diagOpen, setDiagOpen]       = useState(false)
  const [diagLoading, setDiagLoading] = useState(false)
  const [diagText, setDiagText]       = useState("")
  const [diagError, setDiagError]     = useState<string | null>(null)
  const [portalMounted, setPortalMounted] = useState(false)

  // Lightbox (zoom de imagen)
  const [zoomSrc, setZoomSrc]   = useState<string | null>(null)
  const [zoomBig, setZoomBig]   = useState(false)
  function openZoom(src: string | null) { if (src) { setZoomSrc(src); setZoomBig(false) } }

  const esRef     = useRef<EventSource | null>(null)
  const logEndRef = useRef<HTMLDivElement>(null)
  const fileRef   = useRef<HTMLInputElement>(null)
  const diagCache = useRef<Map<string, string>>(new Map())  // diagnóstico cacheado por imagen

  useEffect(() => { logEndRef.current?.scrollIntoView({ behavior: "smooth" }) }, [pipeLog])
  useEffect(() => () => { esRef.current?.close() }, [])
  useEffect(() => { setPortalMounted(true) }, [])
  useEffect(() => {
    if (!zoomSrc) return
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setZoomSrc(null) }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [zoomSrc])

  // Derived
  const predicted = prob !== null ? (prob >= THRESHOLD ? "Patológica" : "Normal") : null
  const correct   = actualLabel && predicted ? predicted === actualLabel : null
  const probPct   = prob !== null ? Math.round(prob * 100) : null
  const done      = prob !== null

  // ── Image selection ──
  function loadCase(i: number) {
    const cs = CASES[i]
    setMode("inference"); setImageSrc(cs.xray); setImageLabel(cs.label)
    setOverlayUrl(cs.overlay); setActualLabel(cs.actual)
    setShowGradcam(false); setProb(null); setInferError(null)
  }

  function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]; if (!file) return
    const name = file.name.length > 22 ? file.name.slice(0, 20) + "…" : file.name
    setMode("inference"); setImageSrc(URL.createObjectURL(file)); setImageLabel(name)
    setOverlayUrl(null); setActualLabel(null)
    setShowGradcam(false); setProb(null); setInferError(null)
    e.target.value = ""
  }

  // ── Inference ──
  async function runAnalysis() {
    if (!imageSrc) return
    setProb(null); setInferError(null); setShowGradcam(false); setInferRunning(true)
    const hadOverlay = !!overlayUrl   // los casos demo ya traen Grad-CAM pre-generado
    try {
      const imgEl  = await loadImg(imageSrc)
      const result = await runInference(imgEl)
      setProb(result)
      setHistory(prev => [{
        id: crypto.randomUUID(), at: new Date(),
        label: imageLabel || "Imagen",
        imageSrc,
        overlayUrl: overlayUrl ?? undefined,
        actual: actualLabel ?? undefined,
        prob: result,
      }, ...prev].slice(0, 12))
      setInferRunning(false)

      // Para imágenes subidas (sin overlay): generamos el mapa de calor por oclusión.
      if (!hadOverlay) {
        setGradcamLoading(true); setGradcamProgress(0)
        try {
          const predClass: ActualLabel = result >= THRESHOLD ? "Patológica" : "Normal"
          const url = await runOcclusion(imgEl, result, predClass, setGradcamProgress)
          setOverlayUrl(url)
        } catch (e) {
          console.error("Grad-CAM por oclusión falló:", e)
        } finally {
          setGradcamLoading(false)
        }
      }
    } catch (err) {
      setInferError("Error al correr el modelo.")
      console.error(err)
      setInferRunning(false)
    }
  }

  // ── History click → show cached result immediately ──
  function selectHistory(entry: HistoryEntry) {
    setMode("inference")
    setImageSrc(entry.imageSrc); setImageLabel(entry.label)
    setOverlayUrl(entry.overlayUrl ?? null); setActualLabel(entry.actual ?? null)
    setShowGradcam(false); setProb(entry.prob); setInferError(null)
  }

  // ── Pipeline ──
  function launchPipeline() {
    if (pipeRun === "running") return
    esRef.current?.close()
    setPipeLog([]); setPipeStep(-1); setPipeRun("running"); setMode("pipeline")

    const es = new EventSource("/api/pipeline-stream")
    esRef.current = es
    es.onmessage = (e) => {
      const line: string = JSON.parse(e.data as string)
      if (line.startsWith("__done__:")) {
        setPipeRun(line === "__done__:ok" ? "ok" : "error"); es.close(); return
      }
      const m = line.match(/^\[(\d+)\/6\]/)
      if (m) setPipeStep(parseInt(m[1]) - 1)
      setPipeLog(prev => [...prev, line])
    }
    es.onerror = () => {
      setPipeLog(prev => [...prev, "[ERROR] Conexión cortada"])
      setPipeRun("error"); es.close()
    }
  }

  function stepStatus(i: number): "pending" | "running" | "ok" | "error" {
    if (pipeRun === "ok") return "ok"
    if (i < pipeStep)  return "ok"
    if (i === pipeStep) return "running"
    return "pending"
  }
  const stepColors = { pending: "#484f58", running: "#388bfd", ok: "#3fb950", error: "#f85149" }

  // ── Diagnosis ──
  async function toBase64(src: string): Promise<{ data: string; mediaType: string }> {
    const res  = await fetch(src)
    const blob = await res.blob()
    const mt   = blob.type || "image/jpeg"
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload  = () => resolve({ data: (reader.result as string).split(",")[1], mediaType: mt })
      reader.onerror = reject
      reader.readAsDataURL(blob)
    })
  }

  async function openDiagnosis() {
    if (!imageSrc || prob === null) return
    setDiagOpen(true); setDiagError(null)

    // Si ya consultamos esta imagen, mostramos el resultado cacheado al instante.
    const cached = diagCache.current.get(imageSrc)
    if (cached) { setDiagText(cached); setDiagLoading(false); return }

    setDiagLoading(true); setDiagText("")
    try {
      const { data, mediaType } = await toBase64(imageSrc)
      const res  = await fetch("/api/diagnose", {
        method:  "POST",
        headers: { "content-type": "application/json" },
        body:    JSON.stringify({ imageBase64: data, mediaType, prob, predicted, actual: actualLabel ?? undefined }),
      })
      const json = await res.json()
      if (!res.ok) throw new Error(json.error || "Error del servidor")
      setDiagText(json.diagnosis)
      diagCache.current.set(imageSrc, json.diagnosis)  // cachear para próximas consultas
    } catch (err: any) {
      setDiagError(err.message)
    } finally {
      setDiagLoading(false)
    }
  }

  function downloadPdf() {
    if (!diagText) return
    const safe  = (imageLabel || "radiografia").replace(/[^\w.-]+/g, "_").slice(0, 40)
    const now   = new Date()
    const html  = buildReportHtml(diagText, {
      title:     `Diagnostico_${safe}`,
      predicted, prob,
      actual:    actualLabel,
      label:     imageLabel || undefined,
      date:      now.toLocaleString("es-AR", { day: "2-digit", month: "2-digit", year: "numeric", hour: "2-digit", minute: "2-digit" }),
    })
    const w = window.open("", "_blank", "width=900,height=1000")
    if (!w) { setDiagError("Permití las ventanas emergentes para descargar el PDF."); return }
    w.document.open(); w.document.write(html); w.document.close()
  }

  // ════════════════════════════════════════════════════════════════════════
  return (
    <>
    <SlideShell kicker="Demo en vivo" title="Predicción en tiempo real · MobileNetV2 + ONNX">
      <div className="flex flex-1 flex-col gap-3">

        {/* ── TOP BAR ───────────────────────────────────────────────────── */}
        <div className="flex shrink-0 items-center gap-2 rounded-lg border border-border bg-card/40 px-3 py-2">

          {/* Pipeline */}
          <button
            onClick={launchPipeline}
            disabled={pipeRun === "running"}
            className={[
              "flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs font-bold transition-colors disabled:opacity-60",
              mode === "pipeline"
                ? "border-hud-amber/60 bg-hud-amber/15 text-hud-amber"
                : "border-hud-amber/40 bg-hud-amber/8 text-hud-amber hover:bg-hud-amber/15",
            ].join(" ")}
          >
            {pipeRun === "running" ? <Loader2 className="size-3 animate-spin" /> : <Play className="size-3" />}
            {pipeRun === "idle"    ? "Ejecutar Pipeline" :
             pipeRun === "running" ? "Ejecutando…"       :
             pipeRun === "ok"      ? "✓ Re-ejecutar"     : "↺ Reintentar"}
          </button>

          {/* Upload */}
          <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={handleUpload} />
          <button onClick={() => fileRef.current?.click()}
            className="flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1.5 text-xs text-muted-foreground transition-colors hover:border-hud-cyan/40 hover:text-hud-cyan">
            <Upload className="size-3" />
            <span className="font-bold">Subir imagen</span>
          </button>

          {/* Model info */}
          <div className="ml-auto flex items-center gap-2 text-[10px] text-muted-foreground">
            <span className="font-mono text-hud-cyan">transfer_mobilenetv2.onnx</span>
            <span className="opacity-40">|</span>
            <span>Umbral <span className="font-mono font-bold text-hud-amber">{THRESHOLD}</span></span>
            <div className="relative h-2 w-20 overflow-hidden rounded-full bg-secondary">
              <div className="h-full w-full rounded-full bg-gradient-to-r from-hud-green via-hud-amber to-hud-red" />
              <div className="absolute top-0 h-full w-0.5 bg-white/80" style={{ left: `${THRESHOLD * 100}%` }} />
            </div>
          </div>
        </div>

        {/* ── MAIN CONTENT ─────────────────────────────────────────────── */}
        {mode === "inference" ? (

          <div className="grid min-h-0 flex-1 overflow-hidden grid-rows-1 grid-cols-[1fr_296px] gap-4">

            {/* Left: X-ray viewer */}
            <Panel className="flex min-h-0 flex-col gap-3">
              <div
                className={["relative overflow-hidden rounded-lg bg-black", imageSrc ? "cursor-zoom-in" : ""].join(" ")}
                style={{ height: 210 }}
                onClick={() => imageSrc && openZoom(showGradcam && overlayUrl ? overlayUrl : imageSrc)}
              >
                {imageSrc ? (
                  <>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={imageSrc} alt="Radiografía"
                      className={["absolute inset-0 h-full w-full object-contain transition-opacity duration-300",
                        showGradcam ? "opacity-0" : "opacity-100"].join(" ")} />
                    {overlayUrl && (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={overlayUrl} alt="Grad-CAM"
                        className={["absolute inset-0 h-full w-full object-contain transition-opacity duration-300",
                          showGradcam ? "opacity-100" : "opacity-0"].join(" ")} />
                    )}
                  </>
                ) : (
                  <div className="flex h-full min-h-[180px] items-center justify-center">
                    <p className="text-sm text-muted-foreground">Seleccioná un caso o subí una imagen</p>
                  </div>
                )}
                {inferRunning && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-background/75 backdrop-blur-sm">
                    <Activity className="size-6 animate-pulse text-hud-cyan" />
                    <span className="text-sm font-semibold text-hud-cyan">Corriendo modelo…</span>
                  </div>
                )}
                {/* Label badge */}
                {imageLabel && (
                  <div className="absolute left-2 top-2 rounded bg-background/80 px-2 py-0.5 font-mono text-[10px] text-hud-cyan backdrop-blur-sm">
                    {imageLabel}
                  </div>
                )}
              </div>

              <div className="flex shrink-0 items-center gap-3">
                <button onClick={runAnalysis} disabled={inferRunning || !imageSrc}
                  className="flex items-center gap-2 rounded-lg border border-hud-cyan/50 bg-hud-cyan/10 px-4 py-1.5 text-sm font-semibold text-hud-cyan transition-colors hover:bg-hud-cyan/20 disabled:opacity-40">
                  <ScanLine className="size-4" />
                  {inferRunning ? "Corriendo…" : "Analizar"}
                </button>
                {done && overlayUrl && (
                  <button onClick={() => setShowGradcam(v => !v)}
                    className={["flex items-center gap-2 rounded-lg border px-4 py-1.5 text-sm font-semibold transition-colors",
                      showGradcam
                        ? "border-hud-amber/50 bg-hud-amber/10 text-hud-amber hover:bg-hud-amber/20"
                        : "border-border text-muted-foreground hover:border-hud-amber/40 hover:text-hud-amber",
                    ].join(" ")}>
                    <Eye className="size-4" />
                    {showGradcam ? "Ver Original" : "Ver Grad-CAM"}
                  </button>
                )}
                {done && (
                  <button onClick={openDiagnosis}
                    className="flex items-center gap-2 rounded-lg border border-hud-cyan/30 bg-hud-cyan/5 px-4 py-1.5 text-sm font-semibold text-hud-cyan transition-colors hover:bg-hud-cyan/15">
                    <Stethoscope className="size-4" />
                    Ver diagnóstico
                  </button>
                )}
                {inferError && <p className="text-xs text-hud-red">{inferError}</p>}
              </div>

              {/* Demo cases — acceso rápido a casos con Grad-CAM */}
              <div className="flex items-center gap-2">
                <span className="text-[10px] text-muted-foreground">Demo:</span>
                {CASES.map((cs, i) => (
                  <button key={cs.label} onClick={() => loadCase(i)}
                    className={[
                      "rounded-md border px-2.5 py-1 text-[10px] font-semibold transition-colors",
                      imageLabel === cs.label
                        ? "border-hud-cyan/60 bg-hud-cyan/10 text-hud-cyan"
                        : "border-border text-muted-foreground hover:border-hud-cyan/40 hover:text-hud-cyan",
                    ].join(" ")}>
                    {cs.label} · {cs.tag}
                  </button>
                ))}
              </div>
            </Panel>

            {/* Right: Grad-CAM + Probability + Verdict */}
            <div className="flex min-h-0 flex-col gap-2 overflow-hidden">

              {/* Grad-CAM image */}
              <Panel className="flex shrink-0 flex-col gap-1.5">
                <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                  Mapa de activación · Grad-CAM
                </p>
                <div className="relative overflow-hidden rounded-md bg-black" style={{ height: 88 }}>
                  {gradcamLoading ? (
                    <div className="flex h-full flex-col items-center justify-center gap-1.5 px-3">
                      <Loader2 className="size-4 animate-spin text-hud-amber" />
                      <p className="text-[10px] font-semibold text-hud-amber">Generando mapa · oclusión</p>
                      <div className="h-1 w-full max-w-[160px] overflow-hidden rounded-full bg-secondary">
                        <div className="h-full rounded-full bg-hud-amber transition-all"
                          style={{ width: `${Math.round(gradcamProgress * 100)}%` }} />
                      </div>
                    </div>
                  ) : overlayUrl && done ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={overlayUrl} alt="Grad-CAM" onClick={() => openZoom(overlayUrl)}
                      className="absolute inset-0 h-full w-full cursor-zoom-in object-contain transition-transform hover:scale-[1.03]" />
                  ) : (
                    <div className="flex h-full items-center justify-center">
                      <p className="whitespace-pre text-center text-[10px] leading-relaxed text-muted-foreground">
                        {done ? "Generando tras el análisis…" : "Disponible tras\nel análisis"}
                      </p>
                    </div>
                  )}
                </div>
              </Panel>

              {/* Probability */}
              <Panel accent={done ? (predicted === "Patológica" ? "red" : "cyan") : undefined} className="shrink-0 !py-3">
                <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                  Probabilidad · clase patológica
                </p>
                <div className="mb-1 flex items-end justify-between">
                  <span className="font-mono text-3xl font-bold"
                    style={{ color: done ? (predicted === "Patológica" ? "var(--hud-red)" : "var(--hud-cyan)") : undefined }}>
                    {done ? `${probPct}%` : "—"}
                  </span>
                  {done && <span className="font-mono text-[10px] text-muted-foreground">p = {prob!.toFixed(3)}</span>}
                </div>
                <div className="h-2 w-full overflow-hidden rounded-full bg-secondary">
                  <div className="h-full rounded-full transition-all duration-700"
                    style={{
                      width: done ? `${probPct}%` : "0%",
                      background: done ? (predicted === "Patológica" ? "var(--hud-red)" : "var(--hud-cyan)") : "var(--hud-cyan)",
                    }} />
                </div>
                <div className="relative mt-0.5 h-3" style={{ marginLeft: `${THRESHOLD * 100}%` }}>
                  <div className="absolute top-0 h-3 w-0.5 bg-hud-amber" />
                  <span className="absolute left-1 top-0.5 text-[9px] text-hud-amber">{THRESHOLD}</span>
                </div>
              </Panel>

              {/* Verdict */}
              {done ? (
                <Panel
                  accent={!actualLabel ? (predicted === "Patológica" ? "red" : "cyan") : correct ? "green" : "red"}
                  glow={correct === false}
                  className="flex flex-1 flex-col gap-2"
                >
                  <div className="flex items-center gap-2">
                    {correct !== false
                      ? <CheckCircle2 className={["size-5", predicted === "Patológica" ? "text-hud-red" : "text-hud-cyan"].join(" ")} />
                      : <AlertTriangle className="size-5 text-hud-red" />}
                    <p className={["text-base font-bold",
                      !actualLabel
                        ? predicted === "Patológica" ? "text-hud-red" : "text-hud-cyan"
                        : correct ? "text-hud-green" : "text-hud-red",
                    ].join(" ")}>
                      {predicted === "Patológica" ? "PATOLÓGICA" : "NORMAL"}
                    </p>
                  </div>
                  {actualLabel && (
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div><p className="text-muted-foreground">Predicción</p><p className="font-semibold">{predicted}</p></div>
                      <div><p className="text-muted-foreground">Real</p><p className="font-semibold">{actualLabel}</p></div>
                    </div>
                  )}
                  <p className="text-[10px] text-muted-foreground">
                    {!actualLabel ? "Predicción real del modelo ONNX."
                      : correct ? "Clasificación correcta."
                      : predicted === "Normal" ? "Falso negativo: patológico no detectado."
                      : "Falso positivo: normal marcado como patológico."}
                  </p>
                </Panel>
              ) : (
                <Panel className="flex flex-1 items-center justify-center">
                  <p className="text-center text-xs text-muted-foreground">
                    {imageSrc
                      ? <><span className="font-semibold text-hud-cyan">Analizar</span><br />para ver el resultado</>
                      : "Seleccioná un caso\no subí una imagen"}
                  </p>
                </Panel>
              )}
            </div>
          </div>

        ) : (

          /* ── Pipeline mode ── */
          <div className="grid min-h-0 flex-1 overflow-hidden grid-rows-1 grid-cols-[1fr_256px] gap-4">
            <Panel className="flex min-h-0 flex-col gap-2">
              <div className="flex shrink-0 items-center gap-2 border-b border-border pb-2">
                <Terminal className="size-3.5 text-hud-amber" />
                <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                  Pipeline Runner · Modo Demo
                </span>
                {pipeRun === "ok"    && <span className="ml-auto text-[10px] font-bold text-hud-green">✓ Completado</span>}
                {pipeRun === "error" && <span className="ml-auto text-[10px] font-bold text-hud-red">✗ Error</span>}
              </div>
              <div className="overflow-y-auto rounded-lg bg-black p-3 font-mono text-[10px] leading-relaxed" style={{ maxHeight: 230 }}>
                {pipeLog.length === 0 && <span className="animate-pulse text-hud-amber">Iniciando pipeline…</span>}
                {pipeLog.map((line, i) => (
                  <div key={i} className={
                    line.startsWith("✓") || line.startsWith("✅") ? "text-green-400" :
                    line.startsWith("[ERROR]") || line.startsWith("❌") || line.startsWith("[ERR") ? "text-red-400" :
                    line.startsWith("▶▶") || line.startsWith("═") ? "text-yellow-400 font-bold" :
                    line.match(/^\[(\d+)\/6\]/) ? "text-hud-cyan font-semibold" :
                    "text-green-300"
                  }>{line}</div>
                ))}
                <div ref={logEndRef} />
              </div>
              <button onClick={() => setMode("inference")}
                className="shrink-0 self-start text-[10px] text-muted-foreground hover:text-foreground">
                ← Volver a Inferencia
              </button>
            </Panel>

            <div className="flex min-h-0 flex-col gap-2 overflow-y-auto">
              <p className="shrink-0 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                Progreso del Pipeline
              </p>
              {STEPS.map((s, i) => {
                const st    = stepStatus(i)
                const color = stepColors[st]
                return (
                  <div key={i} className="flex items-center gap-2 rounded-lg border border-border px-2.5 py-2 transition-colors"
                    style={{ borderLeftColor: color, borderLeftWidth: 3 }}>
                    <span className="text-sm">{s.emoji}</span>
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-[10px] font-semibold" style={{ color }}>{s.title}</p>
                      <p className="text-[9px] text-muted-foreground">
                        {st === "running" ? "ejecutando…" : st === "ok" ? "completado" : st === "error" ? "error" : "pendiente"}
                      </p>
                    </div>
                    {st === "running" && <Loader2 className="size-3 animate-spin" style={{ color }} />}
                    {st === "ok"      && <CheckCircle2 className="size-3 text-hud-green" />}
                    {st === "error"   && <AlertTriangle className="size-3 text-hud-red" />}
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {/* ── BOTTOM HISTORY BAR ───────────────────────────────────────── */}
        <div className="flex shrink-0 items-center gap-2 rounded-lg border border-border bg-card/30 px-3 py-2" style={{ minHeight: 60 }}>
          <div className="flex shrink-0 items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            <Clock className="size-3" />
            Historial
          </div>
          <div className="mx-2 h-4 w-px shrink-0 bg-border" />
          {history.length === 0 ? (
            <p className="text-[10px] text-muted-foreground">Los análisis aparecerán aquí · click para re-cargar</p>
          ) : (
            <div className="flex flex-1 gap-2 overflow-x-auto">
              {history.map(entry => {
                const isPatologica = entry.prob >= THRESHOLD
                return (
                  <button key={entry.id} onClick={() => selectHistory(entry)}
                    className="flex shrink-0 flex-col gap-0.5 rounded-md border border-border px-2.5 py-1.5 text-left transition-colors hover:border-hud-cyan/50 hover:bg-hud-cyan/5">
                    <div className="flex items-center gap-1.5">
                      <span className="max-w-[80px] truncate text-[10px] font-bold text-foreground">{entry.label}</span>
                      <span className={["rounded px-1 text-[8px] font-bold", isPatologica ? "bg-hud-red/15 text-hud-red" : "bg-hud-cyan/15 text-hud-cyan"].join(" ")}>
                        {isPatologica ? "PATO." : "NORM."}
                      </span>
                      <span className="text-[9px] font-mono text-muted-foreground">{Math.round(entry.prob * 100)}%</span>
                    </div>
                    <div className="flex items-center gap-1 text-[9px] text-muted-foreground">
                      <Clock className="size-2.5" />
                      {fmtDate(entry.at)} · {fmtTime(entry.at)}
                    </div>
                  </button>
                )
              })}
            </div>
          )}
        </div>

      </div>
    </SlideShell>

    {/* ── Diagnosis modal (portal: escapa del transform: scale del ScaledSlide) ── */}
    {portalMounted && diagOpen && createPortal(
      <div
        className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/80 p-6 backdrop-blur-sm"
        onClick={(e) => { if (e.target === e.currentTarget) setDiagOpen(false) }}
      >
        <div className="flex h-[72vh] w-full max-w-2xl flex-col rounded-xl border border-border bg-[#0d1117] shadow-2xl">

          {/* Header */}
          <div className="flex shrink-0 items-start justify-between border-b border-border px-6 py-4">
            <div>
              <p className="mb-0.5 text-[11px] font-bold uppercase tracking-[0.2em] text-hud-cyan">
                Análisis IA · Claude
              </p>
              <h3 className="text-lg font-bold text-foreground">Diagnóstico Radiológico</h3>
            </div>
            <div className="flex items-start gap-4">
              {prob !== null && (
                <div className="text-right text-[10px] text-muted-foreground">
                  <p className="font-semibold" style={{ color: predicted === "Patológica" ? "var(--hud-red)" : "var(--hud-cyan)" }}>
                    {predicted} · {Math.round(prob * 100)}%
                  </p>
                  {actualLabel && <p>Real: {actualLabel}</p>}
                </div>
              )}
              <button
                onClick={() => setDiagOpen(false)}
                className="rounded-lg border border-border p-1.5 text-muted-foreground transition-colors hover:border-hud-red/50 hover:text-hud-red"
              >
                <X className="size-4" />
              </button>
            </div>
          </div>

          {/* Body */}
          <div className="min-h-0 flex-1 overflow-y-auto p-6">
            {diagLoading ? (
              <div className="flex h-full flex-col items-center justify-center gap-3">
                <Loader2 className="size-8 animate-spin text-hud-cyan" />
                <p className="text-sm text-muted-foreground">Consultando Claude…</p>
              </div>
            ) : diagError ? (
              <div className="flex h-full items-center justify-center">
                <p className="text-sm text-hud-red">{diagError}</p>
              </div>
            ) : (
              <Markdown text={diagText} />
            )}
          </div>

          {/* Footer */}
          {!diagLoading && !diagError && diagText && (
            <div className="flex shrink-0 items-center justify-between border-t border-border px-6 py-3">
              <p className="text-[10px] text-muted-foreground">
                Generado por Claude · uso académico exclusivo
              </p>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => navigator.clipboard.writeText(diagText)}
                  className="rounded-md border border-border px-3 py-1 text-[10px] text-muted-foreground transition-colors hover:border-hud-cyan/40 hover:text-hud-cyan"
                >
                  Copiar texto
                </button>
                <button
                  onClick={downloadPdf}
                  className="flex items-center gap-1.5 rounded-md border border-hud-cyan/40 bg-hud-cyan/10 px-3 py-1 text-[10px] font-semibold text-hud-cyan transition-colors hover:bg-hud-cyan/20"
                >
                  <Download className="size-3" />
                  Descargar PDF
                </button>
              </div>
            </div>
          )}
        </div>
      </div>,
      document.body,
    )}

    {/* ── Lightbox / zoom de imagen (portal) ── */}
    {portalMounted && zoomSrc && createPortal(
      <div
        className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/90 p-4 backdrop-blur-sm"
        onClick={() => setZoomSrc(null)}
      >
        <button
          onClick={() => setZoomSrc(null)}
          className="absolute right-4 top-4 z-10 rounded-lg border border-border/60 bg-black/40 p-2 text-muted-foreground transition-colors hover:border-hud-red/50 hover:text-hud-red"
        >
          <X className="size-5" />
        </button>
        <span className="pointer-events-none absolute bottom-4 left-1/2 -translate-x-1/2 rounded-md bg-black/50 px-3 py-1 text-[11px] text-muted-foreground">
          Click en la imagen para {zoomBig ? "alejar" : "acercar"} · Esc o fondo para cerrar
        </span>
        <div className="max-h-[92vh] max-w-[94vw] overflow-auto" onClick={(e) => e.stopPropagation()}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={zoomSrc}
            alt="Zoom"
            onClick={() => setZoomBig((v) => !v)}
            style={zoomBig ? { width: "160vw", maxWidth: "none" } : undefined}
            className={[
              "select-none rounded-lg object-contain transition-all duration-200",
              zoomBig ? "cursor-zoom-out" : "max-h-[92vh] max-w-[94vw] cursor-zoom-in",
            ].join(" ")}
          />
        </div>
      </div>,
      document.body,
    )}
    </>
  )
}
