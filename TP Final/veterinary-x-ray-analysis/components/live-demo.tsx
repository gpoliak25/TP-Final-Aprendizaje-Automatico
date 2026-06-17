"use client"

import { useEffect, useRef, useState } from "react"
import { createPortal } from "react-dom"
import Image from "next/image"
import {
  Activity, CheckCircle2, AlertTriangle, ScanLine, Eye,
  Upload, Play, Loader2, Terminal, Clock, Stethoscope, X,
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

  const esRef     = useRef<EventSource | null>(null)
  const logEndRef = useRef<HTMLDivElement>(null)
  const fileRef   = useRef<HTMLInputElement>(null)

  useEffect(() => { logEndRef.current?.scrollIntoView({ behavior: "smooth" }) }, [pipeLog])
  useEffect(() => () => { esRef.current?.close() }, [])
  useEffect(() => { setPortalMounted(true) }, [])

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
    } catch (err) {
      setInferError("Error al correr el modelo.")
      console.error(err)
    } finally {
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
    setDiagOpen(true); setDiagLoading(true); setDiagText(""); setDiagError(null)
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
    } catch (err: any) {
      setDiagError(err.message)
    } finally {
      setDiagLoading(false)
    }
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
              <div className="relative overflow-hidden rounded-lg bg-black" style={{ height: 210 }}>
                {imageSrc ? (
                  <>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={imageSrc} alt="Radiografía"
                      className={["absolute inset-0 h-full w-full object-contain transition-opacity duration-300",
                        showGradcam ? "opacity-0" : "opacity-100"].join(" ")} />
                    {overlayUrl && (
                      <Image src={overlayUrl} alt="Grad-CAM" fill
                        className={["object-contain transition-opacity duration-300",
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
                  {overlayUrl && done ? (
                    <Image src={overlayUrl} alt="Grad-CAM" fill className="object-contain" />
                  ) : (
                    <div className="flex h-full items-center justify-center">
                      <p className="text-center text-[10px] leading-relaxed text-muted-foreground">
                        {done
                          ? "Grad-CAM no disponible\npara imágenes personalizadas"
                          : "Disponible tras\nel análisis"}
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
                {pipeLog.length === 0 && <span className="animate-pulse text-hud-amber">Iniciando Python…</span>}
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
              <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed text-foreground/90">
                {diagText}
              </pre>
            )}
          </div>

          {/* Footer */}
          {!diagLoading && !diagError && diagText && (
            <div className="flex shrink-0 items-center justify-between border-t border-border px-6 py-3">
              <p className="text-[10px] text-muted-foreground">
                Generado por Claude · uso académico exclusivo
              </p>
              <button
                onClick={() => navigator.clipboard.writeText(diagText)}
                className="rounded-md border border-border px-3 py-1 text-[10px] text-muted-foreground transition-colors hover:border-hud-cyan/40 hover:text-hud-cyan"
              >
                Copiar texto
              </button>
            </div>
          )}
        </div>
      </div>,
      document.body,
    )}
    </>
  )
}
