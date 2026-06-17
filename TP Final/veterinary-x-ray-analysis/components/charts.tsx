import { cn } from "@/lib/utils"

const CYAN = "oklch(0.82 0.13 200)"
const AMBER = "oklch(0.8 0.14 68)"
const RED = "oklch(0.64 0.2 24)"
const GRID = "oklch(0.6 0.04 220 / 0.18)"
const AXIS = "oklch(0.7 0.02 240 / 0.6)"

/* ---------------- Donut ---------------- */

export function DonutChart({
  normal,
  pathological,
}: {
  normal: number
  pathological: number
}) {
  const total = normal + pathological
  const r = 70
  const c = 2 * Math.PI * r
  const normalFrac = normal / total
  const stroke = 26
  return (
    <svg viewBox="0 0 200 200" className="h-full w-full" role="img" aria-label="Distribución de clases">
      <circle cx="100" cy="100" r={r} fill="none" stroke="oklch(0.3 0.02 248)" strokeWidth={stroke} />
      {/* normal */}
      <circle
        cx="100"
        cy="100"
        r={r}
        fill="none"
        stroke={CYAN}
        strokeWidth={stroke}
        strokeDasharray={`${c * normalFrac} ${c}`}
        transform="rotate(-90 100 100)"
        strokeLinecap="butt"
      />
      {/* pathological */}
      <circle
        cx="100"
        cy="100"
        r={r}
        fill="none"
        stroke={AMBER}
        strokeWidth={stroke}
        strokeDasharray={`${c * (1 - normalFrac)} ${c}`}
        strokeDashoffset={`${-c * normalFrac}`}
        transform="rotate(-90 100 100)"
        strokeLinecap="butt"
      />
      <text x="100" y="94" textAnchor="middle" className="fill-hud-cyan font-mono" fontSize="30" fontWeight="700">
        {total}
      </text>
      <text x="100" y="116" textAnchor="middle" className="fill-muted-foreground" fontSize="11">
        Imágenes
      </text>
    </svg>
  )
}

/* ---------------- Confusion matrix ---------------- */

export function ConfusionMatrix({
  matrix,
  intent = "cyan",
}: {
  // [[TN, FP],[FN, TP]] in label order ok/patologica
  matrix: [[number, number], [number, number]]
  intent?: "cyan" | "fail"
}) {
  const max = Math.max(...matrix.flat(), 1)
  const labels = ["ok", "patologica"]
  const cell = (v: number, correct: boolean) => {
    const intensity = 0.12 + (v / max) * 0.6
    const base = intent === "fail" ? "200" : correct ? "200" : "200"
    return `oklch(0.7 0.12 ${base} / ${intensity})`
  }
  return (
    <div className="flex w-full flex-col">
      <div className="grid grid-cols-[auto_1fr_1fr] gap-1.5">
        <div />
        <div className="pb-1 text-center text-xs font-medium text-muted-foreground">ok</div>
        <div className="pb-1 text-center text-xs font-medium text-muted-foreground">patologica</div>
        {matrix.map((row, i) =>
          [
            <div
              key={`lbl-${i}`}
              className="flex items-center justify-end pr-2 text-xs font-medium text-muted-foreground"
            >
              {labels[i]}
            </div>,
            ...row.map((v, j) => {
              const correct = i === j
              return (
                <div
                  key={`c-${i}-${j}`}
                  className={cn(
                    "flex aspect-[2/1] items-center justify-center rounded-md border font-mono text-2xl font-bold tabular-nums md:text-3xl",
                    correct ? "border-hud-cyan/40 text-foreground" : "border-border text-muted-foreground",
                  )}
                  style={{ backgroundColor: cell(v, correct) }}
                >
                  {v}
                </div>
              )
            }),
          ],
        )}
      </div>
      <div className="mt-2 grid grid-cols-[auto_1fr] gap-1.5">
        <div />
        <div className="text-center text-xs text-muted-foreground">Predicción</div>
      </div>
    </div>
  )
}

/* ---------------- Line chart ---------------- */

export type Series = { label: string; color: string; points: number[] }

export function LineChart({
  series,
  xMax,
  yMin = 0,
  yMax = 1,
  yLabel,
  marker,
  height = 200,
}: {
  series: Series[]
  xMax: number
  yMin?: number
  yMax?: number
  yLabel?: string
  marker?: { x: number; label: string }
  height?: number
}) {
  const w = 320
  const h = height
  const padL = 34
  const padB = 22
  const padT = 10
  const padR = 10
  const plotW = w - padL - padR
  const plotH = h - padT - padB
  const sx = (i: number, n: number) => padL + (i / (n - 1)) * plotW
  const sy = (v: number) => padT + (1 - (v - yMin) / (yMax - yMin)) * plotH
  const yTicks = 5
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="h-full w-full" role="img" aria-label={yLabel ?? "Gráfico"}>
      {Array.from({ length: yTicks + 1 }).map((_, i) => {
        const v = yMin + ((yMax - yMin) * i) / yTicks
        const y = sy(v)
        return (
          <g key={i}>
            <line x1={padL} y1={y} x2={w - padR} y2={y} stroke={GRID} strokeWidth="0.6" />
            <text x={padL - 4} y={y + 3} textAnchor="end" fontSize="7" fill={AXIS}>
              {v.toFixed(1)}
            </text>
          </g>
        )
      })}
      <line x1={padL} y1={padT} x2={padL} y2={h - padB} stroke={AXIS} strokeWidth="0.8" />
      <line x1={padL} y1={h - padB} x2={w - padR} y2={h - padB} stroke={AXIS} strokeWidth="0.8" />
      {marker ? (
        <g>
          <line
            x1={padL + (marker.x / xMax) * plotW}
            y1={padT}
            x2={padL + (marker.x / xMax) * plotW}
            y2={h - padB}
            stroke={CYAN}
            strokeWidth="1"
            strokeDasharray="3 3"
          />
        </g>
      ) : null}
      {series.map((s) => {
        const n = s.points.length
        const d = s.points
          .map((p, i) => `${i === 0 ? "M" : "L"} ${sx(i, n).toFixed(1)} ${sy(p).toFixed(1)}`)
          .join(" ")
        return <path key={s.label} d={d} fill="none" stroke={s.color} strokeWidth="1.6" strokeLinejoin="round" />
      })}
      <text x={w - padR} y={h - 4} textAnchor="end" fontSize="7" fill={AXIS}>
        época
      </text>
    </svg>
  )
}

export function Legend({ series }: { series: { label: string; color: string; dashed?: boolean }[] }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
      {series.map((s) => (
        <div key={s.label} className="flex items-center gap-1.5">
          <span
            className="inline-block h-0.5 w-4 rounded"
            style={{
              backgroundColor: s.dashed ? "transparent" : s.color,
              borderTop: s.dashed ? `2px dashed ${s.color}` : undefined,
            }}
          />
          <span className="text-xs text-muted-foreground">{s.label}</span>
        </div>
      ))}
    </div>
  )
}

/* ---------------- ROC curve ---------------- */

export function RocCurve({
  curves,
}: {
  curves: { label: string; color: string; points: [number, number][] }[]
}) {
  const w = 300
  const h = 260
  const padL = 36
  const padB = 28
  const pad = 12
  const plotW = w - padL - pad
  const plotH = h - pad - padB
  const sx = (v: number) => padL + v * plotW
  const sy = (v: number) => pad + (1 - v) * plotH
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="h-full w-full" role="img" aria-label="Curva ROC">
      {[0, 0.2, 0.4, 0.6, 0.8, 1].map((t) => (
        <g key={t}>
          <line x1={sx(t)} y1={sy(0)} x2={sx(t)} y2={sy(1)} stroke={GRID} strokeWidth="0.5" />
          <line x1={sx(0)} y1={sy(t)} x2={sx(1)} y2={sy(t)} stroke={GRID} strokeWidth="0.5" />
          <text x={sx(t)} y={h - padB + 12} textAnchor="middle" fontSize="8" fill={AXIS}>
            {t.toFixed(1)}
          </text>
          <text x={padL - 5} y={sy(t) + 3} textAnchor="end" fontSize="8" fill={AXIS}>
            {t.toFixed(1)}
          </text>
        </g>
      ))}
      <line x1={sx(0)} y1={sy(0)} x2={sx(1)} y2={sy(1)} stroke={AXIS} strokeWidth="0.8" strokeDasharray="4 3" />
      <line x1={padL} y1={pad} x2={padL} y2={sy(0)} stroke={AXIS} strokeWidth="0.8" />
      <line x1={padL} y1={sy(0)} x2={sx(1)} y2={sy(0)} stroke={AXIS} strokeWidth="0.8" />
      {curves.map((cv) => {
        const d = cv.points.map((p, i) => `${i === 0 ? "M" : "L"} ${sx(p[0]).toFixed(1)} ${sy(p[1]).toFixed(1)}`).join(" ")
        return <path key={cv.label} d={d} fill="none" stroke={cv.color} strokeWidth="2" strokeLinejoin="round" />
      })}
      <text
        x={12}
        y={h / 2}
        textAnchor="middle"
        fontSize="8"
        fill={AXIS}
        transform={`rotate(-90 12 ${h / 2})`}
      >
        Verdaderos positivos
      </text>
      <text x={w / 2 + 8} y={h - 4} textAnchor="middle" fontSize="8" fill={AXIS}>
        Falsos positivos
      </text>
    </svg>
  )
}

export { CYAN, AMBER, RED }
