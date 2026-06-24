import { LiveDemoSlide } from "@/components/live-demo"
import Image from "next/image"
import {
  Activity,
  AlertTriangle,
  Brain,
  CheckCircle2,
  Eye,
  Layers,
  Microscope,
  ScanLine,
  Stethoscope,
  Target,
  TrendingUp,
  XCircle,
  Boxes,
  GitBranch,
} from "lucide-react"
import { SlideShell, Panel, Stat, Callout } from "@/components/hud"
import {
  DonutChart,
  ConfusionMatrix,
  LineChart,
  Legend,
  RocCurve,
  CYAN,
  AMBER,
  RED,
} from "@/components/charts"

export interface SlideMeta {
  id: number
  title: string
  render: () => React.ReactNode
}

/* ------------------------------------------------------------------ */
/* 1 — Portada                                                         */
/* ------------------------------------------------------------------ */
function Cover() {
  return (
    <div className="relative h-full w-full overflow-hidden">
      <Image
        src="/xray-hero.png"
        alt="Radiografía de tórax canina con mapa de calor Grad-CAM sobre la región torácica central"
        fill
        priority
        className="object-cover opacity-35"
      />
      <div className="absolute inset-0 bg-gradient-to-r from-background via-background/85 to-background/40" />
      <div className="hud-grid absolute inset-0 opacity-60" />
      <div className="relative flex h-full flex-col justify-center px-6 py-10 md:px-16">
        <p className="mb-4 inline-flex w-fit items-center gap-2 rounded-full border border-hud-cyan/40 bg-card/60 px-3 py-1 text-xs font-semibold uppercase tracking-[0.25em] text-hud-cyan backdrop-blur">
          <ScanLine className="size-3.5" /> Trabajo Práctico Integrador
        </p>
        <h1 className="text-balance text-3xl font-bold leading-[1.05] tracking-tight md:text-5xl lg:text-6xl">
          Detección de patologías en radiografías de tórax veterinarias
        </h1>
        <p className="mt-3 max-w-2xl text-pretty text-base text-muted-foreground md:text-xl">
          Mediante redes neuronales convolucionales — un análisis del costo del error y la interpretabilidad clínica.
        </p>
        <div className="mt-8 flex flex-wrap items-center gap-x-8 gap-y-3">
          <div>
            <p className="text-xs uppercase tracking-wider text-muted-foreground">Autoras</p>
            <p className="text-base font-semibold md:text-lg">Lorena López · Gisela Poliak</p>
          </div>
          <div className="h-10 w-px bg-border" />
          <div>
            <p className="text-xs uppercase tracking-wider text-muted-foreground">Institución</p>
            <p className="text-base font-semibold md:text-lg">CAECE · Junio 2026</p>
          </div>
        </div>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* 2 — Desafío clínico                                                 */
/* ------------------------------------------------------------------ */
function ClinicalChallenge() {
  return (
    <SlideShell kicker="El problema real" title="El desafío clínico: fatiga, variabilidad y la necesidad de triage">
      <div className="my-auto grid grid-cols-2 items-stretch gap-5">
        <Panel className="flex flex-col">
          <div className="mb-3 flex items-center gap-2 text-muted-foreground">
            <Stethoscope className="size-5" />
            <h3 className="text-sm font-semibold uppercase tracking-wider">Flujo tradicional</h3>
          </div>
          <ul className="flex flex-col gap-3 text-sm leading-relaxed md:text-base">
            {[
              "Lectura manual de cada placa, una por una",
              "Fatiga acumulada en jornadas largas",
              "Variabilidad inter-observador en casos sutiles",
              "Sin priorización: todo entra en la misma cola",
            ].map((t) => (
              <li key={t} className="flex items-start gap-2.5 text-muted-foreground">
                <XCircle className="mt-0.5 size-4 shrink-0 text-hud-red" />
                <span>{t}</span>
              </li>
            ))}
          </ul>
        </Panel>
        <Panel accent="cyan" glow className="flex flex-col">
          <div className="mb-3 flex items-center gap-2 text-hud-cyan">
            <Brain className="size-5" />
            <h3 className="text-sm font-semibold uppercase tracking-wider">Flujo potenciado por IA</h3>
          </div>
          <ul className="flex flex-col gap-3 text-sm leading-relaxed md:text-base">
            {[
              "Filtro de triage automático previo a la lectura",
              "Prioriza los casos con sospecha de patología",
              "Criterio reproducible y constante",
              "El profesional decide; la IA acelera y ordena",
            ].map((t) => (
              <li key={t} className="flex items-start gap-2.5">
                <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-hud-cyan" />
                <span>{t}</span>
              </li>
            ))}
          </ul>
        </Panel>
      </div>
      <p className="mt-5 text-pretty text-sm text-muted-foreground md:text-base">
        El objetivo no es reemplazar al veterinario, sino construir un <span className="text-hud-cyan">filtro de triage</span>{" "}
        que reduzca la carga y ordene la cola de lectura.
      </p>
    </SlideShell>
  )
}

/* ------------------------------------------------------------------ */
/* 3 — Espacio de datos                                                */
/* ------------------------------------------------------------------ */
function DataSpace() {
  return (
    <SlideShell kicker="¿Con qué datos contamos?" title="El espacio de datos: restricciones y distribución">
      <div className="my-auto grid grid-cols-[280px_1fr] items-stretch gap-6">
        <Panel className="flex flex-col items-center justify-center">
          <div className="h-44 w-44">
            <DonutChart normal={245} pathological={196} />
          </div>
          <div className="mt-4 flex w-full flex-col gap-2">
            <div className="flex items-center justify-between text-sm">
              <span className="flex items-center gap-2">
                <span className="inline-block size-3 rounded-sm bg-hud-cyan" /> Normales
              </span>
              <span className="font-mono text-muted-foreground">245 · 55,6%</span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="flex items-center gap-2">
                <span className="inline-block size-3 rounded-sm bg-hud-amber" /> Patológicas
              </span>
              <span className="font-mono text-muted-foreground">196 · 44,4%</span>
            </div>
          </div>
        </Panel>

        <div className="flex flex-col gap-5">
          <Panel accent="cyan">
            <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-hud-cyan">
              Partición 70 / 15 / 15
            </h3>
            <div className="flex h-9 w-full overflow-hidden rounded-md border border-border font-mono text-xs font-semibold text-background">
              <div className="flex items-center justify-center bg-hud-cyan" style={{ width: "70%" }}>
                Train · 308
              </div>
              <div className="flex items-center justify-center bg-hud-amber" style={{ width: "15%" }}>
                66
              </div>
              <div className="flex items-center justify-center bg-hud-green" style={{ width: "15%" }}>
                67
              </div>
            </div>
            <div className="mt-2 flex justify-between text-xs text-muted-foreground">
              <span>Entrenamiento</span>
              <span>Validación · 66</span>
              <span>Test · 67</span>
            </div>
          </Panel>
          <div className="grid grid-cols-3 gap-4">
            <Panel>
              <Stat label="Total" value="441" sub="imágenes" />
            </Panel>
            <Panel>
              <Stat label="Augmentation" value="Sí" sub="rotación, zoom, flip" accent="amber" />
            </Panel>
            <Panel>
              <Stat label="Origen" value="Homogéneo" sub="misma fuente clínica" accent="green" />
            </Panel>
          </div>
          <p className="text-pretty text-sm text-muted-foreground">
            Dataset pequeño y homogéneo: una restricción clave que condiciona toda la estrategia de modelado.
          </p>
        </div>
      </div>
    </SlideShell>
  )
}

/* ------------------------------------------------------------------ */
/* 4 — Métrica estrella: costo del error                               */
/* ------------------------------------------------------------------ */
function CostMetric() {
  return (
    <SlideShell
      kicker="¿Qué significa equivocarse?"
      title="La métrica estrella: el costo asimétrico del error"
      accent="amber"
    >
      <div className="my-auto grid grid-cols-[1fr_320px] items-stretch gap-6">
        <Panel accent="amber" className="flex flex-col justify-center">
          <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-hud-amber">
            Matriz de costo clínico
          </h3>
          <div className="grid grid-cols-[auto_1fr_1fr] gap-2 text-center">
            <div />
            <div className="pb-1 text-xs font-medium text-muted-foreground">Predice sano</div>
            <div className="pb-1 text-xs font-medium text-muted-foreground">Predice enfermo</div>

            <div className="flex items-center justify-end pr-2 text-xs font-medium text-muted-foreground">
              Sano real
            </div>
            <div className="rounded-md border border-hud-green/40 bg-hud-green/10 p-3">
              <p className="text-sm font-bold text-hud-green">Correcto</p>
              <p className="text-xs text-muted-foreground">Verdadero negativo</p>
            </div>
            <div className="rounded-md border border-hud-amber/40 bg-hud-amber/10 p-3">
              <p className="text-sm font-bold text-hud-amber">Falsa alarma</p>
              <p className="text-xs text-muted-foreground">Costo moderado</p>
            </div>

            <div className="flex items-center justify-end pr-2 text-xs font-medium text-muted-foreground">
              Enfermo real
            </div>
            <div className="rounded-md border border-hud-red/50 bg-hud-red/15 p-3 hud-glow-red">
              <p className="text-sm font-bold text-hud-red">Error crítico</p>
              <p className="text-xs text-muted-foreground">Falso negativo</p>
            </div>
            <div className="rounded-md border border-hud-cyan/40 bg-hud-cyan/10 p-3">
              <p className="text-sm font-bold text-hud-cyan">Correcto</p>
              <p className="text-xs text-muted-foreground">Verdadero positivo</p>
            </div>
          </div>
        </Panel>
        <div className="flex flex-col gap-4">
          <Panel accent="red" glow>
            <div className="mb-2 flex items-center gap-2 text-hud-red">
              <AlertTriangle className="size-5" />
              <h3 className="text-base font-bold">El peor error</h3>
            </div>
            <p className="text-sm leading-relaxed text-muted-foreground">
              Un <span className="text-hud-red">falso negativo</span> deja sin diagnóstico a un paciente enfermo. Cuesta
              mucho más que una falsa alarma.
            </p>
          </Panel>
          <Panel accent="cyan">
            <div className="mb-2 flex items-center gap-2 text-hud-cyan">
              <Target className="size-5" />
              <h3 className="text-base font-bold">Objetivo clínico</h3>
            </div>
            <p className="text-sm leading-relaxed text-muted-foreground">
              Maximizar la <span className="text-hud-cyan">sensibilidad (recall)</span>: detectar a la mayor cantidad
              posible de pacientes enfermos.
            </p>
          </Panel>
        </div>
      </div>
    </SlideShell>
  )
}

/* ------------------------------------------------------------------ */
/* 5 — Dos enfoques                                                    */
/* ------------------------------------------------------------------ */
function TwoApproaches() {
  return (
    <SlideShell
      kicker="Estrategia de modelado"
      title="El enfrentamiento analítico: dos enfoques frente a la escasez de datos"
    >
      <div className="my-auto grid grid-cols-2 items-stretch gap-5">
        <Panel className="flex flex-col">
          <span className="mb-2 inline-flex w-fit items-center gap-2 rounded-full border border-border px-2.5 py-0.5 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            <Boxes className="size-3.5" /> Enfoque 1
          </span>
          <h3 className="text-xl font-bold md:text-2xl">CNN desde cero</h3>
          <p className="mt-1 text-sm text-muted-foreground">La línea de base construida íntegramente a mano.</p>
          <div className="mt-4">
            <Stat label="Parámetros entrenables" value="~423.000" accent="amber" />
          </div>
          <ul className="mt-4 flex flex-col gap-2 text-sm text-muted-foreground">
            <li className="flex items-center gap-2">
              <Layers className="size-4 text-muted-foreground" /> Capas convolucionales + pooling propias
            </li>
            <li className="flex items-center gap-2">
              <Layers className="size-4 text-muted-foreground" /> Todo el peso aprendido desde 308 imágenes
            </li>
          </ul>
        </Panel>
        <Panel accent="cyan" glow className="flex flex-col">
          <span className="mb-2 inline-flex w-fit items-center gap-2 rounded-full border border-hud-cyan/40 px-2.5 py-0.5 text-xs font-semibold uppercase tracking-wider text-hud-cyan">
            <Brain className="size-3.5" /> Enfoque 2
          </span>
          <h3 className="text-xl font-bold md:text-2xl">Transfer Learning · MobileNetV2</h3>
          <p className="mt-1 text-sm text-muted-foreground">Conocimiento preentrenado adaptado al dominio.</p>
          <div className="mt-4 grid grid-cols-2 gap-3">
            <div className="rounded-md border border-border bg-background/40 p-3">
              <p className="text-xs uppercase tracking-wider text-muted-foreground">Fase 1</p>
              <p className="mt-1 text-sm font-semibold text-hud-cyan">Base congelada + cabezal propio</p>
            </div>
            <div className="rounded-md border border-border bg-background/40 p-3">
              <p className="text-xs uppercase tracking-wider text-muted-foreground">Fase 2</p>
              <p className="mt-1 text-sm font-semibold text-hud-cyan">Fine-tuning ajustado</p>
            </div>
          </div>
          <p className="mt-4 text-sm leading-relaxed text-muted-foreground">
            Aprovecha millones de imágenes previas para compensar la escasez de datos propios.
          </p>
        </Panel>
      </div>
    </SlideShell>
  )
}

/* ------------------------------------------------------------------ */
/* 6 — Resultados enfoque 1                                            */
/* ------------------------------------------------------------------ */
function ResultsApproach1() {
  // illustrative training curves showing overfitting
  const trainLoss = [0.69, 0.62, 0.55, 0.48, 0.42, 0.37, 0.32, 0.28, 0.24, 0.21]
  const valLoss = [0.7, 0.69, 0.72, 0.78, 0.85, 0.93, 1.02, 1.12, 1.24, 1.37]
  const trainAcc = [0.52, 0.58, 0.64, 0.69, 0.74, 0.79, 0.83, 0.87, 0.9, 0.93]
  const valAcc = [0.45, 0.46, 0.44, 0.45, 0.43, 0.44, 0.45, 0.43, 0.44, 0.44]
  return (
    <SlideShell kicker="Enfoque 1 · línea de base" title="Resultados: la ilusión de aprender" accent="amber">
      <div className="my-auto grid grid-cols-[1fr_1fr_300px] items-stretch gap-5">
        <Panel className="flex flex-col">
          <h3 className="mb-1 text-xs font-semibold uppercase tracking-wider text-muted-foreground">Pérdida</h3>
          <div className="min-h-0 flex-1">
            <LineChart
              series={[
                { label: "train_loss", color: CYAN, points: trainLoss },
                { label: "val_loss", color: RED, points: valLoss },
              ]}
              xMax={10}
              yMin={0}
              yMax={1.5}
              height={160}
            />
          </div>
          <Legend
            series={[
              { label: "train_loss", color: CYAN },
              { label: "val_loss ▲ desde época 2", color: RED },
            ]}
          />
        </Panel>
        <Panel className="flex flex-col">
          <h3 className="mb-1 text-xs font-semibold uppercase tracking-wider text-muted-foreground">Accuracy</h3>
          <div className="min-h-0 flex-1">
            <LineChart
              series={[
                { label: "train_acc", color: CYAN, points: trainAcc },
                { label: "val_acc", color: AMBER, points: valAcc },
              ]}
              xMax={10}
              yMin={0}
              yMax={1}
              height={160}
            />
          </div>
          <Legend
            series={[
              { label: "train_acc", color: CYAN },
              { label: "val_acc ≈ 0,44 estancada", color: AMBER },
            ]}
          />
        </Panel>
        <div className="flex flex-col gap-4">
          <Panel accent="red" glow>
            <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-hud-red">
              Matriz de confusión · CNN
            </h3>
            <ConfusionMatrix matrix={[[0, 37], [0, 30]]} intent="fail" />
          </Panel>
          <Panel accent="red">
            <div className="flex items-center justify-between">
              <Stat label="AUC" value="0,446" accent="red" />
              <span className="rounded-md border border-hud-red/40 bg-hud-red/10 px-2.5 py-1 text-xs font-semibold text-hud-red">
                Ilusión matemática
              </span>
            </div>
            <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
              El modelo colapsa: predice siempre la misma clase. La accuracy de train sube, pero no generaliza.
            </p>
          </Panel>
        </div>
      </div>
    </SlideShell>
  )
}

/* ------------------------------------------------------------------ */
/* 7 — Resultados enfoque 2                                            */
/* ------------------------------------------------------------------ */
function ResultsApproach2() {
  const trainLoss = [0.68, 0.5, 0.4, 0.34, 0.3, 0.27, 0.22, 0.18, 0.15, 0.12]
  const valLoss = [0.66, 0.52, 0.45, 0.42, 0.4, 0.39, 0.34, 0.31, 0.29, 0.28]
  return (
    <SlideShell kicker="Enfoque 2 · transfer learning" title="Resultados: el poder de la transferencia">
      <div className="my-auto grid grid-cols-[1.3fr_300px] items-stretch gap-4">
        <Panel accent="cyan" className="flex flex-col">
          <h3 className="mb-1 text-xs font-semibold uppercase tracking-wider text-hud-cyan">
            Pérdida · dos fases de entrenamiento
          </h3>
          <div className="relative min-h-0 flex-1">
            <div className="absolute inset-0">
              <LineChart
                series={[
                  { label: "train_loss", color: CYAN, points: trainLoss },
                  { label: "val_loss", color: AMBER, points: valLoss },
                ]}
                xMax={10}
                yMin={0}
                yMax={0.8}
                marker={{ x: 5, label: "fine-tuning" }}
                height={170}
              />
            </div>
          </div>
          <div className="mt-1 flex items-center justify-between">
            <Legend
              series={[
                { label: "train_loss", color: CYAN },
                { label: "val_loss", color: AMBER },
              ]}
            />
            <span className="text-xs text-muted-foreground">┊ inicio de fine-tuning</span>
          </div>
        </Panel>
        <div className="flex flex-col gap-4">
          <Panel accent="green" glow>
            <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-hud-green">
              Matriz de confusión · Transfer
            </h3>
            <ConfusionMatrix matrix={[[31, 6], [7, 23]]} />
          </Panel>
          <Panel accent="green">
            <div className="flex items-center justify-between">
              <Stat label="AUC" value="0,852" accent="green" trend="up" />
              <span className="rounded-md border border-hud-green/40 bg-hud-green/10 px-2.5 py-1 text-xs font-semibold text-hud-green">
                Aprendizaje real
              </span>
            </div>
            <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
              La validación acompaña al entrenamiento: el modelo generaliza y distingue ambas clases.
            </p>
          </Panel>
        </div>
      </div>
    </SlideShell>
  )
}

/* ------------------------------------------------------------------ */
/* 8 — Síntesis de desempeño                                           */
/* ------------------------------------------------------------------ */
function Synthesis() {
  const rows = [
    { metric: "Accuracy", cnn: "0,448", tl: "0,806" },
    { metric: "Precision", cnn: "—", tl: "0,793" },
    { metric: "Recall", cnn: "—", tl: "0,767" },
    { metric: "AUC", cnn: "0,446", tl: "0,852" },
  ]
  // ROC: CNN near diagonal, Transfer bowed toward top-left
  const cnnRoc: [number, number][] = [
    [0, 0], [0.2, 0.18], [0.4, 0.39], [0.6, 0.58], [0.8, 0.82], [1, 1],
  ]
  const tlRoc: [number, number][] = [
    [0, 0], [0.05, 0.4], [0.12, 0.62], [0.25, 0.78], [0.45, 0.9], [0.7, 0.97], [1, 1],
  ]
  return (
    <SlideShell kicker="Comparación final" title="Síntesis de desempeño: Transfer Learning triunfa">
      <div className="grid flex-1 grid-cols-[1fr_320px] items-center gap-6">
        <Panel accent="cyan">
          <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-hud-cyan">
            Métricas en conjunto de test
          </h3>
          <table className="w-full border-collapse text-left">
            <thead>
              <tr className="border-b border-border text-xs uppercase tracking-wider text-muted-foreground">
                <th className="pb-2 font-medium">Métrica</th>
                <th className="pb-2 text-right font-medium">CNN scratch</th>
                <th className="pb-2 text-right font-medium text-hud-cyan">Transfer Learning</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {rows.map((r) => (
                <tr key={r.metric} className="border-b border-border/50">
                  <td className="py-2.5 font-sans text-sm font-medium">{r.metric}</td>
                  <td className="py-2.5 text-right text-sm text-muted-foreground/70">{r.cnn}</td>
                  <td className="py-2.5 text-right text-base font-bold text-hud-cyan">{r.tl}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-4 text-xs text-muted-foreground">
            CNN atenuada por su colapso; el transfer learning domina en todas las métricas.
          </p>
        </Panel>
        <Panel>
          <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-muted-foreground">Curva ROC</h3>
          <div className="h-56">
            <RocCurve
              curves={[
                { label: "CNN", color: "oklch(0.6 0.02 240)", points: cnnRoc },
                { label: "Transfer", color: CYAN, points: tlRoc },
              ]}
            />
          </div>
          <Legend
            series={[
              { label: "CNN · 0,446", color: "oklch(0.6 0.02 240)" },
              { label: "Transfer · 0,852", color: CYAN },
            ]}
          />
        </Panel>
      </div>
    </SlideShell>
  )
}

/* ------------------------------------------------------------------ */
/* 9 — Calibrando el umbral                                            */
/* ------------------------------------------------------------------ */
function Threshold() {
  return (
    <SlideShell
      kicker="Aplicando el costo del error"
      title="Calibrando la decisión: priorizando al paciente"
      accent="amber"
    >
      <div className="grid flex-1 grid-cols-2 items-center gap-6">
        <div className="flex flex-col gap-5">
          <Panel className="flex flex-col">
            <div className="mb-4 flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Umbral de decisión
              </span>
              <span className="font-mono text-sm text-hud-amber">0,50 → 0,30</span>
            </div>
            {/* track */}
            <div className="relative h-2 w-full rounded-full bg-secondary">
              <div className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-hud-amber to-hud-cyan" style={{ width: "60%" }} />
              <div className="absolute top-1/2 size-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-hud-amber bg-background hud-glow-amber" style={{ left: "30%" }} />
              <div className="absolute top-1/2 size-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-hud-cyan bg-background" style={{ left: "50%" }} />
            </div>
            <div className="mt-2 flex justify-between font-mono text-xs text-muted-foreground">
              <span>0,0</span>
              <span className="text-hud-amber">0,30</span>
              <span className="text-hud-cyan">0,50</span>
              <span>1,0</span>
            </div>
          </Panel>

          <div className="grid grid-cols-2 gap-4">
            <Panel accent="cyan">
              <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-hud-cyan">Umbral 0,50</p>
              <div className="flex flex-col gap-3">
                <Stat label="Recall" value="0,767" accent="cyan" />
                <Stat label="Precision" value="0,793" accent="cyan" />
              </div>
            </Panel>
            <Panel accent="amber" glow>
              <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-hud-amber">Umbral 0,30</p>
              <div className="flex flex-col gap-3">
                <Stat label="Recall" value="0,867" accent="amber" trend="up" />
                <Stat label="Precision" value="0,650" accent="amber" trend="down" />
              </div>
            </Panel>
          </div>
        </div>

        <div className="flex flex-col gap-4">
          <Panel>
            <Callout title="Mecánica del umbral" accent="cyan">
              Bajar el umbral hace al modelo más sensible: clasifica como “patológico” ante menos evidencia, capturando
              más casos enfermos a costa de más falsas alarmas.
            </Callout>
          </Panel>
          <Panel accent="amber">
            <Callout title="Impacto clínico" accent="amber">
              En triage preferimos errar por exceso de cautela. A 0,30 ganamos recall de forma significativa sin que la
              precisión se desplome: es el punto de operación elegido según la prioridad clínica.
            </Callout>
          </Panel>
        </div>
      </div>
    </SlideShell>
  )
}

/* ------------------------------------------------------------------ */
/* 10 — Caja negra / Grad-CAM                                          */
/* ------------------------------------------------------------------ */
function BlackBox() {
  return (
    <SlideShell kicker="Interpretabilidad" title="Abriendo la “caja negra”: transparencia mediante Grad-CAM">
      <div className="flex flex-1 flex-col gap-4">
        <div className="grid grid-cols-[1fr_auto_1fr_auto_1fr] items-center gap-3">
          <Panel className="flex flex-col items-center">
            <div className="relative aspect-square w-full max-w-[150px] overflow-hidden rounded-lg">
              <Image src="/xray-original.png" alt="Radiografía original de tórax" fill className="object-cover" />
            </div>
            <p className="mt-2 text-center text-xs font-medium text-muted-foreground">Imagen original</p>
          </Panel>
          <div className="text-2xl text-hud-cyan">→</div>
          <Panel className="flex flex-col items-center">
            <div className="relative aspect-square w-full max-w-[150px] overflow-hidden rounded-lg">
              <Image src="/gradcam-heatmap.png" alt="Mapas de características de MobileNetV2" fill className="object-cover" />
            </div>
            <p className="mt-2 text-center text-xs font-medium text-muted-foreground">Mapas de características · MobileNetV2</p>
          </Panel>
          <div className="text-2xl text-hud-cyan">→</div>
          <Panel accent="cyan" glow className="flex flex-col items-center">
            <div className="relative aspect-square w-full max-w-[150px] overflow-hidden rounded-lg">
              <Image src="/gradcam-overlay.png" alt="Mapa de calor anatómico superpuesto" fill className="object-cover" />
            </div>
            <p className="mt-2 text-center text-xs font-medium text-hud-cyan">Mapa de calor anatómico</p>
          </Panel>
        </div>

        <div className="grid grid-cols-3 gap-4">
          <Panel accent="red">
            <Callout title="Problema" accent="red">
              Una red profunda es opaca: da una probabilidad, pero no explica en qué se basa.
            </Callout>
          </Panel>
          <Panel accent="cyan">
            <Callout title="Solución" accent="cyan">
              Grad-CAM proyecta la atención del modelo como un mapa de calor sobre la anatomía.
            </Callout>
          </Panel>
          <Panel accent="green">
            <Callout title="Objetivo de auditoría" accent="green">
              Verificar que el modelo mira regiones clínicamente relevantes y no artefactos.
            </Callout>
          </Panel>
        </div>
      </div>
    </SlideShell>
  )
}

/* ------------------------------------------------------------------ */
/* 11 — Verdadero positivo                                             */
/* ------------------------------------------------------------------ */
function TruePositive() {
  const panels = [
    { src: "/xray-original.png", label: "Original", caption: "Radiografía de entrada" },
    { src: "/gradcam-heatmap.png", label: "Mapa de calor", caption: "Activación Grad-CAM" },
    { src: "/gradcam-overlay.png", label: "Superposición", caption: "Atención sobre anatomía" },
  ]
  return (
    <SlideShell kicker="Auditoría · caso real" title="Evidencia clínica: el acierto (verdadero positivo)">
      <div className="flex flex-1 flex-col gap-4">
        <div className="grid grid-cols-3 gap-4">
          {panels.map((p, i) => (
            <Panel key={p.label} accent={i === 2 ? "green" : undefined} glow={i === 2} className="flex flex-col">
              <div className="relative mx-auto aspect-square w-full max-w-[180px] overflow-hidden rounded-lg">
                <Image src={p.src || "/placeholder.svg"} alt={p.caption} fill className="object-cover" />
              </div>
              <p className="mt-2 text-sm font-semibold">{p.label}</p>
              <p className="text-xs text-muted-foreground">{p.caption}</p>
            </Panel>
          ))}
        </div>
        <Panel accent="green">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <CheckCircle2 className="size-6 text-hud-green" />
              <div>
                <p className="text-base font-bold text-hud-green">Verdadero positivo</p>
                <p className="text-sm text-muted-foreground">
                  El modelo enfoca la región torácica relevante: la decisión es clínicamente coherente.
                </p>
              </div>
            </div>
            <Stat label="Probabilidad" value="0,61" accent="green" />
          </div>
        </Panel>
      </div>
    </SlideShell>
  )
}

/* ------------------------------------------------------------------ */
/* 12 — Anatomía del error                                             */
/* ------------------------------------------------------------------ */
function ErrorAnatomy() {
  return (
    <SlideShell kicker="Auditoría · errores honestos" title="Anatomía del error: falsos negativos y positivos" accent="amber">
      <div className="grid flex-1 grid-cols-2 gap-4">
        <Panel accent="red" glow className="flex flex-col justify-center">
          <div className="mb-2 flex items-center justify-between">
            <div className="flex items-center gap-2 text-hud-red">
              <AlertTriangle className="size-5" />
              <h3 className="text-base font-bold">Falso negativo · error crítico</h3>
            </div>
            <span className="font-mono text-sm text-hud-red">p = 0,36</span>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="relative aspect-[4/3] overflow-hidden rounded-lg">
              <Image src="/fn-original.png" alt="Radiografía de un falso negativo" fill className="object-cover" />
            </div>
            <div className="relative aspect-[4/3] overflow-hidden rounded-lg">
              <Image src="/fn-gradcam.png" alt="Grad-CAM disperso en falso negativo" fill className="object-cover" />
            </div>
          </div>
          <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
            Activación <span className="text-hud-red">dispersa / dorsal</span>: el modelo no encuentra un foco claro en
            un caso sutil.
          </p>
        </Panel>

        <Panel accent="amber" className="flex flex-col justify-center">
          <div className="mb-2 flex items-center justify-between">
            <div className="flex items-center gap-2 text-hud-amber">
              <Eye className="size-5" />
              <h3 className="text-base font-bold">Falso positivo · falsa alarma</h3>
            </div>
            <span className="font-mono text-sm text-hud-amber">p = 0,54</span>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="relative aspect-[4/3] overflow-hidden rounded-lg">
              <Image src="/fp-original.png" alt="Radiografía de un falso positivo" fill className="object-cover" />
            </div>
            <div className="relative aspect-[4/3] overflow-hidden rounded-lg">
              <Image src="/fp-gradcam.png" alt="Grad-CAM focalizado en falso positivo" fill className="object-cover" />
            </div>
          </div>
          <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
            Foco en una <span className="text-hud-amber">región densa</span> que confunde al modelo, pero sin patología
            real.
          </p>
        </Panel>
      </div>
      <Panel accent="cyan" className="mt-4">
        <p className="text-center text-sm font-medium text-hud-cyan md:text-base">
          Conclusión: los errores ocurren en casos sutiles y limítrofes, no son fallas sistémicas del modelo.
        </p>
      </Panel>
    </SlideShell>
  )
}

/* ------------------------------------------------------------------ */
/* 13 — Conclusiones                                                   */
/* ------------------------------------------------------------------ */
function Conclusions() {
  const future = [
    { icon: GitBranch, title: "Validación cruzada k-fold", desc: "Estimar el desempeño de forma más robusta con datos escasos." },
    { icon: ScanLine, title: "Robustez de origen", desc: "Validar con radiografías de distintos equipos y clínicas." },
    { icon: Layers, title: "Transición a multiclase", desc: "Pasar de sano/patológico a clasificar patologías específicas." },
  ]
  return (
    <SlideShell kicker="Cierre" title="Conclusiones y evolución del sistema">
      <div className="flex flex-1 flex-col gap-6">
        <Panel accent="cyan" glow>
          <div className="flex items-start gap-4">
            <div className="flex size-12 shrink-0 items-center justify-center rounded-lg border border-hud-cyan/40 bg-hud-cyan/10">
              <TrendingUp className="size-6 text-hud-cyan" />
            </div>
            <div>
              <h3 className="text-lg font-bold text-hud-cyan md:text-xl">Logro principal</h3>
              <p className="mt-1 text-pretty text-sm leading-relaxed text-muted-foreground md:text-base">
                Transfer Learning (MobileNetV2) calibrado con un umbral de <span className="text-hud-cyan">0,30</span>{" "}
                produce un <span className="font-semibold text-foreground">filtro de triage interpretable</span>: sensible
                al paciente y auditable mediante Grad-CAM.
              </p>
            </div>
          </div>
        </Panel>

        <div>
          <p className="mb-3 text-sm font-semibold uppercase tracking-wider text-muted-foreground">Líneas de evolución</p>
          <div className="grid grid-cols-3 gap-4">
            {future.map((f) => (
              <Panel key={f.title} className="flex flex-col">
                <f.icon className="mb-3 size-6 text-hud-cyan" />
                <h4 className="text-base font-bold">{f.title}</h4>
                <p className="mt-1 text-sm leading-relaxed text-muted-foreground">{f.desc}</p>
              </Panel>
            ))}
          </div>
        </div>

        <Panel accent="amber" className="mt-auto">
          <div className="flex items-center gap-3">
            <Microscope className="size-5 text-hud-amber" />
            <p className="text-sm font-medium md:text-base">
              No se los queremos contar: se los queremos mostrar. <span className="text-hud-amber">Demo en vivo</span> —
              modelo <span className="font-mono">transfer_mobilenetv2.onnx</span> en la sección de Predicción en Vivo.
            </p>
          </div>
        </Panel>
      </div>
    </SlideShell>
  )
}

/* ------------------------------------------------------------------ */

export const slideMetas: SlideMeta[] = [
  { id: 1, title: "Portada", render: Cover },
  { id: 2, title: "El desafío clínico", render: ClinicalChallenge },
  { id: 3, title: "El espacio de datos", render: DataSpace },
  { id: 4, title: "El costo del error", render: CostMetric },
  { id: 5, title: "Dos enfoques", render: TwoApproaches },
  { id: 6, title: "Resultados · CNN", render: ResultsApproach1 },
  { id: 7, title: "Resultados · Transfer", render: ResultsApproach2 },
  { id: 8, title: "Síntesis de desempeño", render: Synthesis },
  { id: 9, title: "Calibrando la decisión", render: Threshold },
  { id: 10, title: "Caja negra · Grad-CAM", render: BlackBox },
  { id: 11, title: "Acierto (VP)", render: TruePositive },
  { id: 12, title: "Anatomía del error", render: ErrorAnatomy },
  { id: 13, title: "Conclusiones", render: Conclusions },
  { id: 14, title: "Demo en vivo", render: LiveDemoSlide },
]
