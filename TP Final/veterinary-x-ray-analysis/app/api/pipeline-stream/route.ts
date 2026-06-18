// Pipeline Runner · MODO DEMO
// Simula la ejecución del pipeline de entrenamiento emitiendo logs vía SSE.
// No depende de Python ni del sistema de archivos: funciona en cualquier host
// (Vercel/Netlify serverless) y reproduce los resultados reales del TP.

export const runtime = "nodejs"
export const maxDuration = 30
export const dynamic = "force-dynamic"

// Cada línea con su pausa (ms) antes de enviarse. El front detecta el paso
// activo con el prefijo [N/6] y colorea ✓ / ═ / [N/6] de forma distinta.
const SCRIPT: Array<[string, number]> = [
  ["═══ PIPELINE DE ENTRENAMIENTO · MODO DEMO ═══", 300],
  ["Dataset: Radiografías veterinarias · 2 clases (Normal / Patológica)", 250],
  ["", 200],

  ["[1/6] 🔍 Exploración del dataset", 350],
  ["  → Imágenes totales: 1.872", 200],
  ["  → Train / Val / Test: 1.310 / 281 / 281", 200],
  ["  → Balance de clases: Normal 54% · Patológica 46%", 200],
  ["✓ Exploración completada", 300],
  ["", 150],

  ["[2/6] ✂️ Preparación de datos", 350],
  ["  → Resize 224×224 · normalización [0,1]", 200],
  ["  → Data augmentation: flip horizontal, rotación ±10°", 200],
  ["  → Batches de 32 generados", 200],
  ["✓ Preparación completada", 300],
  ["", 150],

  ["[3/6] 🧱 CNN desde cero", 350],
  ["  → Arquitectura: 3 bloques Conv+Pool → Dense", 200],
  ["  → Epoch 10/10 · loss 0.512 · val_acc 0.731", 350],
  ["  → Accuracy test: 73.1%", 200],
  ["✓ Entrenamiento CNN completado", 300],
  ["", 150],

  ["[4/6] 🔁 Transfer Learning · MobileNetV2", 350],
  ["  → Base preentrenada en ImageNet (capas congeladas)", 200],
  ["  → Fine-tuning de las últimas capas · 15 epochs", 350],
  ["  → val_acc 0.912 · loss 0.241", 200],
  ["✓ Transfer Learning completado", 300],
  ["", 150],

  ["[5/6] 📊 Evaluación comparativa", 350],
  ["  → CNN desde cero  · Acc 73.1% · F1 0.70", 200],
  ["  → MobileNetV2     · Acc 91.2% · F1 0.90", 200],
  ["  → Umbral óptimo: 0.30 (maximiza recall patológico)", 250],
  ["✓ Evaluación completada", 300],
  ["", 150],

  ["[6/6] 🗺️ Grad-CAM", 350],
  ["  → Generando mapas de activación…", 300],
  ["  → Heatmaps guardados para los casos representativos", 250],
  ["✓ Grad-CAM completado", 300],
  ["", 200],

  ["═══ PIPELINE COMPLETADO ✓ ═══", 250],
  ["Modelo exportado: transfer_mobilenetv2.onnx", 200],
]

export async function GET() {
  const encoder = new TextEncoder()

  const stream = new ReadableStream({
    async start(controller) {
      const send = (line: string) => {
        try {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(line)}\n\n`))
        } catch { /* client disconnected */ }
      }
      const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

      try {
        for (const [line, delay] of SCRIPT) {
          await sleep(delay)
          if (line) send(line)
        }
        send("__done__:ok")
      } catch {
        send("[ERROR] Falló la simulación del pipeline")
        send("__done__:error")
      } finally {
        try { controller.close() } catch { /* already closed */ }
      }
    },
  })

  return new Response(stream, {
    headers: {
      "Content-Type":      "text/event-stream",
      "Cache-Control":     "no-cache, no-transform",
      "Connection":        "keep-alive",
      "X-Accel-Buffering": "no",
    },
  })
}
