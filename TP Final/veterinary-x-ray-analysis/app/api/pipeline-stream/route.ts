import { spawn } from "child_process"

const PYTHON     = String.raw`C:\Users\gpoli\venvs\caece-mineria\Scripts\python.exe`
const SCRIPT     = String.raw`C:\Users\gpoli\OneDrive\Desktop\0-Maestrias\1-Caece\Ciencia de Datos\0-2026\1er Cuatrimestre\2do Bimestre\Aprendizaje Automatico\run_pipeline_cli.py`
const SCRIPT_CWD = String.raw`C:\Users\gpoli\OneDrive\Desktop\0-Maestrias\1-Caece\Ciencia de Datos\0-2026\1er Cuatrimestre\2do Bimestre\Aprendizaje Automatico`

export async function GET() {
  const encoder = new TextEncoder()

  const stream = new ReadableStream({
    start(controller) {
      const send = (line: string) => {
        try {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(line)}\n\n`))
        } catch { /* client disconnected */ }
      }

      const proc = spawn(PYTHON, ["-u", SCRIPT], {
        stdio: ["ignore", "pipe", "pipe"],
        cwd: SCRIPT_CWD,
        env: {
          ...process.env,
          PYTHONIOENCODING: "utf-8",  // force UTF-8 on Windows subprocesses
          PYTHONUTF8: "1",
        },
      })

      const onChunk = (chunk: Buffer) => {
        for (const line of chunk.toString("utf-8").split("\n")) {
          if (line.trim()) send(line)
        }
      }

      proc.stdout.on("data", onChunk)
      proc.stderr.on("data", (chunk: Buffer) => {
        for (const line of chunk.toString().split("\n")) {
          if (line.trim()) send(`[stderr] ${line}`)
        }
      })

      proc.on("close", (code) => {
        send(`__done__:${code === 0 ? "ok" : "error"}`)
        try { controller.close() } catch { /* already closed */ }
      })

      proc.on("error", (err) => {
        send(`[ERROR] No se pudo iniciar Python: ${err.message}`)
        send("__done__:error")
        try { controller.close() } catch { /* already closed */ }
      })
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
