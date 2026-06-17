import { NextResponse } from "next/server"
import { spawn }         from "child_process"
import net               from "net"
import path              from "path"

const RUNNER_PATH = String.raw`C:\Users\gpoli\OneDrive\Desktop\0-Maestrias\1-Caece\Ciencia de Datos\0-2026\1er Cuatrimestre\2do Bimestre\Aprendizaje Automatico\runner_app.py`
const STREAMLIT   = String.raw`C:\Users\gpoli\venvs\caece-mineria\Scripts\streamlit.exe`
const RUNNER_DIR  = path.dirname(RUNNER_PATH)
const PORT        = 8501

function isPortOpen(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const socket = new net.Socket()
    socket.setTimeout(600)
    socket.on("connect", () => { socket.destroy(); resolve(true) })
    socket.on("timeout", () => { socket.destroy(); resolve(false) })
    socket.on("error",   () => { socket.destroy(); resolve(false) })
    socket.connect(port, "127.0.0.1")
  })
}

export async function GET() {
  const running = await isPortOpen(PORT)
  return NextResponse.json({ running, url: `http://localhost:${PORT}` })
}

export async function POST() {
  const already = await isPortOpen(PORT)
  if (!already) {
    const proc = spawn(STREAMLIT, ["run", RUNNER_PATH], {
      detached: true,
      stdio:    "ignore",
      cwd:      RUNNER_DIR,
    })
    proc.unref()
    // Wait for Streamlit to bind the port (usually ~3 s)
    await new Promise((r) => setTimeout(r, 3500))
  }
  return NextResponse.json({ ok: true, url: `http://localhost:${PORT}` })
}
