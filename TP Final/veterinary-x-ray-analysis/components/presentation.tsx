"use client"

import { useCallback, useEffect, useState } from "react"
import { ChevronLeft, ChevronRight } from "lucide-react"
import { slideMetas } from "@/components/slides-content"
import { ScaledSlide } from "@/components/scaled-slide"
import { cn } from "@/lib/utils"

export function Presentation() {
  const [index, setIndex] = useState(0)
  const total = slideMetas.length

  const goTo = useCallback(
    (next: number) => {
      setIndex(() => (next + total) % total)
    },
    [total],
  )

  const prev = useCallback(() => goTo(index - 1), [goTo, index])
  const next = useCallback(() => goTo(index + 1), [goTo, index])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight" || e.key === " ") {
        e.preventDefault()
        next()
      } else if (e.key === "ArrowLeft") {
        e.preventDefault()
        prev()
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [next, prev])

  const current = slideMetas[index]
  const Slide = current.render

  return (
    <div className="flex min-h-screen flex-col bg-background text-foreground">
      <header className="flex shrink-0 items-center justify-between gap-4 border-b border-border px-4 py-3 md:px-6">
        <div className="flex min-w-0 items-center gap-3">
          <span className="hidden size-2 shrink-0 rounded-full bg-hud-cyan md:block" />
          <div className="min-w-0">
            <p className="text-[10px] font-semibold uppercase tracking-[0.25em] text-hud-cyan md:text-xs">
              Detección de Patologías · CNN
            </p>
            <h1 className="truncate text-sm font-semibold md:text-base">{current.title}</h1>
          </div>
        </div>
        <span className="shrink-0 rounded-full border border-border bg-card px-3 py-1 font-mono text-xs font-medium tabular-nums text-muted-foreground">
          {String(index + 1).padStart(2, "0")} / {String(total).padStart(2, "0")}
        </span>
      </header>

      <main className="flex flex-1 items-center justify-center px-3 py-5 md:px-10 md:py-8">
        <div className="relative w-full max-w-6xl">
          <ScaledSlide key={current.id}>
            <Slide />
          </ScaledSlide>

          <button
            type="button"
            onClick={prev}
            aria-label="Diapositiva anterior"
            className="absolute left-1 top-1/2 -translate-y-1/2 rounded-full border border-border bg-background/80 p-2 text-foreground backdrop-blur transition-colors hover:border-hud-cyan/50 hover:text-hud-cyan md:-left-5"
          >
            <ChevronLeft className="size-5" />
          </button>
          <button
            type="button"
            onClick={next}
            aria-label="Diapositiva siguiente"
            className="absolute right-1 top-1/2 -translate-y-1/2 rounded-full border border-border bg-background/80 p-2 text-foreground backdrop-blur transition-colors hover:border-hud-cyan/50 hover:text-hud-cyan md:-right-5"
          >
            <ChevronRight className="size-5" />
          </button>
        </div>
      </main>

      <footer className="shrink-0 border-t border-border px-4 py-4 md:px-6">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-center gap-2">
          {slideMetas.map((slide, i) => (
            <button
              key={slide.id}
              type="button"
              onClick={() => goTo(i)}
              aria-label={`Ir a la diapositiva ${i + 1}: ${slide.title}`}
              aria-current={i === index}
              className={cn(
                "h-2 rounded-full transition-all",
                i === index ? "w-7 bg-hud-cyan" : "w-2 bg-muted-foreground/40 hover:bg-muted-foreground",
              )}
            />
          ))}
        </div>
      </footer>
    </div>
  )
}
