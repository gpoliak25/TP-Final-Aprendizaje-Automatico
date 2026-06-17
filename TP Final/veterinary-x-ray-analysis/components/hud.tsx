import type { ReactNode } from "react"
import { cn } from "@/lib/utils"

/* ---------- Slide shell ---------- */

export function SlideShell({
  title,
  kicker,
  accent = "cyan",
  children,
  className,
}: {
  title: string
  kicker?: string
  accent?: "cyan" | "amber"
  children: ReactNode
  className?: string
}) {
  return (
    <div className="hud-grid flex h-full w-full flex-col overflow-hidden">
      <header className="shrink-0 px-6 pt-6 md:px-12 md:pt-10">
        {kicker ? (
          <p
            className={cn(
              "mb-2 text-xs font-semibold uppercase tracking-[0.25em] md:text-sm",
              accent === "cyan" ? "text-hud-cyan" : "text-hud-amber",
            )}
          >
            {kicker}
          </p>
        ) : null}
        <h2 className="text-balance text-2xl font-bold leading-tight tracking-tight text-foreground md:text-4xl lg:text-5xl">
          {title}
        </h2>
        <div
          className={cn(
            "mt-4 h-px w-full",
            accent === "cyan"
              ? "bg-gradient-to-r from-hud-cyan/70 via-hud-cyan/20 to-transparent"
              : "bg-gradient-to-r from-hud-amber/70 via-hud-amber/20 to-transparent",
          )}
        />
      </header>
      <div className={cn("flex min-h-0 flex-1 flex-col px-6 py-5 md:px-12 md:py-7", className)}>{children}</div>
    </div>
  )
}

/* ---------- Panel ---------- */

export function Panel({
  children,
  className,
  accent,
  glow = false,
}: {
  children: ReactNode
  className?: string
  accent?: "cyan" | "amber" | "red" | "green"
  glow?: boolean
}) {
  const styles =
    accent === "cyan"
      ? "border-hud-cyan/45 bg-gradient-to-br from-hud-cyan/[0.14] via-card/80 to-card/80"
      : accent === "amber"
        ? "border-hud-amber/50 bg-gradient-to-br from-hud-amber/[0.14] via-card/80 to-card/80"
        : accent === "red"
          ? "border-hud-red/55 bg-gradient-to-br from-hud-red/[0.16] via-card/80 to-card/80"
          : accent === "green"
            ? "border-hud-green/50 bg-gradient-to-br from-hud-green/[0.14] via-card/80 to-card/80"
            : "border-hud-cyan/20 bg-gradient-to-br from-hud-panel/90 to-card/70"
  const deco =
    accent === "amber"
      ? "text-hud-amber"
      : accent === "red"
        ? "text-hud-red"
        : accent === "green"
          ? "text-hud-green"
          : accent === "cyan"
            ? "text-hud-cyan"
            : "text-hud-cyan/45"
  const glowClass = glow
    ? accent === "amber"
      ? "hud-glow-amber"
      : accent === "red"
        ? "hud-glow-red"
        : accent === "green"
          ? "hud-glow-green"
          : "hud-glow-cyan"
    : ""
  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-xl border p-5 backdrop-blur-sm md:p-6",
        styles,
        glowClass,
        className,
      )}
    >
      {/* top accent strip */}
      <span className={cn("pointer-events-none absolute inset-x-0 top-0 h-0.5 bg-current opacity-70", deco)} />
      {/* corner ticks */}
      <span className={cn("pointer-events-none absolute left-2 top-2 size-2.5 border-l border-t border-current opacity-50", deco)} />
      <span className={cn("pointer-events-none absolute right-2 top-2 size-2.5 border-r border-t border-current opacity-50", deco)} />
      <span className={cn("pointer-events-none absolute bottom-2 left-2 size-2.5 border-b border-l border-current opacity-50", deco)} />
      <span className={cn("pointer-events-none absolute bottom-2 right-2 size-2.5 border-b border-r border-current opacity-50", deco)} />
      {children}
    </div>
  )
}

/* ---------- Stat ---------- */

export function Stat({
  label,
  value,
  sub,
  accent = "cyan",
  trend,
}: {
  label: string
  value: string
  sub?: string
  accent?: "cyan" | "amber" | "red" | "green"
  trend?: "up" | "down"
}) {
  const color =
    accent === "cyan"
      ? "text-hud-cyan"
      : accent === "amber"
        ? "text-hud-amber"
        : accent === "red"
          ? "text-hud-red"
          : "text-hud-green"
  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">{label}</span>
      <span className={cn("font-mono text-3xl font-bold tabular-nums md:text-4xl", color)}>
        {value}
        {trend ? <span className="ml-1 text-xl">{trend === "up" ? "▲" : "▼"}</span> : null}
      </span>
      {sub ? <span className="text-sm text-muted-foreground">{sub}</span> : null}
    </div>
  )
}

/* ---------- Callout ---------- */

export function Callout({
  title,
  accent = "cyan",
  children,
}: {
  title: string
  accent?: "cyan" | "amber" | "red" | "green"
  children: ReactNode
}) {
  const color =
    accent === "cyan"
      ? "text-hud-cyan"
      : accent === "amber"
        ? "text-hud-amber"
        : accent === "red"
          ? "text-hud-red"
          : "text-hud-green"
  return (
    <div>
      <h3 className={cn("mb-1.5 text-base font-bold md:text-lg", color)}>{title}</h3>
      <p className="text-pretty text-sm leading-relaxed text-muted-foreground md:text-base">{children}</p>
    </div>
  )
}
