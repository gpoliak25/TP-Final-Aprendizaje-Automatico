"use client"

import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react"

const DESIGN_WIDTH = 1280
const DESIGN_MIN_HEIGHT = 720

// Vertical chrome around the slide (header + footer + main padding).
// Used to cap the on-screen height so a tall slide scrolls instead of overflowing.
const VERTICAL_CHROME = 200

/**
 * Renders its children at a fixed design width (1280px) and scales the whole
 * canvas down uniformly by WIDTH so it always fills the available horizontal
 * space and stays legible. The canvas height tracks the slide's natural
 * content height. When a slide is taller than the viewport, it scrolls
 * vertically inside the frame instead of being shrunk or clipped.
 */
export function ScaledSlide({ children }: { children: ReactNode }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const contentRef = useRef<HTMLDivElement>(null)
  const [scale, setScale] = useState(0)
  const [contentHeight, setContentHeight] = useState(0)
  const [maxHeight, setMaxHeight] = useState(0)

  const useIsomorphicLayoutEffect = typeof window !== "undefined" ? useLayoutEffect : useEffect

  useIsomorphicLayoutEffect(() => {
    const container = containerRef.current
    const content = contentRef.current
    if (!container || !content) return

    const update = () => {
      const naturalHeight = Math.max(content.scrollHeight, DESIGN_MIN_HEIGHT)
      const availableWidth = container.clientWidth
      const availableHeight = window.innerHeight - VERTICAL_CHROME

      const widthScale = availableWidth / DESIGN_WIDTH
      const heightScale = availableHeight > 0 ? availableHeight / naturalHeight : widthScale
      // Fit within BOTH dimensions so the whole slide is always visible (no scroll).
      const fitScale = Math.min(widthScale, heightScale)

      setContentHeight(naturalHeight)
      setScale(fitScale > 0 ? fitScale : 1)
      setMaxHeight(availableHeight > 0 ? availableHeight : 0)
    }

    update()

    const ro = new ResizeObserver(update)
    ro.observe(container)
    ro.observe(content)
    window.addEventListener("resize", update)
    return () => {
      ro.disconnect()
      window.removeEventListener("resize", update)
    }
  }, [children])

  const scaledHeight = scale ? contentHeight * scale : undefined

  return (
    <div ref={containerRef} className="flex w-full justify-center">
      <div
        className="relative overflow-hidden"
        style={{
          width: scale ? DESIGN_WIDTH * scale : "100%",
          height: scaledHeight,
          maxHeight: maxHeight || undefined,
        }}
      >
        <div
          ref={contentRef}
          className="absolute left-0 top-0 origin-top-left"
          style={{
            width: DESIGN_WIDTH,
            minHeight: DESIGN_MIN_HEIGHT,
            height: contentHeight || undefined,
            transform: scale ? `scale(${scale})` : undefined,
          }}
        >
          {children}
        </div>
      </div>
    </div>
  )
}
