"use client"

import type React from "react"

import { useEffect, useMemo, useState } from "react"
import { useUIStore, type AppKey } from "@/lib/ui-store"
import Eyes from "./components/Eyes"
import Dock from "./components/Dock"
import DesktopWindow from "./components/Window"
import CommandPalette from "./components/CommandPalette"

type WindowSpec = {
  key: Exclude<AppKey, "palette" | null>
  title: string
  content: React.ReactNode
}

// Placeholder content builders
function AboutContent() {
  return (
    <div className="p-5 md:p-6">
      <h1 className="font-black leading-none" style={{ fontSize: 52 }}>
        Hello, I&apos;m Alex.
      </h1>
      <p className="mt-4 text-lg">Neo‑brutalist designer/developer crafting playful, resilient interfaces.</p>
      <div className="mt-5 flex flex-wrap gap-2">
        {["Design Systems", "Creative Coding", "p5.js", "Next.js", "A11y"].map((chip) => (
          <span
            key={chip}
            className="text-sm font-semibold px-3 py-1 rounded-md border-[3px] border-black"
            style={{ backgroundColor: "#FF2E63", color: "#000" }}
          >
            {chip}
          </span>
        ))}
      </div>
    </div>
  )
}

function ArtContent() {
  const items = Array.from({ length: 8 }).map((_, i) => i)
  return (
    <div className="p-4 md:p-5">
      <div className="grid grid-cols-2 md:grid-cols-3 gap-3 md:gap-4">
        {items.map((i) => (
          <div key={i} className="aspect-square border-[3px] border-black bg-white shadow-[6px_6px_0_0_#000]">
            <img
              src={`/abstract-art-tile.png?height=600&width=600&query=abstract+art+tile+${i}`}
              alt={`Art placeholder ${i + 1}`}
              className="w-full h-full object-cover"
            />
          </div>
        ))}
      </div>
    </div>
  )
}

function PhilosophyContent() {
  const thoughts = [
    "Constraints sharpen creativity.",
    "Interfaces are invitations, not instructions.",
    "Taste is a muscle—practice it daily.",
    "The best tool is the one you actually use.",
  ]
  return (
    <div className="p-5 grid gap-4 md:grid-cols-2">
      {thoughts.map((t, i) => (
        <div key={i} className="p-4 border-[3px] border-black bg-white shadow-[6px_6px_0_0_#000] text-lg font-semibold">
          {t}
        </div>
      ))}
    </div>
  )
}

function ResumeContent() {
  const cards = [
    { title: "Work", body: "Senior Product Designer @ Studio (2021—Now)" },
    { title: "Tech", body: "TypeScript, React, Next.js, Tailwind, p5.js" },
    { title: "Focus", body: "Design systems, creative tooling, AI UX" },
    { title: "Education", body: "BDes Hons, Interaction Design" },
  ]
  return (
    <div className="p-5 grid gap-4 md:grid-cols-2">
      {cards.map((c, i) => (
        <div key={i} className="p-4 border-[3px] border-black bg-white shadow-[6px_6px_0_0_#000]">
          <div className="text-xl font-black">{c.title}</div>
          <div className="mt-2">{c.body}</div>
        </div>
      ))}
    </div>
  )
}

export default function Page() {
  // Windows order doubles as z-order; last is topmost
  const [openApps, setOpenApps] = useState<Exclude<AppKey, "palette" | null>[]>(["about"])
  const [paletteOpen, setPaletteOpen] = useState(false)
  const setActiveApp = useUIStore((s) => s.setActiveApp)
  const activeApp = useUIStore((s) => s.activeApp)

  // Initialize active app (About by default)
  useEffect(() => {
    setActiveApp("about")
  }, [setActiveApp])

  // Global key handling: Esc closes palette or topmost window
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (paletteOpen) {
          setPaletteOpen(false)
          setActiveApp(null)
          return
        }
        if (openApps.length > 0) {
          const top = openApps[openApps.length - 1]
          closeApp(top)
          return
        }
      }
      // Enter on focused dock item is native click; no interception necessary.
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [paletteOpen, openApps])

  const windows: WindowSpec[] = useMemo(
    () => [
      { key: "about", title: "About", content: <AboutContent /> },
      { key: "art", title: "Art", content: <ArtContent /> },
      { key: "philosophy", title: "Philosophy", content: <PhilosophyContent /> },
      { key: "resume", title: "Resume", content: <ResumeContent /> },
    ],
    [],
  )

  function openApp(app: Exclude<AppKey, "palette" | null>) {
    setOpenApps((prev) => {
      if (prev.includes(app)) {
        const without = prev.filter((a) => a !== app)
        return [...without, app]
      }
      return [...prev, app]
    })
    setActiveApp(app)
  }

  function closeApp(app: Exclude<AppKey, "palette" | null>) {
    setOpenApps((prev) => prev.filter((a) => a !== app))
    // If top was closed, set active to new top or null
    setActiveApp((s) => {
      // s can accept direct value; but we need latest openApps after removal
      return null
    })
  }

  function focusApp(app: Exclude<AppKey, "palette" | null>) {
    setOpenApps((prev) => {
      const without = prev.filter((a) => a !== app)
      return [...without, app]
    })
    setActiveApp(app)
  }

  function resetAll() {
    setOpenApps(["about"])
    setPaletteOpen(false)
    setActiveApp("about")
  }

  return (
    <main className="fixed inset-0 overflow-hidden">
      {/* Background: off-white, 8px grid + subtle grain */}
      <div
        className="absolute inset-0"
        style={{
          backgroundColor: "#FAFAF0",
          backgroundImage: `
            repeating-linear-gradient(0deg, rgba(0,0,0,0.04) 0, rgba(0,0,0,0.04) 1px, transparent 1px, transparent 8px),
            repeating-linear-gradient(90deg, rgba(0,0,0,0.04) 0, rgba(0,0,0,0.04) 1px, transparent 1px, transparent 8px)
          `,
          backgroundSize: "8px 8px, 8px 8px",
        }}
      />
      {/* Grain overlay */}
      <div
        className="absolute inset-0 pointer-events-none mix-blend-multiply opacity-30"
        style={{
          backgroundImage: "radial-gradient(rgba(0,0,0,0.04) 1px, transparent 1px)",
          backgroundSize: "2px 2px",
        }}
      />

      {/* p5 Eyes wallpaper */}
      <Eyes activeApp={activeApp ?? null} />

      {/* Windows */}
      <div className="absolute inset-0 z-10">
        {windows
          .filter((w) => openApps.includes(w.key))
          .map((w) => {
            const zIndex = 100 + openApps.indexOf(w.key)
            return (
              <DesktopWindow
                key={w.key}
                appKey={w.key}
                title={w.title}
                zIndex={zIndex}
                onClose={() => closeApp(w.key)}
                onFocus={() => focusApp(w.key)}
              >
                {w.content}
              </DesktopWindow>
            )
          })}
      </div>

      {/* Dock */}
      <div className="absolute left-0 right-0 bottom-6 z-20 flex justify-center">
        <Dock
          activeApp={activeApp ?? null}
          onOpen={(k) => openApp(k)}
          onOpenPalette={() => {
            setPaletteOpen(true)
            setActiveApp("palette")
          }}
        />
      </div>

      {/* Command Palette */}
      <CommandPalette
        open={paletteOpen}
        onOpenChange={(o) => {
          setPaletteOpen(o)
          if (!o) setActiveApp(null)
          if (o) setActiveApp("palette")
        }}
        onAction={(k) => {
          openApp(k)
          setPaletteOpen(false)
        }}
        onReset={resetAll}
      />
    </main>
  )
}
