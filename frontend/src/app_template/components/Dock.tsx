"use client"

import type React from "react"
import { User, Images, Brain, FileText, Palette } from "lucide-react"
import type { AppKey } from "@/lib/ui-store"
import { cn } from "@/lib/utils"

const ACCENT = "#FF2E63"

type DockItem = {
  key: Exclude<AppKey, "palette" | null>
  label: string
  icon: React.ComponentType<any>
}

const ITEMS: DockItem[] = [
  { key: "about", label: "About", icon: User },
  { key: "art", label: "Art", icon: Images },
  { key: "philosophy", label: "Philosophy", icon: Brain },
  { key: "resume", label: "Resume", icon: FileText },
]

export default function Dock({
  activeApp = null,
  onOpen = () => {},
  onOpenPalette = () => {},
}: {
  activeApp?: AppKey | null
  onOpen?: (k: Exclude<AppKey, "palette" | null>) => void
  onOpenPalette?: () => void
}) {
  return (
    <div className="flex items-center gap-3 bg-white border-[3px] border-black px-3 py-3 rounded-xl shadow-[6px_6px_0_0_#000]">
      {ITEMS.map((it) => {
        const Icon = it.icon
        const isActive = activeApp === it.key
        return (
          <button
            key={it.key}
            onClick={() => onOpen(it.key)}
            className={cn(
              "relative w-16 h-16 md:w-18 md:h-18 grid place-items-center border-[3px] border-black rounded-lg bg-[#FAFAF0]",
              "hover:translate-y-[-2px] transition-transform",
              "focus-visible:outline-4",
              isActive ? "shadow-[6px_6px_0_0_#000]" : "shadow-[4px_4px_0_0_#000]",
            )}
            style={{ outlineColor: ACCENT }}
            aria-label={it.label}
          >
            <Icon className="w-7 h-7" />
            <span className="sr-only">{it.label}</span>
          </button>
        )
      })}
      <div className="mx-1 w-[2px] self-stretch bg-black/20" />
      <button
        onClick={onOpenPalette}
        className={cn(
          "relative w-16 h-16 md:w-18 md:h-18 grid place-items-center border-[3px] border-black rounded-lg bg-[#FAFAF0]",
          "hover:translate-y-[-2px] transition-transform shadow-[4px_4px_0_0_#000]",
          "focus-visible:outline-4",
        )}
        style={{ outlineColor: ACCENT }}
        aria-label="Command Palette"
        title="Command Palette"
      >
        <Palette className="w-7 h-7" />
        <span className="sr-only">Command Palette</span>
      </button>
    </div>
  )
}
