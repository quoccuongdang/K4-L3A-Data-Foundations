import { useEffect, useRef, useState } from "react"
import {
  ArrowUp,
  BookOpenText,
  ExternalLink,
  LoaderCircle,
  RotateCcw,
  Sparkles,
} from "lucide-react"
import ReactMarkdown from "react-markdown"

import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

const VIDEO_URL =
  "https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260314_131748_f2ca2a28-fed7-44c8-b9a9-bd9acdd5ec31.mp4"

const suggestions = [
  "Điều kiện để tham gia OJT là gì?",
  "Học phí tại campus TP.HCM được tính thế nào?",
  "Điều kiện xét học bổng là gì?",
  "Tôi cần liên hệ phòng ban nào để được hỗ trợ?",
]

type Source = {
  file: string
  title: string
  url: string
  category: string
}

type Message = {
  id: string
  role: "user" | "assistant"
  content: string
  sources?: Source[]
  error?: boolean
}

type Health = {
  state: "loading" | "ready" | "error"
  chunks: number
  documents: number
  backend: string
  message: string
}

const initialHealth: Health = {
  state: "loading",
  chunks: 0,
  documents: 0,
  backend: "",
  message: "Đang kết nối kho tri thức…",
}

function StatusBadge({ health }: { health: Health }) {
  const label =
    health.state === "ready"
      ? `${health.documents} tài liệu · ${health.chunks} đoạn dữ liệu`
      : health.message

  return (
    <div
      className="liquid-glass flex max-w-[15rem] items-center gap-2 rounded-full px-3.5 py-2 text-xs text-white/75 sm:max-w-none"
      title={health.backend || health.message}
    >
      <span
        className={cn(
          "size-1.5 shrink-0 rounded-full",
          health.state === "ready" && "bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.9)]",
          health.state === "loading" && "animate-pulse bg-amber-300",
          health.state === "error" && "bg-red-400",
        )}
      />
      <span className="truncate">{label}</span>
    </div>
  )
}

function Composer({
  value,
  onChange,
  onSubmit,
  disabled,
  compact = false,
}: {
  value: string
  onChange: (value: string) => void
  onSubmit: () => void
  disabled: boolean
  compact?: boolean
}) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (!compact) textareaRef.current?.focus()
  }, [compact])

  return (
    <form
      className={cn(
        "liquid-glass mx-auto flex w-full items-end gap-3 rounded-[1.75rem] p-2 pl-5 transition-colors focus-within:bg-white/[0.06]",
        compact ? "max-w-4xl" : "max-w-2xl",
      )}
      onSubmit={(event) => {
        event.preventDefault()
        onSubmit()
      }}
    >
      <label className="sr-only" htmlFor={compact ? "follow-up" : "first-question"}>
        Câu hỏi dành cho trợ lý sinh viên
      </label>
      <textarea
        ref={textareaRef}
        id={compact ? "follow-up" : "first-question"}
        rows={1}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault()
            onSubmit()
          }
        }}
        placeholder={disabled ? "Kho tri thức đang được chuẩn bị…" : "Hỏi về học phí, OJT, học bổng, quy chế…"}
        className="max-h-32 min-h-11 flex-1 resize-none bg-transparent py-3 text-[15px] leading-5 text-white outline-none placeholder:text-white/45 disabled:cursor-wait"
      />
      <Button
        type="submit"
        size="icon"
        disabled={disabled || value.trim().length < 2}
        aria-label="Gửi câu hỏi"
        className="size-11 shrink-0 rounded-full bg-white text-slate-950 hover:scale-[1.03] hover:bg-white/90"
      >
        {disabled && value.trim() ? <LoaderCircle className="animate-spin" /> : <ArrowUp />}
      </Button>
    </form>
  )
}

function SourceList({ sources }: { sources: Source[] }) {
  if (!sources.length) return null

  return (
    <div className="mt-6 border-t border-white/10 pt-4">
      <div className="mb-3 flex items-center gap-2 text-xs font-medium uppercase tracking-[0.16em] text-white/45">
        <BookOpenText className="size-3.5" />
        Nguồn đã truy xuất
      </div>
      <div className="flex flex-wrap gap-2">
        {sources.map((source) => {
          const label = source.title.replace(/\s+-\s+Trường Đại học FPT.*$/i, "")
          return source.url ? (
            <a
              key={source.file}
              href={source.url}
              target="_blank"
              rel="noreferrer"
              className="liquid-glass group inline-flex max-w-full items-center gap-2 rounded-full px-3 py-2 text-xs text-white/70 transition-colors hover:text-white"
            >
              <span className="truncate">{label}</span>
              <ExternalLink className="size-3 shrink-0 opacity-60 transition-transform group-hover:-translate-y-0.5 group-hover:translate-x-0.5" />
            </a>
          ) : (
            <span
              key={source.file}
              className="liquid-glass inline-flex max-w-full rounded-full px-3 py-2 text-xs text-white/70"
            >
              <span className="truncate">{label}</span>
            </span>
          )
        })}
      </div>
    </div>
  )
}

function App() {
  const [health, setHealth] = useState<Health>(initialHealth)
  const [messages, setMessages] = useState<Message[]>([])
  const [question, setQuestion] = useState("")
  const [isAnswering, setIsAnswering] = useState(false)
  const conversationEnd = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let active = true
    let timer: number | undefined

    const checkHealth = async () => {
      try {
        const response = await fetch("/api/health")
        if (!response.ok) throw new Error("Backend unavailable")
        const nextHealth = (await response.json()) as Health
        if (!active) return
        setHealth(nextHealth)
        if (nextHealth.state === "loading") {
          timer = window.setTimeout(checkHealth, 1800)
        }
      } catch {
        if (!active) return
        setHealth({
          ...initialHealth,
          state: "error",
          message: "Không thể kết nối máy chủ RAG",
        })
        timer = window.setTimeout(checkHealth, 3000)
      }
    }

    void checkHealth()
    return () => {
      active = false
      if (timer) window.clearTimeout(timer)
    }
  }, [])

  useEffect(() => {
    conversationEnd.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, isAnswering])

  const sendQuestion = async (preset?: string) => {
    const text = (preset ?? question).trim()
    if (text.length < 2 || isAnswering || health.state !== "ready") return

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
    }
    setMessages((current) => [...current, userMessage])
    setQuestion("")
    setIsAnswering(true)

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text }),
      })
      const payload = await response.json()
      if (!response.ok) throw new Error(payload.detail || "Không thể tạo câu trả lời.")

      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: payload.answer,
          sources: payload.sources,
        },
      ])
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: error instanceof Error ? error.message : "Đã có lỗi xảy ra. Vui lòng thử lại.",
          error: true,
        },
      ])
    } finally {
      setIsAnswering(false)
    }
  }

  const hasConversation = messages.length > 0

  return (
    <div className="relative isolate min-h-dvh overflow-hidden bg-background text-foreground">
      <video
        className="fixed inset-0 z-0 h-full w-full object-cover"
        autoPlay
        loop
        muted
        playsInline
        aria-hidden="true"
      >
        <source src={VIDEO_URL} type="video/mp4" />
      </video>

      <div className="relative z-10 flex min-h-dvh flex-col">
        <header className="mx-auto flex w-full max-w-7xl items-center justify-between px-5 py-5 sm:px-8 sm:py-6">
          <button
            type="button"
            className="font-display text-2xl tracking-tight text-white sm:text-3xl"
            onClick={() => setMessages([])}
            aria-label="Về trang bắt đầu"
          >
            FPTU Guide<sup className="ml-1 align-super font-body text-[9px] tracking-normal text-white/55">AI</sup>
          </button>

          <div className="flex items-center gap-2 sm:gap-3">
            <StatusBadge health={health} />
            {hasConversation && (
              <Button
                type="button"
                variant="glass"
                size="icon"
                className="hidden rounded-full sm:inline-flex"
                onClick={() => setMessages([])}
                aria-label="Bắt đầu cuộc trò chuyện mới"
              >
                <RotateCcw />
              </Button>
            )}
          </div>
        </header>

        {!hasConversation ? (
          <main className="mx-auto flex w-full max-w-7xl flex-1 flex-col items-center justify-center px-5 pb-16 pt-8 text-center sm:px-8 sm:pb-24">
            <div className="mb-5 flex animate-fade-rise items-center gap-2 text-xs font-medium uppercase tracking-[0.22em] text-white/55">
              <Sparkles className="size-3.5" />
              Trợ lý dữ liệu sinh viên FPTU HCM
            </div>
            <h1 className="animate-fade-rise max-w-6xl font-display text-5xl font-normal leading-[0.95] tracking-[-0.035em] text-white sm:text-7xl md:text-8xl">
              Mọi câu hỏi sinh viên,
              <br />
              <em className="not-italic text-white/55">được dẫn lối bởi dữ liệu.</em>
            </h1>
            <p className="animate-fade-rise-delay mt-7 max-w-2xl text-sm leading-relaxed text-white/60 sm:mt-8 sm:text-base">
              Tra cứu quy chế đào tạo, học phí, học bổng, OJT và dịch vụ campus từ kho tài liệu
              FPTU đã được thu thập — kèm nguồn để bạn kiểm chứng.
            </p>

            <div className="animate-fade-rise-delay-2 mt-9 w-full sm:mt-11">
              <Composer
                value={question}
                onChange={setQuestion}
                onSubmit={() => void sendQuestion()}
                disabled={health.state !== "ready" || isAnswering}
              />
              <div className="mx-auto mt-4 flex max-w-3xl flex-wrap justify-center gap-2">
                {suggestions.map((suggestion) => (
                  <Button
                    key={suggestion}
                    type="button"
                    variant="glass"
                    size="sm"
                    disabled={health.state !== "ready"}
                    className="h-auto rounded-full px-3.5 py-2 text-xs font-normal text-white/65 hover:text-white"
                    onClick={() => void sendQuestion(suggestion)}
                  >
                    {suggestion}
                  </Button>
                ))}
              </div>
            </div>
          </main>
        ) : (
          <main className="mx-auto flex min-h-0 w-full max-w-5xl flex-1 flex-col px-5 sm:px-8">
            <div className="conversation-scroll flex-1 overflow-y-auto pb-7 pt-5 sm:pt-10">
              <div className="mx-auto max-w-4xl space-y-8">
                {messages.map((message) =>
                  message.role === "user" ? (
                    <div key={message.id} className="flex justify-end">
                      <div className="liquid-glass max-w-[88%] rounded-[1.5rem] rounded-br-md px-5 py-3.5 text-sm leading-6 text-white sm:max-w-[75%] sm:text-[15px]">
                        {message.content}
                      </div>
                    </div>
                  ) : (
                    <article
                      key={message.id}
                      className={cn(
                        "animate-fade-rise max-w-3xl",
                        message.error && "text-red-200",
                      )}
                    >
                      <div className="mb-3 flex items-center gap-2 text-xs font-medium uppercase tracking-[0.16em] text-white/45">
                        <Sparkles className="size-3.5" />
                        FPTU Guide
                      </div>
                      <div className="markdown text-[15px] leading-7 text-white/85 sm:text-base">
                        <ReactMarkdown>{message.content}</ReactMarkdown>
                      </div>
                      {message.sources && <SourceList sources={message.sources} />}
                    </article>
                  ),
                )}

                {isAnswering && (
                  <div className="flex items-center gap-3 text-sm text-white/55">
                    <LoaderCircle className="size-4 animate-spin" />
                    Đang truy xuất tài liệu và soạn câu trả lời…
                  </div>
                )}
                <div ref={conversationEnd} />
              </div>
            </div>

            <div className="pb-5 pt-2 sm:pb-7">
              <Composer
                compact
                value={question}
                onChange={setQuestion}
                onSubmit={() => void sendQuestion()}
                disabled={health.state !== "ready" || isAnswering}
              />
              <p className="mt-2 text-center text-[11px] text-white/35">
                Câu trả lời được tạo từ kho dữ liệu hiện có. Hãy kiểm tra nguồn khi ra quyết định quan trọng.
              </p>
            </div>
          </main>
        )}
      </div>
    </div>
  )
}

export default App
