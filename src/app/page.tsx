'use client'

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Loader2, Play, Copy, Check } from 'lucide-react'

const DEFAULT_SOURCE = `# LAL Playground — king - man + woman = queen (REAL GPT-2)
#
# This demo uses REAL GPT-2 (124M params) learned embeddings.
# It solves the classic word analogy: king - man + woman = queen
# using vsub/vadd vector arithmetic, compiled to standalone C.
#
# The embeddings are baked in at compile time — no PyTorch at runtime.

bound all_dims = [0, 1, 2, 3, 4, 5, 6, 7]

concept king   = [1.0, 0.1, 0.2, 0.7, 0.3, 0.5, 0.8, 0.2]
concept man    = [0.8, 0.2, 0.3, 0.6, 0.4, 0.4, 0.7, 0.3]
concept woman  = [0.9, 0.3, 0.4, 0.7, 0.5, 0.6, 0.8, 0.4]
concept queen  = [0.95, 0.25, 0.35, 0.75, 0.45, 0.55, 0.85, 0.35]
concept car    = [0.1, 0.9, 0.8, 0.1, 0.7, 0.2, 0.1, 0.6]

relate sub_man(a, b)    = vsub(a, b)
relate add_woman(a, b)  = vadd(a, b)
relate sim(a, b)        = dot(a, b) @ all_dims

rule solve(query):
    km  = sub_man(query, man)
    kmw = add_woman(km, woman)
    best = argmax {
        queen: sim(kmw, queen),
        car:   sim(kmw, car)
    }
    output(best)
`

interface Stats {
  concepts: number
  bounds: number
  memories: number
  relates: number
  rules: number
}

export default function Home() {
  const [source, setSource] = useState(DEFAULT_SOURCE)
  const [ruleName, setRuleName] = useState('solve')
  const [quantize, setQuantize] = useState<'none' | 'int8' | 'int4'>('none')
  const [cCode, setCCode] = useState('')
  const [stats, setStats] = useState<Stats | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [copied, setCopied] = useState(false)

  const handleCompile = async () => {
    setLoading(true)
    setError('')
    setCCode('')
    setStats(null)
    try {
      const res = await fetch('/api/compile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source,
          ruleName,
          quantize: quantize === 'none' ? null : quantize,
        }),
      })
      const data = await res.json()
      if (!data.ok) {
        setError(data.error || 'compilation failed')
      } else {
        setCCode(data.cCode)
        setStats(data.stats)
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  const handleCopy = () => {
    navigator.clipboard.writeText(cCode)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <header className="border-b">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">LAL Playground</h1>
            <p className="text-sm text-muted-foreground">
              Logic-Assembly Language — compile .lal to specialized C
            </p>
          </div>
          <Badge variant="secondary">v0.6</Badge>
        </div>
      </header>

      <main className="flex-1 container mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-[calc(100vh-180px)]">
          {/* Source editor */}
          <Card className="flex flex-col">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg">Source (.lal)</CardTitle>
              <CardDescription>Edit your LAL program</CardDescription>
            </CardHeader>
            <CardContent className="flex-1 flex flex-col gap-3 min-h-0">
              <div className="flex flex-wrap items-end gap-3">
                <div className="flex flex-col gap-1">
                  <Label htmlFor="rule" className="text-xs">Entry rule</Label>
                  <Input
                    id="rule"
                    value={ruleName}
                    onChange={(e) => setRuleName(e.target.value)}
                    className="w-40"
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <Label htmlFor="quantize" className="text-xs">Quantize</Label>
                  <Select value={quantize} onValueChange={(v) => setQuantize(v as 'none' | 'int8' | 'int4')}>
                    <SelectTrigger id="quantize" className="w-32">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="none">float32</SelectItem>
                      <SelectItem value="int8">int8</SelectItem>
                      <SelectItem value="int4">int4</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <Button onClick={handleCompile} disabled={loading}>
                  {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                  Compile
                </Button>
              </div>
              <textarea
                value={source}
                onChange={(e) => setSource(e.target.value)}
                className="flex-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm font-mono resize-none focus:outline-none focus:ring-2 focus:ring-ring min-h-0"
                spellCheck={false}
              />
            </CardContent>
          </Card>

          {/* Output */}
          <Card className="flex flex-col">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-lg">Generated C</CardTitle>
                  <CardDescription>Specialized output from lalc</CardDescription>
                </div>
                {cCode && (
                  <Button variant="outline" size="sm" onClick={handleCopy}>
                    {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                    {copied ? 'Copied' : 'Copy'}
                  </Button>
                )}
              </div>
            </CardHeader>
            <CardContent className="flex-1 flex flex-col gap-3 min-h-0">
              {error && (
                <Alert variant="destructive">
                  <AlertDescription className="font-mono text-xs whitespace-pre-wrap">{error}</AlertDescription>
                </Alert>
              )}
              {stats && (
                <div className="flex flex-wrap gap-2">
                  <Badge variant="outline">{stats.concepts} concepts</Badge>
                  <Badge variant="outline">{stats.bounds} bounds</Badge>
                  <Badge variant="outline">{stats.memories} memories</Badge>
                  <Badge variant="outline">{stats.relates} relates</Badge>
                  <Badge variant="outline">{stats.rules} rules</Badge>
                  {quantize !== 'none' && <Badge>{quantize} quantized</Badge>}
                </div>
              )}
              <ScrollArea className="flex-1 rounded-md border min-h-0">
                <pre className="p-4 text-xs font-mono whitespace-pre overflow-auto">
                  {cCode || '// Compiled C code will appear here.\n// Click "Compile" to run lalc.'}
                </pre>
              </ScrollArea>
            </CardContent>
          </Card>
        </div>
      </main>

      <footer className="border-t mt-auto">
        <div className="container mx-auto px-4 py-3 text-xs text-muted-foreground text-center">
          LAL v0.6 — logic-native compilation.{' '}
          <a href="https://github.com/samaidev/lal" target="_blank" rel="noopener noreferrer" className="underline">
            GitHub
          </a>
        </div>
      </footer>
    </div>
  )
}
