import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import { writeFile, mkdir, readFile } from 'fs/promises';
import { join } from 'path';
import { tmpdir } from 'os';

export const runtime = 'nodejs';

interface CompileRequest {
  source: string;
  ruleName: string;
  quantize?: 'int8' | 'int4' | null;
}

interface CompileResponse {
  ok: boolean;
  cCode?: string;
  error?: string;
  stats?: {
    concepts: number;
    bounds: number;
    memories: number;
    relates: number;
    rules: number;
  };
}

export async function POST(req: NextRequest): Promise<NextResponse<CompileResponse>> {
  try {
    const body: CompileRequest = await req.json();
    const { source, ruleName, quantize } = body;

    if (!source || !ruleName) {
      return NextResponse.json(
        { ok: false, error: 'source and ruleName are required' },
        { status: 400 }
      );
    }

    const tmpDir = join(tmpdir(), 'lal-playground');
    await mkdir(tmpDir, { recursive: true });
    const srcPath = join(tmpDir, `input_${Date.now()}.lal`);
    const outPath = join(tmpDir, `output_${Date.now()}.c`);
    await writeFile(srcPath, source);

    const args = ['src/lib/lal.py', srcPath, ruleName, outPath];
    if (quantize) {
      args.push('--quantize', quantize);
    }

    const result = await new Promise<{ ok: boolean; stdout: string; stderr: string }>((resolve) => {
      const py = spawn('python3', args, { cwd: process.cwd() });
      let stdout = '';
      let stderr = '';
      py.stdout.on('data', (d) => { stdout += d.toString(); });
      py.stderr.on('data', (d) => { stderr += d.toString(); });
      py.on('close', (code) => {
        resolve({ ok: code === 0, stdout, stderr });
      });
    });

    if (!result.ok) {
      return NextResponse.json(
        { ok: false, error: result.stderr || result.stdout || 'compilation failed' },
        { status: 500 }
      );
    }

    const cCode = await readFile(outPath, 'utf-8');
    const stats = parseStats(result.stdout);

    return NextResponse.json({ ok: true, cCode, stats });
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ ok: false, error: msg }, { status: 500 });
  }
}

function parseStats(stdout: string): CompileResponse['stats'] {
  const stats: NonNullable<CompileResponse['stats']> = {
    concepts: 0, bounds: 0, memories: 0, relates: 0, rules: 0,
  };
  const m = stdout.match(/concepts:\s*(\d+).*bounds:\s*(\d+).*memories:\s*(\d+).*relates:\s*(\d+).*rules:\s*(\d+)/);
  if (m) {
    stats.concepts = parseInt(m[1]);
    stats.bounds = parseInt(m[2]);
    stats.memories = parseInt(m[3]);
    stats.relates = parseInt(m[4]);
    stats.rules = parseInt(m[5]);
  }
  return stats;
}
