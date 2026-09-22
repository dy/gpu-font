import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const python = process.env.GPU_FONT_PYTHON || fileURLToPath(new URL('../.venv/bin/python', import.meta.url))
const result = spawnSync(python, process.argv.slice(2), {
  stdio: 'inherit', env: { ...process.env, PYTHONDONTWRITEBYTECODE: '1' },
})
if (result.error) throw new Error(`Cannot run ${python}. Install requirements.txt in .venv or set GPU_FONT_PYTHON.`, { cause: result.error })
process.exit(result.status ?? 1)
