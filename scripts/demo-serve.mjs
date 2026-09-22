import { createServer } from 'node:http'
import { readFile, realpath } from 'node:fs/promises'
import { resolve, extname, sep } from 'node:path'

const root = await realpath('dist').catch(() => { throw new Error('Run npm run demo:build first') })
const types = { '.html': 'text/html', '.css': 'text/css', '.mjs': 'text/javascript', '.json': 'application/json', '.ttf': 'font/ttf', '.png': 'image/png' }
const server = createServer(async (req, res) => {
  try {
    if (!['GET', 'HEAD'].includes(req.method)) { res.writeHead(405).end(); return }
    const pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname)
    const file = await realpath(resolve(root, `.${pathname === '/' ? '/index.html' : pathname}`))
    if (!file.startsWith(root + sep)) { res.writeHead(403).end(); return }
    const bytes = await readFile(file)
    res.writeHead(200, { 'Content-Type': types[extname(file)] || 'application/octet-stream', 'Cache-Control': 'no-cache', 'X-Content-Type-Options': 'nosniff' })
    res.end(req.method === 'HEAD' ? undefined : bytes)
  } catch { res.writeHead(404).end('Not found') }
})
server.on('error', error => { console.error(`Cannot start demo: ${error.message}`); process.exitCode = 1 })
server.listen(Number(process.env.PORT || 4179), '127.0.0.1', () => console.log(`gpu-font → http://localhost:${server.address().port}`))
