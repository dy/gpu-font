"""Text lines in photographs, boxed by macOS Vision, so a collected set of whole scenes becomes crops without hand work.
Reads a collection folder: images.jsonl names each downloaded picture (title, file under images/), info.jsonl what the
collector recorded about it (typeface, licence, author). Writes boxes.jsonl beside them: per picture, the text lines
Vision finds, each with its pixel box, the words it read and its confidence, largest first. A reviewer keeps the lines
set in the typeface; the benchmark takes their boxes as crop boxes.
    python scripts/text_boxes.py .data/photos/commons        # macOS only (pyobjc-framework-Vision)
"""
import json
import sys
from pathlib import Path

import Quartz
import Vision


def lines_of(path):
    """Vision's recognized text lines in image pixels: [{'box': [x0, y0, x1, y1], 'text', 'confidence'}], largest first."""
    url = Quartz.CFURLCreateWithFileSystemPath(None, str(path), Quartz.kCFURLPOSIXPathStyle, False)
    source = Quartz.CGImageSourceCreateWithURL(url, None)
    if source is None: return None
    image = Quartz.CGImageSourceCreateImageAtIndex(source, 0, None)
    if image is None: return None
    width, height = Quartz.CGImageGetWidth(image), Quartz.CGImageGetHeight(image)
    request = Vision.VNRecognizeTextRequest.alloc().init(); request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(image, None)
    ok, error = handler.performRequests_error_([request], None)
    if not ok: raise RuntimeError(str(error))
    found = []
    for observation in request.results() or []:
        box = observation.boundingBox()  # normalized, origin at the bottom left
        x0, y0 = box.origin.x * width, (1 - box.origin.y - box.size.height) * height
        x1, y1 = x0 + box.size.width * width, y0 + box.size.height * height
        best = observation.topCandidates_(1)
        text = str(best[0].string()) if best else ''
        found.append({'box': [round(x0), round(y0), round(x1), round(y1)], 'text': text, 'confidence': float(observation.confidence())})
    return {'width': width, 'height': height, 'lines': sorted(found, key=lambda l: -(l['box'][2] - l['box'][0]) * (l['box'][3] - l['box'][1]))}


def main(folder):
    folder = Path(folder); out = folder/'boxes.jsonl'
    rows = lambda name: [json.loads(line) for line in (folder/name).read_text().splitlines() if line.strip()]
    about = {row['title']: row for row in rows('info.jsonl')}; pictures = rows('images.jsonl')
    done = {json.loads(line)['file'] for line in out.read_text().splitlines() if line.strip()} if out.exists() else set()
    with out.open('a') as stream:
        for n, picture in enumerate(pictures):
            if picture['file'] in done: continue
            path = folder/'images'/picture['file']; meta = about.get(picture['title'], {})
            try: found = lines_of(path)
            except Exception as error: found = None; print(f'{path}: {error}', file=sys.stderr)
            stream.write(json.dumps({'file': picture['file'], 'title': picture['title'], 'typeface': meta.get('typeface'), **(found or {'lines': None})}) + '\n'); stream.flush()
            if (n + 1) % 200 == 0: print(f'{n + 1}/{len(pictures)}', flush=True)
    boxed = sum(1 for line in out.read_text().splitlines() if line.strip() and json.loads(line).get('lines'))
    print(f'{boxed} of {len(pictures)} pictures have text lines; boxes in {out}', flush=True)


if __name__ == '__main__': main(sys.argv[1])
