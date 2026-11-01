#!/usr/bin/env python3
"""Blinded annotation UI for the iSafetyBench binary label-ceiling study.

Run:  .venv/bin/python tools/kappa-annotation-server.py     then open http://127.0.0.1:8765

The blinding is the whole job of this file. In this corpus the real filename carries the source
video id and the parent folder IS the label (hazard/ vs normal/), and every source is
single-class -- so a single leaked path or source name hands over the answer. The browser
therefore only ever sees K-ids; the key stays in this process. Nothing in the page says how many
hazards you have called so far either: a running tally would let you balance toward the
prevalence you expect, which is exactly the bias this study must not have.

Writes append-only JSONL. A correction is a NEW line, never an edit -- the analysis takes the
last line per (kid, pass), and the discarded lines stay on disk as part of the record.
"""
import json, pathlib, re, sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

REPO = pathlib.Path(__file__).resolve().parent.parent
BASE = REPO / 'annotation_guide/kappa-isb-binary'
PORT = 8765
SESSION_SOFT_CAP = 60

PAGE = """<!doctype html><title>Gan nhan - nhi phan</title><style>
body{margin:0;background:#111;color:#eee;font:14px system-ui;display:flex;flex-direction:column;
height:100vh;align-items:center}#top{padding:8px;color:#888}video{max-height:62vh;max-width:96vw;
background:#000}#btns{display:flex;gap:14px;margin:12px}button{font:16px system-ui;padding:14px 28px;
border:1px solid #444;background:#1e1e1e;color:#eee;border-radius:6px;cursor:pointer}
button:hover{background:#2a2a2a}#conf span,#rec{display:inline-block;padding:6px 12px;margin:0 3px;
border:1px solid #333;border-radius:4px;cursor:pointer}.on{background:#2d4a2d!important;border-color:#4a7a4a!important}
#done{font-size:18px;margin-top:20vh;text-align:center}kbd{background:#222;padding:1px 5px;border-radius:3px}
</style><div id=top></div><video id=v controls autoplay></video>
<div id=conf>Do tin cay: <span data-c=1>1 thap</span><span data-c=2>2 vua</span><span data-c=3>3 cao</span>
&nbsp;&nbsp;<span id=rec>r &mdash; toi NHAN RA clip nay</span></div>
<div id=btns><button id=b0>bINH THUONG &nbsp;<kbd>f</kbd></button>
<button id=b1>NGUY HIEM &nbsp;<kbd>j</kbd></button></div>
<div id=hint style="color:#666">chon do tin cay truoc khi gui &middot; <kbd>space</kbd> xem lai</div>
<script>
let ev=null,conf=null,rec=false,t0=0,replays=0;
const $=i=>document.getElementById(i);
function setConf(c){conf=c;document.querySelectorAll('#conf span[data-c]').forEach(s=>
  s.classList.toggle('on',+s.dataset.c===c));}
function setRec(r){rec=r;$('rec').classList.toggle('on',r);}
async function load(){const r=await fetch('/next');const d=await r.json();
  if(d.done){document.body.innerHTML='<div id=done>Xong '+d.n+' su kien.<br>'+
    'Chay: .venv/bin/python tools/analyse-kappa.py</div>';return;}
  ev=d;conf=null;rec=false;replays=0;setConf(null);setRec(false);
  $('top').textContent='su kien '+d.event+' / '+d.total+(d.session_note?'  -  '+d.session_note:'');
  $('v').src='/clip/'+d.kid+'?e='+d.event;t0=Date.now();}
async function send(lab){if(conf===null){$('hint').style.color='#e55';return;}
  await fetch('/save',{method:'POST',body:JSON.stringify({kid:ev.kid,pass:ev.pass,event:ev.event,
    label:lab,confidence:conf,recognised:rec,dwell_ms:Date.now()-t0,replays:replays})});
  $('hint').style.color='#666';load();}
$('b0').onclick=()=>send(0);$('b1').onclick=()=>send(1);
document.querySelectorAll('#conf span[data-c]').forEach(s=>s.onclick=()=>setConf(+s.dataset.c));
$('rec').onclick=()=>setRec(!rec);
$('v').onplay=()=>{replays++};
document.onkeydown=e=>{if(e.key==='f')send(0);else if(e.key==='j')send(1);
  else if('123'.includes(e.key))setConf(+e.key);else if(e.key==='r')setRec(!rec);
  else if(e.key===' '){e.preventDefault();$('v').currentTime=0;$('v').play();}};
load();
</script>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _j(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == '/':
            b = PAGE.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif self.path == '/next':
            self._j(self.server.state.next_event())
        elif self.path.startswith('/clip/'):
            kid = re.split(r'[?/]', self.path[6:])[0]
            if kid not in self.server.state.key:
                self._j({'error': 'unknown'}, 404)
                return
            try:
                self.serve_clip(REPO / self.server.state.key[kid]['path'])
            except Exception:
                # never surface the exception: the real path contains the source id AND the
                # label-bearing folder, so an ordinary traceback is a blinding leak
                self._j({'error': 'clip unavailable', 'kid': kid}, 500)
        else:
            self._j({'error': 'no'}, 404)

    def serve_clip(self, p):
        data = p.read_bytes()
        rng = self.headers.get('Range')
        s, e = 0, len(data) - 1
        code = 200
        if rng and (m := re.match(r'bytes=(\d+)-(\d*)', rng)):
            s = int(m.group(1))
            e = int(m.group(2)) if m.group(2) else e
            code = 206
        chunk = data[s:e + 1]
        self.send_response(code)
        self.send_header('Content-Type', 'video/mp4')
        self.send_header('Accept-Ranges', 'bytes')
        if code == 206:
            self.send_header('Content-Range', f'bytes {s}-{e}/{len(data)}')
        self.send_header('Content-Length', str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)

    def do_POST(self):
        if self.path != '/save':
            self._j({'error': 'no'}, 404)
            return
        r = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.state.save(r)
        self._j({'ok': True})


class State:
    def __init__(self):
        self.key = json.loads((BASE / 'key.json').read_text())
        ev = json.loads((BASE / 'events.json').read_text())
        self.events = ev['events']
        self.log = BASE / 'annotations.jsonl'
        self.n_this_session = 0

    def done_set(self):
        if not self.log.exists():
            return set()
        return {(json.loads(l)['kid'], json.loads(l)['pass'])
                for l in self.log.read_text().splitlines() if l.strip()}

    def next_event(self):
        done = self.done_set()
        for e in self.events:
            if (e['kid'], e['pass']) not in done:
                note = ('nghi di -- da %d su kien trong phien nay' % self.n_this_session
                        if self.n_this_session >= SESSION_SOFT_CAP else '')
                return {**e, 'total': len(self.events), 'session_note': note}
        return {'done': True, 'n': len(done)}

    def save(self, r):
        r['ts'] = datetime.now(timezone.utc).isoformat()
        with self.log.open('a') as f:
            f.write(json.dumps(r, sort_keys=True) + '\n')
        self.n_this_session += 1


def main():
    if not (BASE / 'key.json').exists():
        sys.exit('chay truoc: .venv/bin/python tools/sample-kappa-clips.py --write')
    srv = HTTPServer(('127.0.0.1', PORT), H)
    srv.state = State()
    d = len(srv.state.done_set())
    print(f'da gan {d}/{len(srv.state.events)} su kien -> http://127.0.0.1:{PORT}  (ctrl-c de dung)')
    srv.serve_forever()


if __name__ == '__main__':
    main()
