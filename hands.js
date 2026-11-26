/* 摄像头 + 手势。画面只在本机浏览器里处理，这里没有任何上传视频的代码。 */

/* ================= 可以随意调的参数 ================= */
const HAND = {
  pinchOn: 0.25,       // 拇指食指距离 ÷ 手的大小，小于它算“捏住”
  pinchOff: 0.40,      // 大于它算“松开”（两个门槛不同，防止来回抖）
  pluckOpen: 0.45,     // 手张开到这个程度才会拨弦
  pinchDwell: 90,      // 双手捏住多少毫秒后才开始拉弦，防误触
  releaseBack: 100,    // 松手时取多少毫秒之前的位置，避免松手动作带歪弦
  afterReleaseMute: 250, // 松手后这只手多少毫秒内不拨弦
  minPluckSpeed: 0.25, // 指尖每秒至少划过多少个屏幕高度才算拨弦
  rearm: 14,           // 拨过后，指尖要离开弦多少像素才能再拨它
  grab: 30,            // 捏在弦端点多少像素以内，算抓住端点
  bridge: 150,         // 手快速划动时短暂跟丢，这么多毫秒内找回来就接着算
  lost: 300,           // 丢失超过这么久就当手离开了
  smooth: 1.6,         // 越小越稳但越慢
  follow: 0.012,       // 越大，快速移动时跟得越紧
  dim: 0.55            // 画面压暗程度 0~1
};
const MP = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
const MODEL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const BONES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];

// One Euro 滤波：慢的时候稳，快的时候跟手
class Smooth {
  constructor() { this.v = null; this.d = 0; this.t = 0; }
  next(x, t) {
    if (this.v === null) { this.v = x; this.t = t; return x; }
    const dt = Math.max((t - this.t) / 1000, .004); this.t = t;
    const k = c => 1 / (1 + 1 / (2 * Math.PI * c * dt));
    this.d += k(1) * ((x - this.v) / dt - this.d);
    this.v += k(HAND.smooth + HAND.follow * Math.abs(this.d)) * (x - this.v);
    return this.v;
  }
}

Hands = {
  on: false, video: null, stream: null, model: null, lastVideo: -1,
  tracks: [], draft: null, trail: [], bothSince: 0,

  async loadModel() {
    if (this.model) return;
    const { FilesetResolver, HandLandmarker } = await import(MP + '/vision_bundle.mjs');
    const files = await FilesetResolver.forVisionTasks(MP + '/wasm');
    const make = delegate => HandLandmarker.createFromOptions(files, {
      baseOptions: { modelAssetPath: MODEL, delegate }, runningMode: 'VIDEO', numHands: 2 });
    try { this.model = await make('GPU'); } catch (e) { this.model = await make('CPU'); }
  },
  async start() {
    if (!navigator.mediaDevices) throw new DOMException('', 'NotSupportedError');
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: false, video: { facingMode: 'user', width: { ideal: 960 }, height: { ideal: 540 }, frameRate: { ideal: 60 } } });
    try {
      const v = this.video = document.createElement('video');
      v.muted = true; v.playsInline = true; v.srcObject = this.stream;
      await v.play();
      await Promise.all([this.loadModel(), Face.wanted() ? Face.load() : null]);
      if (Face.model) Face.model.detectForVideo(v, now());
      this.model.detectForVideo(v, now()); // 预热：第一次识别很慢，放在加载时做
      this.stream.getVideoTracks()[0].addEventListener('ended', () => { this.stop(); cameraStopped('hintLost'); });
      this.on = true;
    } catch (e) { this.stop(); throw e; }
  },
  stop() {
    this.on = false;
    if (this.stream) this.stream.getTracks().forEach(t => t.stop());
    this.stream = this.video = this.draft = null;
    this.tracks = []; this.trail = []; Face.tracks = []; this.lastVideo = -1; this.bothSince = 0;
  },

  // 视频铺满屏幕时的位置
  cover() {
    const vw = this.video.videoWidth || 16, vh = this.video.videoHeight || 9, k = Math.max(W / vw, H / vh);
    return { x: (W - vw * k) / 2, y: (H - vh * k) / 2, w: vw * k, h: vh * k };
  },
  tick() {
    if (!this.on || this.video.currentTime === this.lastVideo || this.video.readyState < 2) return;
    this.lastVideo = this.video.currentTime;
    // 识别模型要求时间只增不减；万一出错只跳过这一帧，不让画面卡死
    const t = this.ts = Math.max(now(), (this.ts || 0) + 1), c = this.cover();
    let res;
    try { res = this.model.detectForVideo(this.video, t); } catch (e) { console.warn(e); return; }
    this.process(res.landmarks.map(h => h.map(p => ({ x: c.x + (1 - p.x) * c.w, y: c.y + p.y * c.h }))), t);
    Face.tick(this.video, t, c);
  },

  // hands：每只手 21 个点的屏幕坐标（已镜像）
  process(hands, t) {
    this.tracks = this.tracks.filter(k => t - k.seen < HAND.lost);
    // 把这一帧的手和上一帧的手按距离配对，不依赖左右手标签
    const pairs = [];
    hands.forEach((h, i) => this.tracks.forEach(k => pairs.push({ i, k, d: dist(h[9], k.pts[9]) })));
    pairs.sort((a, b) => a.d - b.d);
    const usedHand = new Map(), usedTrack = new Set();
    for (const p of pairs) {
      if (usedHand.has(p.i) || usedTrack.has(p.k) || p.d > W * .35) continue;
      usedHand.set(p.i, p.k); usedTrack.add(p.k);
    }
    hands.forEach((h, i) => {
      let k = usedHand.get(i);
      if (!k) {
        k = { f: {}, pinch: false, mute: 0, tip: null, tipT: 0, blocked: new Set(), grab: null, past: [], seen: t };
        this.tracks.push(k);
      }
      this.updateHand(k, h, t);
    });
    this.updateDraft(t);
  },
  updateHand(k, h, t) {
    const f = (name, p) => {
      const s = k.f[name] ??= { x: new Smooth(), y: new Smooth() };
      return { x: s.x.next(p.x, t), y: s.y.next(p.y, t) };
    };
    k.pts = h;
    const size = Math.max(dist(h[0], h[9]), 1.4 * dist(h[5], h[17]), 1);
    const ratio = dist(h[4], h[8]) / size;
    k.mid = f('mid', { x: (h[4].x + h[8].x) / 2, y: (h[4].y + h[8].y) / 2 });
    const tip = f('tip', h[8]);

    if (!k.pinch && ratio < HAND.pinchOn) { k.pinch = true; k.grab = this.endNear(k.mid); if (k.grab) Sound.stop(k.grab.s); }
    else if (k.pinch && ratio > HAND.pinchOff) { k.pinch = false; k.mute = t + HAND.afterReleaseMute; this.letGo(k); }

    if (k.grab && !strings.includes(k.grab.s)) k.grab = null;
    if (k.grab) { // 抓着端点：跟着手移动并重新定音
      const { s, end } = k.grab;
      k.past.push({ t, p: k.mid }); k.past = k.past.filter(o => t - o.t < 400);
      s[end] = toRel(k.mid);
      this.retune(s);
    }
    // 拨弦：只用张开的手的食指尖
    const open = !k.pinch && ratio > HAND.pluckOpen && t > k.mute;
    if (open && k.tip && t - k.tipT < HAND.bridge) {
      const ms = t - k.tipT;
      for (const s of k.blocked) if (!strings.includes(s) || lineDist(s, k.tip) > HAND.rearm) k.blocked.delete(s);
      if (dist(k.tip, tip) / Math.max(ms, 4) / H * 1000 > HAND.minPluckSpeed) sweep(k.tip, tip, ms, k.blocked);
    }
    k.tip = open ? tip : null; k.tipT = t; k.seen = t;
  },
  endNear(p) {
    let best = null, bd = HAND.grab;
    for (const s of strings) for (const end of ['a', 'b']) {
      if (this.tracks.some(k => k.grab && k.grab.s === s && k.grab.end === end)) continue;
      const d = dist(p, toPx(s[end]));
      if (d < bd) { bd = d; best = { s, end }; }
    }
    return best;
  },
  retune(s) {
    s.len = Math.max(dist(toPx(s.a), toPx(s.b)) / W, .001);
    s.midi = lengthToMidi(s.len); s.free = picked.mode === 'free';
    s.born = now();
  },
  letGo(k) {
    if (!k.grab) return;
    const { s, end } = k.grab, at = this.before(k.past, now());
    if (at) s[end] = toRel(at.p);
    k.grab = null; k.past = [];
    // 把端点推到另一端附近再松手 = 删除这根弦
    if (dist(toPx(s.a), toPx(s.b)) / W < CFG.minLength) removeString(s); else this.retune(s);
  },
  before(list, t) {
    const old = list.filter(o => t - o.t >= HAND.releaseBack);
    return old[old.length - 1] || list[0];
  },
  updateDraft(t) {
    const p = this.tracks.filter(k => k.pinch && !k.grab);
    if (p.length >= 2) {
      this.bothSince ||= t;
      if (t - this.bothSince < HAND.pinchDwell) return;
      const a = p[0].mid, b = p[1].mid, len = dist(a, b) / W;
      this.draft = len < CFG.minLength ? null : { a, b, midi: lengthToMidi(len) };
      this.trail.push({ t, a, b }); this.trail = this.trail.filter(o => t - o.t < 400);
    } else {
      if (this.trail.length) { const o = this.before(this.trail, t); addString(o.a, o.b); }
      this.draft = null; this.trail = []; this.bothSince = 0;
    }
  },

  drawVideo() {
    const c = this.cover(), t = now(), avatar = Face.wanted(), blur = avatar && Face.blurNow(t);
    g2.save(); g2.translate(W, 0); g2.scale(-1, 1);
    g2.drawImage(blur ? Face.blurred(this.video) : this.video, W - c.x - c.w, c.y, c.w, c.h);
    g2.restore();
    g2.fillStyle = `rgba(24,13,7,${HAND.dim})`; g2.fillRect(0, 0, W, H);
    if (avatar) Face.draw(t, blur);
  },
  draw() {
    const t = now();
    for (const k of this.tracks) {
      if (t - k.seen > 120) continue;
      g2.strokeStyle = 'rgba(243,226,199,.3)'; g2.lineWidth = 1; g2.beginPath();
      for (const [i, j] of BONES) { g2.moveTo(k.pts[i].x, k.pts[i].y); g2.lineTo(k.pts[j].x, k.pts[j].y); }
      g2.stroke();
      if (k.pinch) {
        g2.fillStyle = k.grab ? '#ffc46b' : '#fff3df';
        g2.beginPath(); g2.arc(k.mid.x, k.mid.y, 9, 0, 7); g2.fill();
      } else for (const i of [4, 8]) {
        g2.strokeStyle = i === 8 ? '#ffc46b' : 'rgba(243,226,199,.7)'; g2.lineWidth = 2;
        g2.beginPath(); g2.arc(k.pts[i].x, k.pts[i].y, i === 8 ? 10 : 7, 0, 7); g2.stroke();
      }
    }
    if (this.draft) drawString(this.draft.a, this.draft.b, this.draft.midi, .12, t, .5, 1,
      `${noteName(this.draft.midi)}  ${Math.round(midiToFreq(this.draft.midi))} Hz`, nextHue);
    if (!this.tracks.length) {
      g2.font = '18px "Microsoft YaHei", sans-serif'; g2.textAlign = 'center'; g2.textBaseline = 'middle';
      g2.fillStyle = '#f3e2c7'; g2.fillText(tr('noHand'), W / 2, H - 70);
    }
  }
};

/* ================= 按钮 ================= */
const startBox = document.getElementById('start'), camBtn = document.getElementById('cam'), goCam = document.getElementById('goCam');
function cameraStopped(key = 'hintMouse') { hintKey = key; applyText(); }
async function openCamera() {
  Sound.init();
  goCam.disabled = camBtn.disabled = true;
  msgKey = 'loading'; applyText();
  startBox.hidden = false;
  try {
    await Hands.start();
    startBox.hidden = true; msgKey = ''; hintKey = 'hintHand';
  } catch (e) {
    console.warn(e);
    msgKey = TEXT.zh[e.name] ? e.name : 'loadFailed';
  }
  applyText();
  goCam.disabled = camBtn.disabled = false;
}
goCam.onclick = openCamera;
camBtn.onclick = () => { if (Hands.on) { Hands.stop(); cameraStopped(); } else openCamera(); };
