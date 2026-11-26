/* 人脸换成卡通头像。和手部识别一样，画面只在本机处理。 */

const FACE_MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const FACE = {
  keep: 1000,     // 认不出脸时，头像在原地多留多少毫秒
  blurFor: 3000,  // 之后画面再模糊多少毫秒（以防真脸露出来），然后才当作人已离开
  every: 2,       // 每几帧视频识别一次脸，减轻电脑负担
  size: 0.85      // 头像大小，盖不住脸就调大，太大就调小
};

const Face = {
  model: null, loading: null, failed: false, tracks: [], count: 0, small: null,
  wanted: () => picked.view === 'cat' || picked.view === 'dog',

  load() {
    this.loading ||= (async () => {
      const { FilesetResolver, FaceLandmarker } = await import(MP + '/vision_bundle.mjs');
      const files = await FilesetResolver.forVisionTasks(MP + '/wasm');
      const make = delegate => FaceLandmarker.createFromOptions(files, {
        baseOptions: { modelAssetPath: FACE_MODEL, delegate }, runningMode: 'VIDEO', numFaces: 3, outputFaceBlendshapes: true });
      try { this.model = await make('GPU'); } catch (e) { this.model = await make('CPU'); }
      this.failed = false;
    })().catch(e => { console.warn(e); this.failed = true; this.loading = null; });
    return this.loading;
  },

  tick(video, t, c) {
    if (!this.wanted() || !this.model || this.count++ % FACE.every) return;
    let res;
    try { res = this.model.detectForVideo(video, t); } catch (e) { console.warn(e); return; }
    const faces = res.faceLandmarks.map((pts, i) =>
      this.measure(pts.map(p => ({ x: c.x + (1 - p.x) * c.w, y: c.y + p.y * c.h })), res.faceBlendshapes[i]));
    this.tracks = this.tracks.filter(k => t - k.seen < FACE.keep + FACE.blurFor);
    // 和上一次的脸按距离配对，这样头像不会在两个人之间跳
    const pairs = [];
    faces.forEach((f, i) => this.tracks.forEach(k => pairs.push({ i, k, d: dist(f, k) })));
    pairs.sort((a, b) => a.d - b.d);
    const done = new Set(), used = new Set();
    for (const { i, k, d } of pairs) {
      if (done.has(i) || used.has(k) || d > k.r * 1.5) continue;
      done.add(i); used.add(k);
      const f = faces[i], fresh = t - k.seen > 300;
      for (const key of ['x', 'y', 'r', 'roll', 'yaw']) k[key] = fresh ? f[key] : k[key] + (f[key] - k[key]) * .6;
      for (const key of ['blinkL', 'blinkR', 'jaw', 'smile']) k[key] = f[key];
      k.seen = t;
    }
    faces.forEach((f, i) => { if (!done.has(i)) this.tracks.push({ ...f, seen: t }); });
  },
  measure(p, shapes) {
    const s = {};
    for (const c of shapes ? shapes.categories : []) s[c.categoryName] = c.score;
    const top = p[10], chin = p[152], h = dist(top, chin), w = dist(p[234], p[454]);
    const [a, b] = p[33].x < p[263].x ? [p[33], p[263]] : [p[263], p[33]];
    const mid = { x: (p[234].x + p[454].x) / 2 };
    return {
      x: (top.x + chin.x) / 2 + (top.x - chin.x) * .1,
      y: (top.y + chin.y) / 2 + (top.y - chin.y) * .1,
      r: Math.max(.75 * h, .9 * w) * FACE.size,
      roll: Math.atan2(b.y - a.y, b.x - a.x),
      yaw: Math.max(-.4, Math.min(.4, (p[1].x - mid.x) / (w || 1))),
      blinkL: s.eyeBlinkLeft || 0, blinkR: s.eyeBlinkRight || 0,
      jaw: s.jawOpen || 0, smile: ((s.mouthSmileLeft || 0) + (s.mouthSmileRight || 0)) / 2
    };
  },

  // 模型还没好、一个脸都没认出、或者有人的脸刚跟丢：整个画面模糊
  blurNow(t) {
    if (!this.model) return true;
    return !this.tracks.some(k => t - k.seen < FACE.keep) || this.tracks.some(k => t - k.seen >= FACE.keep);
  },
  // 缩到很小再放大 = 模糊，又快又认不出人
  blurred(video) {
    const w = 32, h = Math.max(1, Math.round(32 * (video.videoHeight || 9) / (video.videoWidth || 16)));
    if (!this.small) this.small = document.createElement('canvas');
    if (this.small.width !== w || this.small.height !== h) { this.small.width = w; this.small.height = h; }
    this.small.getContext('2d').drawImage(video, 0, 0, w, h);
    return this.small;
  },

  draw(t, blurred) {
    if (blurred) {
      g2.font = '16px "Microsoft YaHei", sans-serif'; g2.textAlign = 'center'; g2.textBaseline = 'middle';
      g2.fillStyle = '#f3e2c7';
      g2.fillText(tr(this.failed ? 'faceFailed' : this.model ? 'faceBlur' : 'faceLoading'), W / 2, H - 100);
      return;
    }
    for (const k of this.tracks) if (t - k.seen < FACE.keep) drawHead(k, picked.view);
  }
};

/* ================= 原创卡通头像（在 200×200 的格子里画，再缩放到脸的大小） ================= */
const INK = '#3a2012';
function blob(x, y, rx, ry, fill, rot = 0) {
  g2.beginPath(); g2.ellipse(x, y, rx, ry, rot, 0, 7);
  g2.fillStyle = fill; g2.fill(); g2.stroke();
}
function eyes(k, dx) {
  for (const [side, blink] of [[-1, k.blinkL], [1, k.blinkR]]) {
    const x = side * 36 + dx, y = -6, open = Math.max(0, 1 - blink * 1.3);
    if (open < .25) { // 闭眼：一条弯线
      g2.beginPath(); g2.arc(x, y - 4, 12, .2 * Math.PI, .8 * Math.PI); g2.lineWidth = 5; g2.stroke(); g2.lineWidth = 4;
      continue;
    }
    g2.beginPath(); g2.ellipse(x, y, 13, 16 * open, 0, 0, 7); g2.fillStyle = '#24140c'; g2.fill();
    g2.beginPath(); g2.arc(x + 4, y - 5 * open, 4.5 * open, 0, 7); g2.fillStyle = '#fff'; g2.fill();
  }
}
function mouth(k, dx, y, dog) {
  const open = Math.max(0, k.jaw - .08) / .6;
  if (open > .05) {
    const h = 6 + 30 * Math.min(1, open);
    g2.beginPath(); g2.ellipse(dx, y + h / 2, 15, h / 2, 0, 0, 7); g2.fillStyle = '#7a2420'; g2.fill(); g2.stroke();
    blob(dx, y + h * (dog ? 1.05 : .8), 10, dog ? 10 + h * .3 : 7, '#ff8f9a');
    return;
  }
  const lift = 6 * k.smile; // 笑的时候嘴角往上
  g2.beginPath();
  g2.moveTo(dx - 20, y - lift); g2.quadraticCurveTo(dx - 10, y + 10, dx, y);
  g2.quadraticCurveTo(dx + 10, y + 10, dx + 20, y - lift);
  g2.stroke();
}
function drawHead(k, kind) {
  const dx = k.yaw * 70; // 转头时五官跟着偏
  g2.save();
  g2.translate(k.x, k.y); g2.rotate(k.roll); g2.scale(k.r / 100, k.r / 100);
  g2.lineWidth = 4; g2.strokeStyle = INK; g2.lineJoin = 'round'; g2.lineCap = 'round';
  if (kind === 'cat') {
    for (const s of [-1, 1]) { // 耳朵
      g2.beginPath(); g2.moveTo(s * 92, -30); g2.lineTo(s * 74, -128); g2.lineTo(s * 18, -82); g2.closePath();
      g2.fillStyle = '#f2a65a'; g2.fill(); g2.stroke();
      g2.beginPath(); g2.moveTo(s * 76, -48); g2.lineTo(s * 70, -106); g2.lineTo(s * 36, -78); g2.closePath();
      g2.fillStyle = '#ffb3b0'; g2.fill();
    }
    blob(0, 0, 104, 94, '#f2a65a');
    g2.strokeStyle = '#c8702f'; g2.lineWidth = 7; // 额头条纹
    for (const x of [-22, 0, 22]) { g2.beginPath(); g2.moveTo(x + dx * .5, -88); g2.lineTo(x * .8 + dx * .5, -62); g2.stroke(); }
    g2.strokeStyle = INK; g2.lineWidth = 4;
    g2.save(); g2.lineWidth = 0; g2.strokeStyle = 'transparent'; blob(dx, 34, 50, 36, '#fff1de'); g2.restore();
    for (const s of [-1, 1]) { g2.beginPath(); g2.ellipse(s * 64 + dx, 14, 14, 8, 0, 0, 7); g2.fillStyle = 'rgba(255,120,120,.4)'; g2.fill(); } // 腮红
    eyes(k, dx);
    g2.beginPath(); g2.moveTo(dx - 9, 16); g2.lineTo(dx + 9, 16); g2.lineTo(dx, 26); g2.closePath();
    g2.fillStyle = '#ff7f8f'; g2.fill(); g2.stroke();
    mouth(k, dx, 30, false);
    g2.lineWidth = 2.5; // 胡须
    for (const s of [-1, 1]) for (const a of [-.15, .1]) {
      g2.beginPath(); g2.moveTo(s * 52 + dx, 38); g2.lineTo(s * 118 + dx, 38 + a * 120); g2.stroke();
    }
  } else {
    for (const s of [-1, 1]) blob(s * 92, 6, 30, 64, '#8a5634', s * .25); // 耷拉的耳朵
    blob(0, 0, 98, 94, '#dca46a');
    g2.save(); g2.lineWidth = 0; g2.strokeStyle = 'transparent';
    blob(36 + dx, -8, 28, 30, '#b77b45'); // 一只眼睛周围的深色斑
    blob(dx, 40, 54, 40, '#f7e6cc');
    g2.restore();
    eyes(k, dx);
    blob(dx, 18, 17, 12, '#24140c');
    g2.beginPath(); g2.arc(dx - 5, 14, 4, 0, 7); g2.fillStyle = '#fff'; g2.fill();
    g2.beginPath(); g2.moveTo(dx, 30); g2.lineTo(dx, 36); g2.stroke();
    mouth(k, dx, 36, true);
  }
  g2.restore();
}
