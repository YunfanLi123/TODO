/* 内层页面的氛围背景。
 *
 * 手法出自 GitHub 的 atmosphere-background skill
 * （github.com/mengto/skills · agent-skills/web-design/atmosphere-background）：
 * 缓慢漂移的竖向光褶 + screen 混合 + 角落的一团辉光。
 *
 * 规矩（skill 里的 Scope 一节）：
 *   只作用在**背景层** —— 版式、字体、内容一个字都不动。
 *
 * 三条自我约束：
 *   1. 画布不可用时页面照常可读（底色在 html 上，不在画布里）；
 *   2. 「减少动态效果」时只画一帧，不动；
 *   3. 标签页藏起来就停 —— 满屏动画在后台跑是白烧电。
 */
(function () {
  'use strict';

  var canvas = document.getElementById('atmo-canvas');
  if (!canvas) return;
  var ctx = canvas.getContext && canvas.getContext('2d');
  if (!ctx) return;   // 画不出来就算了，页面照常用

  var reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var w = 0, h = 0, raf = 0, t = 0, last = 0;
  var accent = [124, 151, 255];
  var dark = false;
  var folds = [];

  // 光褶的数量。十八条是 skill 给的参考值 —— 少一档就单薄，多一档就糊。
  var COUNT = 18;

  function seed() {
    folds = [];
    for (var i = 0; i < COUNT; i++) {
      folds.push({
        x: (i + 0.5) / COUNT + (Math.random() - 0.5) * 0.05,
        w: 0.035 + Math.random() * 0.055,
        amp: 0.008 + Math.random() * 0.02,
        spd: 0.05 + Math.random() * 0.07,
        ph: Math.random() * Math.PI * 2,
        a: 0.35 + Math.random() * 0.65
      });
    }
  }

  function readTheme() {
    var cs = getComputedStyle(document.documentElement);
    var hex = (cs.getPropertyValue('--accent') || '').trim();
    var m = /^#?([0-9a-f]{6})$/i.exec(hex);
    if (m) {
      var n = parseInt(m[1], 16);
      accent = [(n >> 16) & 255, (n >> 8) & 255, n & 255];
    }
    dark = document.documentElement.getAttribute('data-theme') === 'dark';
  }

  function rgba(a) {
    return 'rgba(' + accent[0] + ',' + accent[1] + ',' + accent[2] + ',' + a + ')';
  }

  function resize() {
    var rect = canvas.getBoundingClientRect();
    // 像素比压到 1.5：满屏背景不需要 2 倍，再高只是白烧 GPU，
    // 而光褶本来就是柔和的东西，看不出差别。
    var ratio = Math.min(window.devicePixelRatio || 1, 1.5);
    w = rect.width;
    h = rect.height;
    canvas.width = Math.round(w * ratio);
    canvas.height = Math.round(h * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  }

  function draw(now) {
    raf = window.requestAnimationFrame(draw);
    // 这么慢的光，30 帧足够 —— 60 帧只是把同一件事算两遍。
    if (now - last < 32) return;
    last = now;
    t += 0.016;

    ctx.clearRect(0, 0, w, h);
    if (!w || !h) return;

    // 深色底上用 screen（叠在一起自然变亮）；
    // 浅色底上用 multiply（压出一点冷调的柔光）。
    ctx.globalCompositeOperation = dark ? 'screen' : 'multiply';

    for (var i = 0; i < folds.length; i++) {
      var f = folds[i];
      var x = (f.x + Math.sin(t * f.spd + f.ph) * f.amp) * w;
      // 越靠右越亮：亮度往一处聚，画面才有焦点，而不是一片均匀的雾。
      var alpha = f.a * (dark ? 0.30 : 0.10) * (0.3 + 0.7 * (x / w));
      var rx = Math.max(1, f.w * w);
      ctx.save();
      // 光褶从底边升起、向上淡出：底部最亮，顶部化开。
      ctx.translate(x, h * 1.02);
      ctx.scale(1, (h * 0.96) / rx);
      var g = ctx.createRadialGradient(0, 0, 0, 0, 0, rx);
      g.addColorStop(0, rgba(alpha));
      g.addColorStop(0.5, rgba(alpha * 0.34));
      g.addColorStop(1, rgba(0));
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(0, 0, rx, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // 角落的辉光：焦点那一处最亮。
    var bx = w * 0.86, by = h * 0.95;
    var br = Math.max(w, h) * 0.52;
    var bg = ctx.createRadialGradient(bx, by, 0, bx, by, br);
    bg.addColorStop(0, rgba(dark ? 0.36 : 0.12));
    bg.addColorStop(1, rgba(0));
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);

    // 一道慢慢扫过的窄光。
    ctx.globalCompositeOperation = 'screen';
    var sx = w * (0.5 + 0.22 * Math.sin(t * 0.11));
    var sg = ctx.createLinearGradient(sx - 30, 0, sx + 30, 0);
    sg.addColorStop(0, 'rgba(255,255,255,0)');
    sg.addColorStop(0.5, dark ? 'rgba(198,212,255,0.15)' : 'rgba(255,255,255,0.12)');
    sg.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = sg;
    ctx.fillRect(sx - 30, 0, 60, h);

    ctx.globalCompositeOperation = 'source-over';
  }

  function start() {
    if (raf) return;
    raf = window.requestAnimationFrame(draw);
  }

  function stop() {
    if (!raf) return;
    window.cancelAnimationFrame(raf);
    raf = 0;
  }

  seed();
  readTheme();
  resize();
  window.addEventListener('resize', resize);
  if (window.ResizeObserver) {
    new ResizeObserver(resize).observe(canvas);
  }

  // 主题切换后要重取颜色：深浅两套底色下，光的强度差一档。
  if (window.MutationObserver) {
    new MutationObserver(function () {
      readTheme();
      if (reduced) { draw(performance.now() + 999); }
    }).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  }

  // 标签页藏起来就停 —— 满屏动画在后台跑是白烧电。
  document.addEventListener('visibilitychange', function () {
    if (document.hidden) { stop(); } else if (!reduced) { start(); }
  });

  if (reduced) {
    // 减少动态效果：只画一帧，静静待着。
    draw(performance.now() + 999);
    stop();
  } else {
    start();
  }
})();
