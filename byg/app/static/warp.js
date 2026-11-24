/* 页与页之间的过场：**从被点的那个元素开始，丝滑接到下一页**。
   --------------------------------------------------------------------------
   这一段管三件事：

   1. **被点的元素自己飞过去** —— 浏览器原生能力（View Transitions）：
      旧页给那个元素挂一个 ``view-transition-name``，新页给**标题**挂同一个
      名字，中间浏览器自动把前者变形过去（位置、大小、内容的交叉淡化全包）。
      点「探索玩法」进去，是那颗按钮自己长大成新页的标题。

   2. **星星从那个元素那儿甩出来** —— 一块全屏 canvas 自己画的星野
      （浏览器的原生过渡只是两张快照的交叉淡化，**没有「冲过去」这件事**）。

   3. **新页里星星减速散场** —— 到站了，页面从星空里露出来。

   少动模式（``prefers-reduced-motion``）下整套不启动：直接换页。
 */
(function () {
  'use strict';

  if (window.matchMedia
      && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    return;
  }

  /* 旧页 → 新页的暗号。整页跳转会销毁旧页的 JS，新页只能靠这个知道
     「我是被穿梭过来的」。用完即清：手动刷新不会再演一遍。 */
  var KEY = 'mo-warp';
  /* 两页之间对上的那个过渡名。 */
  var FLY = 'mo-fly';

  var COUNT = 650;       /* 星星数量 */
  var Z_FAR = 1.0;
  var Z_NEAR = 0.08;

  var cameByWarp = false;
  try {
    cameByWarp = !!sessionStorage.getItem(KEY);
    if (cameByWarp) sessionStorage.removeItem(KEY);
  } catch (e) {
    /* 隐私模式下 sessionStorage 可能不可用 —— 没有过场，但页面照常。 */
  }

  /* ---------- 新页：标题接住旧页飞过来的那个元素 ----------
     注意两件事：
     ① ``pagereveal`` 是在 **window** 上触发的（挂 document 上接不到）。
     ② 从缓存里回来（bfcache）时浏览器**跳过**过渡 —— 这时候别插手，
        否则页面像在不停自己动（那套参考里点名的失败模式）。 */
  window.addEventListener('pagereveal', function (e) {
    if (!cameByWarp || !e.viewTransition) return;
    var target = document.querySelector(
      '.hero-id h1, .page-head h1, .hero-inner h1, h1');
    if (!target || !('viewTransitionName' in target.style)) return;
    target.style.viewTransitionName = FLY;
    /* **用完必须清掉**：名字留着，下一页再挂同一个名字，浏览器会判成
       重复 → 直接跳过过渡（第二次点就没动画了）。 */
    e.viewTransition.finished.then(function () {
      target.style.viewTransitionName = '';
    })['catch'](function () { /* 过渡被打断就算了 */ });
  });

  function spawn() {
    return {
      x: Math.random() * 2 - 1,
      y: Math.random() * 2 - 1,
      z: Z_NEAR + Math.random() * (Z_FAR - Z_NEAR),
      /* 每颗星一点点色差：全是同一个白，像颗粒不像星空。 */
      hue: 205 + Math.random() * 45,
      px: null,
      py: null,
      /* 拖尾记的是**几帧之前**的位置 —— 只记上一帧的话，
         星星的尾巴只有十几像素，看不出「从那儿甩出来」的放射感。 */
      age: 0
    };
  }

  /**
   * 画一块星野。
   * @param mode  'depart' 出发（旧页）｜ 'arrive' 到站（新页）
   * @param ox,oy 视线的灭点 —— 出发时取被点元素的中心，星星就从那儿甩出来
   */
  function warp(mode, ox, oy) {
    var canvas = document.createElement('canvas');
    canvas.className = 'mo-warp';
    document.body.appendChild(canvas);
    var ctx = canvas.getContext('2d');
    if (!ctx) { canvas.remove(); return { close: function () {} }; }

    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var W = 0, H = 0, cx = 0, cy = 0;

    function resize() {
      W = canvas.clientWidth || window.innerWidth;
      H = canvas.clientHeight || window.innerHeight;
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      cx = W / 2;
      cy = H / 2;
      /* 灭点可以偏离屏幕中心：出发时就是被点的那个元素。 */
      if (ox !== undefined) { cx = ox; cy = oy; }
    }
    resize();
    window.addEventListener('resize', resize);

    var stars = [];
    for (var i = 0; i < COUNT; i++) stars.push(spawn());

    /* 出发：从慢起步，越走越快；到达：从快开始，慢慢停住。 */
    var speed = mode === 'depart' ? 0.10 : 2.2;
    var speedTo = mode === 'depart' ? 2.6 : 0;
    /* 这一层自己的不透明度：出发时涨上去把旧页盖住，到达时落下去露出新页。 */
    var veil = mode === 'depart' ? 0 : 1;
    var veilTo = mode === 'depart' ? 1 : 0;

    var last = performance.now();
    var age = 0;
    var raf = 0;

    function frame(now) {
      var dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      age += dt;
      speed += (speedTo - speed) * Math.min(dt * (mode === 'depart' ? 2.0 : 2.6), 1);
      veil += (veilTo - veil) * Math.min(dt * (mode === 'depart' ? 6 : 7), 1);

      /* **保险**：换页没发生（比如离线导出的文件点了站内的链接），
         就把这一层收掉 —— 别让人盯着一片星空走不了。 */
      if (mode === 'depart' && age > 1.5) {
        speedTo = 0;
        veilTo = 0;
      }

      ctx.clearRect(0, 0, W, H);
      /* 底：越穿梭越黑（到站时再一点点退回去）。 */
      ctx.fillStyle = 'rgba(4, 6, 14, ' + (0.88 * veil) + ')';
      ctx.fillRect(0, 0, W, H);
      /* 中心那点光：像银河的核，冲得越快越亮。 */
      var glow = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(W, H) * 0.55);
      glow.addColorStop(0, 'rgba(150, 180, 255, ' + (0.30 * veil * Math.min(speed, 1)) + ')');
      glow.addColorStop(1, 'rgba(150, 180, 255, 0)');
      ctx.fillStyle = glow;
      ctx.fillRect(0, 0, W, H);

      for (var i = 0; i < stars.length; i++) {
        var s = stars[i];
        s.z -= speed * dt;
        if (s.z <= Z_NEAR) {
          /* 冲到跟前了 → 换一颗新的从远处来。 */
          var fresh = spawn();
          fresh.z = Z_FAR;
          stars[i] = fresh;
          continue;
        }
        /* 透视：z 越小越靠前，投影就越往外甩 —— 这就是「冲过去」。 */
        var k = 0.95 / s.z;
        var sx = cx + s.x * k * cx;
        var sy = cy + s.y * k * cy;
        if (s.px !== null) {
          var dx = sx - s.px;
          var dy = sy - s.py;
          if (dx * dx + dy * dy > 0.4) {
            var near = 1 - s.z;                       /* 0 远 → 1 近 */
            var alpha = Math.min(1, near * 1.6) * veil;
            ctx.strokeStyle = 'hsla(' + s.hue + ', 92%, 82%, ' + (alpha * 0.95) + ')';
            ctx.lineWidth = Math.max(0.7, near * 3);
            ctx.beginPath();
            ctx.moveTo(s.px, s.py);
            ctx.lineTo(sx, sy);
            ctx.stroke();
          }
        }
        /* 位置每 3 帧挪一次 —— 尾巴就有三帧那么长，
           甩出来的放射感才出得来。 */
        s.age += 1;
        if (s.age % 3 === 0) {
          s.px = sx;
          s.py = sy;
        }
      }

      /* 到站了：星星停了、层淡完了 → 把这一页还给人。 */
      if (veil < 0.015 && speed < 0.04) {
        canvas.remove();
        window.removeEventListener('resize', resize);
        return;
      }
      raf = requestAnimationFrame(frame);
    }

    /* 先把第一帧画出来：新页的快照是在「首帧」上拍的，
       这一帧没有星星的话，浏览器拍到的就是一张空页面。 */
    frame(performance.now());
    raf = requestAnimationFrame(frame);

    return {
      /* 收掉这一层（换页前做，好让浏览器拍到干净的快照）。 */
      close: function () {
        speedTo = 0;
        veilTo = 0;
      }
    };
  }

  /* ---------- 新页：我是被穿梭过来的 → 星星减速散场 ---------- */
  if (cameByWarp) warp('arrive');

  /* ---------- 旧页：点站内链接 → 从那个元素开始冲出去 ---------- */
  document.addEventListener('click', function (e) {
    if (e.defaultPrevented || e.button !== 0) return;
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    var el = e.target.closest ? e.target.closest('a[href]') : null;
    if (!el) return;
    /* 下载、新窗口、页内锚点 —— 都不接。 */
    if (el.hasAttribute('download') || el.target) return;
    var href = el.getAttribute('href') || '';
    if (!href || href.charAt(0) === '#') return;
    var url;
    try { url = new URL(href, location.href); } catch (err) { return; }
    if (url.origin !== location.origin) return;          /* 外站不接 */
    if (url.pathname === location.pathname
        && url.search === location.search) return;        /* 同一页不接 */

    try { sessionStorage.setItem(KEY, '1'); } catch (err) { /* 忽略 */ }

    /* **被点的元素接住这一下** —— 浏览器会把它变形成新页的标题。 */
    if ('viewTransitionName' in el.style) el.style.viewTransitionName = FLY;

    /* 星星从它那儿甩出来。换页这段时间它就是「在加载」的提示 ——
       旧页还在，星星还在跑，新页一来就接上。 */
    var r = el.getBoundingClientRect();
    warp('depart', r.left + r.width / 2, r.top + r.height / 2);

    /* **不拦默认行为**：浏览器自己换页 —— 后退、Ctrl+点击、下载、
       外站链接的语义一点没动；星星演它的，页面换它的。 */
  });
})();
