/* ==========================================================================
   对局执行度报告 · 交互与图表
   零依赖：不引任何 CDN 图表库，折线全部用原生 SVG 画，保证离线可用。
   ========================================================================== */
(function () {
  'use strict';

  var SVG_NS = 'http://www.w3.org/2000/svg';

  /* ---------- 工具函数 ---------- */

  function el(tag, attrs, text) {
    var n = document.createElementNS(SVG_NS, tag);
    if (attrs) {
      for (var k in attrs) {
        if (Object.prototype.hasOwnProperty.call(attrs, k) && attrs[k] !== null &&
            attrs[k] !== undefined) {
          n.setAttribute(k, String(attrs[k]));
        }
      }
    }
    if (text !== null && text !== undefined) n.textContent = String(text);
    return n;
  }

  /* rate 统一在这里过一道闸：非有限数一律当作「无数据」。
     任何分支都不允许 NaN / Infinity 流进 SVG 属性。 */
  function isNum(v) {
    return typeof v === 'number' && isFinite(v);
  }

  function pct(v) {
    return isNum(v) ? (Math.round(v * 1000) / 10).toFixed(1) + '%' : '—';
  }

  function clamp01(v) {
    return v < 0 ? 0 : (v > 1 ? 1 : v);
  }

  /* ---------- 01 差距证据表展开 ---------- */

  function bindEvidenceToggles() {
    var toggles = document.querySelectorAll('.ev-toggle');
    Array.prototype.forEach.call(toggles, function (btn) {
      btn.addEventListener('click', function () {
        var target = document.getElementById(btn.getAttribute('aria-controls'));
        if (!target) return;
        var open = target.classList.toggle('open');
        btn.setAttribute('aria-expanded', open ? 'true' : 'false');
        var caret = btn.querySelector('.caret');
        if (caret) caret.textContent = open ? '▼' : '▶';
        // 同步文案，让折叠状态下也知道里面有什么
        var label = btn.childNodes[btn.childNodes.length - 1];
        if (label && label.nodeType === 3) {
          label.nodeValue = open
            ? ' 收起判定证据'
            : ' 判定证据 · ' + (target.querySelectorAll('tbody tr').length) + ' 项';
        }
      });
    });
  }

  /* ---------- 02 执行度曲线（原生 SVG） ---------- */

  var W = 980;
  var H = 300;
  var PAD = { top: 28, right: 26, bottom: 46, left: 52 };

  var C = {
    accent: '#E8A33D',
    muted: '#8FA1B3',
    line: '#243447',
    text: '#F2F6FA',
    panel: '#16212F'
  };

  function drawCurve() {
    var host = document.getElementById('curve-chart');
    if (!host) return;

    var data = { points: [], axis: { min: 0, max: 1 } };
    var island = document.getElementById('curve-data');
    if (island) {
      try {
        data = JSON.parse(island.textContent || '{}') || data;
      } catch (err) {
        data = { points: [], axis: { min: 0, max: 1 } };
      }
    }
    var pts = Array.isArray(data.points) ? data.points : [];
    if (!pts.length) return;

    var axis = data.axis || {};
    var lo = isNum(axis.min) ? axis.min : 0;
    var hi = isNum(axis.max) && axis.max > lo ? axis.max : 1;

    var innerW = W - PAD.left - PAD.right;
    var innerH = H - PAD.top - PAD.bottom;

    /* 单点时居中，避免除以 0 */
    var x = function (i) {
      return PAD.left + (pts.length === 1
        ? innerW / 2
        : (i / (pts.length - 1)) * innerW);
    };
    var y = function (rate) {
      var t = clamp01((rate - lo) / (hi - lo));
      return PAD.top + (1 - t) * innerH;
    };

    var svg = el('svg', {
      'class': 'chart-svg',
      viewBox: '0 0 ' + W + ' ' + H,
      preserveAspectRatio: 'xMidYMid meet',
      role: 'img',
      'aria-label': '规则执行度随对局变化的折线图'
    });

    /* --- 网格与 Y 轴刻度 --- */
    var steps = 4;
    for (var s = 0; s <= steps; s++) {
      var r = lo + (hi - lo) * (s / steps);
      var gy = y(r);
      svg.appendChild(el('line', {
        x1: PAD.left, y1: gy, x2: W - PAD.right, y2: gy,
        stroke: C.line, 'stroke-width': 1
      }));
      svg.appendChild(el('text', {
        x: PAD.left - 10, y: gy + 4, 'text-anchor': 'end',
        fill: C.muted, 'font-size': 11, 'font-family': 'Arial, sans-serif'
      }, pct(r)));
    }

    /* --- 版本切换处的竖直虚线 --- */
    /* data-points 里没有版本字段时（历史数据）自动跳过，不影响出图。 */
    var marks = [];
    for (var k = 1; k < pts.length; k++) {
      var pv = pts[k - 1].game_version;
      var cv = pts[k].game_version;
      if (pv && cv && pv !== cv) {
        var vx = (x(k - 1) + x(k)) / 2;
        marks.push({ x: vx, from: pv, to: cv });
        svg.appendChild(el('line', {
          x1: vx, y1: PAD.top - 6, x2: vx, y2: H - PAD.bottom,
          stroke: C.muted, 'stroke-width': 1.5,
          'stroke-dasharray': '5 5', opacity: 0.8
        }));
        svg.appendChild(el('text', {
          x: vx, y: PAD.top - 12, 'text-anchor': 'middle',
          fill: C.muted, 'font-size': 10.5, 'font-family': 'Arial, sans-serif'
        }, pv + ' → ' + cv));
      }
    }

    /* --- 折线：遇到 null 立即断开，绝不跨空洞连线 --- */
    var segs = [];
    var cur = [];
    for (var i = 0; i < pts.length; i++) {
      if (isNum(pts[i].compliance_rate)) {
        cur.push({ i: i, rate: pts[i].compliance_rate, cx: x(i), cy: y(pts[i].compliance_rate) });
      } else if (cur.length) {
        segs.push(cur);
        cur = [];
      }
    }
    if (cur.length) segs.push(cur);

    segs.forEach(function (seg) {
      var d = seg.map(function (p, idx) {
        return (idx === 0 ? 'M' : 'L') + p.cx.toFixed(1) + ' ' + p.cy.toFixed(1);
      }).join(' ');
      if (seg.length > 1) {
        svg.appendChild(el('path', {
          d: d, fill: 'none', stroke: C.accent,
          'stroke-width': 2.5, 'stroke-linejoin': 'round', 'stroke-linecap': 'round'
        }));
      } else {
        /* 孤立的有效点画不出线段，补一段极短的横线让它可见 */
        var p = seg[0];
        svg.appendChild(el('line', {
          x1: p.cx - 3, y1: p.cy, x2: p.cx + 3, y2: p.cy,
          stroke: C.accent, 'stroke-width': 2.5, 'stroke-linecap': 'round'
        }));
      }
    });

    /* --- 逐点标记 --- */
    pts.forEach(function (p, idx) {
      var cx = x(idx);
      var date = p.played_at || '';

      if (isNum(p.compliance_rate)) {
        var cy = y(p.compliance_rate);
        svg.appendChild(el('circle', {
          cx: cx, cy: cy, r: 4.5, fill: C.accent, stroke: C.panel, 'stroke-width': 2
        }));
        var hit = el('circle', { cx: cx, cy: cy, r: 14, fill: 'transparent' });
        hit.appendChild(el('title', {}, describePoint(p)));
        svg.appendChild(hit);

        svg.appendChild(el('text', {
          x: cx, y: cy - 13, 'text-anchor': 'middle',
          fill: C.accent, 'font-size': 11, 'font-family': 'Arial, sans-serif'
        }, pct(p.compliance_rate)));
      } else {
        /* 无规则触发：落在基线上方的空心圈。
           空洞本身就是信息 —— 不能用插值把它抹平。 */
        var hy = H - PAD.bottom + 9;
        svg.appendChild(el('circle', {
          cx: cx, cy: hy, r: 4, fill: C.panel, stroke: C.muted,
          'stroke-width': 1.5, 'stroke-dasharray': '2 2'
        }));
        var h2 = el('circle', { cx: cx, cy: hy, r: 14, fill: 'transparent' });
        h2.appendChild(el('title', {}, describePoint(p)));
        svg.appendChild(h2);
      }

      svg.appendChild(el('text', {
        x: cx, y: H - PAD.bottom + 30, 'text-anchor': 'middle',
        fill: C.muted, 'font-size': 10.5, 'font-family': 'Arial, sans-serif'
      }, date.slice(5)));
    });

    /* --- 坐标轴 --- */
    svg.appendChild(el('line', {
      x1: PAD.left, y1: H - PAD.bottom, x2: W - PAD.right, y2: H - PAD.bottom,
      stroke: C.line, 'stroke-width': 1.5
    }));
    svg.appendChild(el('text', {
      x: PAD.left, y: 14, fill: C.muted, 'font-size': 11,
      'font-family': 'Microsoft YaHei, sans-serif'
    }, '规则执行度'));

    host.appendChild(svg);
  }

  function describePoint(p) {
    var rate = isNum(p.compliance_rate)
      ? pct(p.compliance_rate)
      : '无规则触发（未计入）';
    var out = (p.played_at || '') + ' · 执行度 ' + rate +
      ' · 触发 ' + (p.triggered || 0) +
      ' · 符合 ' + (p.satisfied || 0) +
      ' · 偏离 ' + (p.deviated || 0);
    if (p.game_version) out += ' · 版本 ' + p.game_version;
    return out;
  }

  /* ---------- 03 分类进度条入场 ---------- */

  function animateBars() {
    var fills = document.querySelectorAll('.cat-fill');
    Array.prototype.forEach.call(fills, function (f) {
      var w = f.style.width;
      f.style.width = '0%';
      // 下一帧再设回目标宽度，触发一次过渡动画
      requestAnimationFrame(function () {
        requestAnimationFrame(function () { f.style.width = w; });
      });
    });
  }

  function boot() {
    bindEvidenceToggles();
    drawCurve();
    animateBars();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
