/* 动效层：入场、滚动进场。
   --------------------------------------------------------------------------
   配套 motion.css（那里写着为什么这么设计）。这里做两件事：

   ① 给要出场的元素挂上「还没出场」的标记；
   ② 元素滚进视口时换成「出场」。

   为什么先藏再放：**JS 不跑时元素必须照常显示**。
   所以「藏」的样式挂在 ``html.mo-ready`` 下面 —— 那行 class 是这里加的；
   没有 JS 就没有 mo-ready，页面还是原来那个样子。

   页与页之间的「穿梭」在 warp.js 里 —— 那是另一件事（整页跳转的过场），
   不跟入场混在一个文件里。
 */
(function () {
  'use strict';

  /* 系统里说了「少动」就不动 —— 少动是设置里的一条，不是可选项。 */
  if (window.matchMedia
      && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    return;
  }

  /* 要出场的元素。写在一处，改口径只改这一行。 */
  var TARGETS = [
    '.section',                 /* 报告页的每个区块 */
    '.analysis-part',           /* 三部分正文 */
    '.mode-item', '.coach-item', '.wrong-list > li',
    '.lane-sec', '.item',       /* 对局记录页 */
    '.mode', '.phase-h', '.hero-card',   /* 规则库页 */
    '.footer'
  ].join(', ');

  var els = document.querySelectorAll(TARGETS);
  if (!els.length) return;

  /* 同组的第几条 → 依次错开一点（CSS 里用 --mo-i 取）。
     「同组」是同一个父元素下的兄弟：一条分路下的几局之间错开，
     不会跟另一条分路串成一长串。 */
  var groups = new Map();
  Array.prototype.forEach.call(els, function (el) {
    var key = el.parentElement || document.body;
    var i = groups.get(key) || 0;
    groups.set(key, i + 1);
    /* 最多错开 8 条 —— 再多就等太久了。 */
    el.style.setProperty('--mo-i', Math.min(i, 8));
    el.classList.add('mo-wait');
  });
  document.documentElement.classList.add('mo-ready');

  /* 老浏览器没有 IntersectionObserver：一次全放出来，别让人等。 */
  if (!window.IntersectionObserver) {
    Array.prototype.forEach.call(els, function (el) {
      el.classList.remove('mo-wait');
      el.classList.add('mo-in');
    });
    return;
  }

  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (!entry.isIntersecting) return;
      entry.target.classList.remove('mo-wait');
      entry.target.classList.add('mo-in');
      io.unobserve(entry.target);   /* 只演一次 */
    });
  }, {
    /* 元素刚露头一点就开始动，不等到完全进视口。 */
    rootMargin: '0px 0px -6% 0px',
    threshold: .06
  });

  Array.prototype.forEach.call(els, function (el) { io.observe(el); });
})();
