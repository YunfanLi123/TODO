/* 点狄仁杰或诸葛亮 → 打开聊天面板，跟他聊。
   --------------------------------------------------------------------------
   回答的依据是**操作模式库**与**这一局的报告**（后端 app/report/chat.py），
   前端只管把话说出去、把答案显示出来。
 */
(function () {
  'use strict';

  var panel = document.getElementById('chat');
  if (!panel) return;

  var nameEl = document.getElementById('chatName');
  var roleEl = document.getElementById('chatRole');
  var logEl = document.getElementById('chatLog');
  var formEl = document.getElementById('chatForm');
  var inputEl = document.getElementById('chatInput');
  var closeEl = document.getElementById('chatClose');

  /* 兜底的人设 —— 后端没给（老包）时页面也还能用。 */
  var FALLBACK = {
    drj: { name: '狄仁杰', role: '判官', greeting: '坐。哪一段不对劲，说来听听。' },
    zgl: { name: '诸葛亮', role: '军师', greeting: '这局我看过了。要问什么？' }
  };
  var people = FALLBACK;
  var who = 'zgl';
  var history = [];
  var busy = false;

  function add(cls, text) {
    var div = document.createElement('div');
    div.className = 'chat-msg ' + cls;
    /* 文本就是文本：回答是模型写的，不当 HTML 解析 */
    div.textContent = text;
    logEl.appendChild(div);
    logEl.scrollTop = logEl.scrollHeight;
    return div;
  }

  function paintHead() {
    var p = people[who] || FALLBACK.zgl;
    nameEl.textContent = p.name;
    roleEl.textContent = p.role;
    inputEl.placeholder = '问' + p.name + '一句…';
  }

  function open(character) {
    who = people[character] ? character : 'zgl';
    if (panel.hidden) {
      history = [];
      logEl.textContent = '';
      var p = people[who] || FALLBACK.zgl;
      add('them', p.greeting);
    }
    panel.hidden = false;
    panel.classList.add('is-open');
    paintHead();
    markTalking();
    inputEl.focus();
  }

  function close() {
    panel.hidden = true;
    panel.classList.remove('is-open');
    markTalking();
  }

  /* 谁在说话，谁轻轻点个头 */
  function markTalking() {
    ['drj', 'zgl'].forEach(function (key) {
      document.querySelectorAll('.' + (key === 'drj' ? 'fg-drj' : 'fg-zgl'))
        .forEach(function (g) { g.classList.toggle('is-talking', !panel.hidden && key === who); });
    });
  }

  function send(text) {
    if (busy) return;
    var msg = (text || '').trim();
    if (!msg) return;
    busy = true;
    var sendBtn = formEl.querySelector('.chat-send');
    if (sendBtn) sendBtn.disabled = true;
    inputEl.value = '';
    add('me', msg);
    var thinking = add('thinking', '正在翻操作模式库…');

    fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        character: who,
        message: msg,
        history: history
      })
    }).then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    }).then(function (data) {
      thinking.remove();
      var reply = (data && data.reply) || '（这句我没答上来）';
      add('them', reply);
      history.push({ role: 'user', text: msg });
      history.push({ role: 'them', text: reply });
    }).catch(function (err) {
      thinking.remove();
      add('warn', '（没答上来：' + (err && err.message ? err.message : err) + '）');
    }).then(function () {
      busy = false;
      if (sendBtn) sendBtn.disabled = false;
      inputEl.focus();
    });
  }

  /* ---------- 事件 ---------- */
  formEl.addEventListener('submit', function (e) {
    e.preventDefault();
    send(inputEl.value);
  });
  closeEl.addEventListener('click', close);
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && !panel.hidden) close();
  });

  /* 那两个人：点一下就开聊 */
  document.querySelectorAll('.fg-drj, .fg-zgl').forEach(function (g) {
    var key = g.classList.contains('fg-drj') ? 'drj' : 'zgl';
    g.setAttribute('tabindex', '0');
    g.setAttribute('role', 'button');
    g.setAttribute('aria-label', key === 'drj' ? '和狄仁杰聊聊' : '和诸葛亮聊聊');
    var title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
    title.textContent = '点我聊聊';
    g.appendChild(title);
    g.addEventListener('click', function () { open(key); });
    g.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(key); }
    });
  });

  /* 人设（名字、身份、开场白）从后端拿 —— 别在前端抄一份，改了会忘。 */
  fetch('/api/chat/people').then(function (r) {
    if (!r.ok) return null;
    return r.json();
  }).then(function (data) {
    if (data && data.people) {
      Object.keys(data.people).forEach(function (k) {
        people[k] = Object.assign({}, FALLBACK[k] || {}, data.people[k]);
      });
      paintHead();
    }
  }).catch(function () { /* 拿不到就用兜底的人设 */ });
})();
