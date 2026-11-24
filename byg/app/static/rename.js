// 手动改名：一局对局可以自己起个名字。
//
// 交互只有三步：点「改名」→ 变成输入框 → 回车保存。
// 不弹对话框、不加说明文字 —— 名字是用户自己的事，页面只管收下。
//
// 两处用同一套：报告页的标题、对局记录页每一行的名字。
// 所以标记是通用的：带 data-rename 的按钮点一下，
// 它所在的 [data-match-scope] 里的 [data-rename-label] 就变成输入框。

(function () {
  var API = '/api/matches/';

  function labelOf(btn) {
    var scope = btn.closest('[data-match-scope]');
    return scope ? scope.querySelector('[data-rename-label]') : null;
  }

  function put(matchId, title) {
    return fetch(API + encodeURIComponent(matchId) + '/title', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({title: title})
    }).then(function (res) {
      if (!res.ok) throw new Error('HTTP ' + res.status);
      return res.json();
    });
  }

  function start(btn) {
    var label = labelOf(btn);
    if (!label || btn.dataset.busy) return;

    var oldText = label.textContent.trim();
    var oldBtnText = btn.textContent;
    var holder = document.createElement('span');
    holder.className = 'rename-holder';

    var input = document.createElement('input');
    input.className = 'rename-input';
    input.type = 'text';
    input.maxLength = 60;
    input.value = oldText;
    holder.appendChild(input);
    label.replaceWith(holder);

    input.focus();
    input.select();

    var settled = false;
    function finish(save) {
      if (settled) return;
      settled = true;

      var title = input.value.trim();
      holder.replaceWith(label);
      label.textContent = title || oldText;

      // 没改就别打扰服务器
      if (!save || title === oldText) return;

      btn.dataset.busy = '1';
      put(btn.dataset.matchId, title)
        .then(function (data) {
          var name = data.title || title;
          label.textContent = name;
          // 列表页还要更新搜索用的名字和导出文件名
          document.dispatchEvent(new CustomEvent('ma:renamed', {
            detail: {id: btn.dataset.matchId, title: name}
          }));
        })
        .catch(function (err) {
          label.textContent = oldText;
          btn.textContent = '没存上';
          setTimeout(function () { btn.textContent = oldBtnText; }, 2000);
          if (window.console) console.error(err);
        })
        .finally(function () { btn.dataset.busy = ''; });
    }

    input.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') { e.preventDefault(); finish(true); }
      else if (e.key === 'Escape') { e.preventDefault(); finish(false); }
    });
    input.addEventListener('blur', function () { finish(true); });
  }

  document.addEventListener('click', function (e) {
    var btn = e.target.closest ? e.target.closest('[data-rename]') : null;
    if (!btn) return;
    e.preventDefault();
    start(btn);
  });
})();
