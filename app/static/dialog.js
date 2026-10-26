/* ---------- 页内弹层的用法 ----------
 *
 *   askConfirm({title, text, node, yes, no, danger, focus})  → Promise<boolean>
 *   askNotice({title, text, node, ok})                       → Promise<void>
 *
 * 为什么自己写而不用 confirm()
 * --------------------------------
 * confirm() 是系统样式，跟产品对不上；而且只能给一行字，
 * 说不清「删的是哪一局、删了会怎样」。
 *
 * 两条约定：
 *   1. 正文用 **textContent** 写 —— 对局名是用户自己起的，
 *      可能带尖括号，当 HTML 塞进去就是注入。用 node 传富文本时，
 *      由调用方自己建节点。
 *   2. 危险动作（danger）默认把焦点放在「再想想」上：
 *      手一按回车不该把局删掉。
 */
(function () {
  const dlg = document.getElementById('dlg');
  if (!dlg) return;

  // 按 id 取，不按 class 取：class 是样式的事，id 才是这份模板的契约。
  // （按 class 取过一次，取消按钮没挂那个 class，脚本就在这儿断了。）
  const titleEl = document.getElementById('dlgTitle');
  const bodyEl = document.getElementById('dlgBody');
  const yesBtn = document.getElementById('dlgYes');
  const noBtn = document.getElementById('dlgNo');
  if (!titleEl || !bodyEl || !yesBtn || !noBtn) return;

  let settle = null;

  function close(result) {
    if (!settle) return;
    const done = settle;
    settle = null;
    dlg.hidden = true;
    done(result);
  }

  function fill(opts) {
    titleEl.textContent = opts.title || '确认？';
    bodyEl.textContent = '';
    if (opts.node) {
      bodyEl.appendChild(opts.node);
    } else if (opts.text) {
      bodyEl.textContent = opts.text;
    }
    yesBtn.textContent = opts.yes || '确认';
    noBtn.textContent = opts.no || '再想想';
    yesBtn.classList.toggle('dlg-danger', !!opts.danger);
    noBtn.hidden = !!opts.notice;
  }

  /**
   * 问一句，等一个答案。
   * @param {{title?:string, text?:string, node?:Node, yes?:string, no?:string,
   *          danger?:boolean, focus?:'yes'|'no', notice?:boolean}} opts
   * @returns {Promise<boolean>} 确认 = true，取消 = false
   */
  window.askConfirm = function (opts) {
    opts = opts || {};
    return new Promise((resolve) => {
      // 上一次还没答完又问了一次 —— 先把旧的按「取消」收掉
      if (settle) { const prev = settle; settle = null; dlg.hidden = true; prev(false); }
      fill(opts);
      dlg.hidden = false;
      const focusYes = opts.focus === 'yes' || (opts.danger === false && opts.focus !== 'no');
      (focusYes ? yesBtn : noBtn).focus();
      settle = resolve;
    });
  };

  /**
   * 只通知一件事，没有「确认 / 取消」。
   * @param {{title?:string, text?:string, node?:Node, ok?:string}} opts
   * @returns {Promise<void>}
   */
  window.askNotice = function (opts) {
    opts = opts || {};
    return new Promise((resolve) => {
      if (settle) { const prev = settle; settle = null; dlg.hidden = true; prev(false); }
      fill(Object.assign({}, opts, {notice: true, yes: opts.ok || '知道了'}));
      dlg.hidden = false;
      yesBtn.focus();
      settle = () => resolve();
    });
  };

  yesBtn.addEventListener('click', () => close(true));
  noBtn.addEventListener('click', () => close(false));
  // 点遮罩 = 取消（点卡片本身不算）
  dlg.addEventListener('click', (e) => { if (e.target === dlg) close(false); });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !dlg.hidden) close(false);
  });
})();
