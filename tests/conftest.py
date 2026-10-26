"""pytest 的全局设置。

临时目录必须落在 **D 盘**
------------------------
pytest 默认把每个测试的临时文件（临时数据库、导出物）写在系统临时目录里 ——
而 Windows 的系统临时目录在 **C 盘**。

跑一轮全量回归会在那里堆出 1~2 GB（一次实测：``<本机目录><用户>\\AppData\\
Local\\Temp\\pytest-of-<用户>`` 占了 1.8 GB）。C 盘写满时，SQLite 直接报
``database or disk is full``，测试成片地失败 —— 而失败原因跟被测代码毫无
关系，很容易被误判成「代码有问题」。

所以这里把临时目录指到项目自己的 ``.tmp/pytest`` 下。

必须在 pytest 决定临时目录**之前**改掉环境变量 —— ``conftest.py`` 在
收集阶段就被导入，早于任何 fixture 调用，时机正好。
"""

from __future__ import annotations

import os

#: 临时目录放哪（项目自己的 ``.tmp`` 下，跟其它临时产物在一处）。
_TMP_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".tmp", "pytest"
)

os.makedirs(_TMP_ROOT, exist_ok=True)

#: 三个都改：``tempfile`` 在 Windows 上认 TEMP/TMP，POSIX 上认 TMPDIR。
for _var in ("TEMP", "TMP", "TMPDIR"):
    os.environ[_var] = _TMP_ROOT
