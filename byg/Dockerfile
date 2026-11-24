# ============================================================================
# CloudBase 云托管镜像（腾讯云开发）
#
# 云托管拿这份 Dockerfile 在**云端**构建镜像 —— 本地不需要装 Docker。
# 构建上下文是仓库根目录，所以下面只 COPY 真正需要的目录。
#
# 三个关键约定，改之前先想清楚：
#   1. PORT 由平台注入（默认 8080），必须监听 0.0.0.0，不能是 127.0.0.1
#      —— 监听回环地址的话，容器外访问不到，表现为「部署成功但访问 503」。
#   2. MATCH_DATA_DIR 指向 /data：容器本地磁盘是临时的，数据目录要落在
#      CFS 挂载点上（见 deploy/cloudbase/README.md 的「持久化」一节）。
#   3. --workers 1：视频解析结果只存在**进程内存**里，多 worker 会让
#      「传完视频立刻去标注页」找不到解析结果（解析在 A 进程，页面在 B 进程）。
# ============================================================================
FROM python:3.12-slim

# cv2 需要 OpenMP 运行时和 glib。两个包都很小，缺了会直接 ImportError，
# 而报错发生在延迟导入里 —— 只有真的去扫视频时才炸，很难第一时间想到是镜像缺包。
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080 \
    MATCH_DATA_DIR=/data

WORKDIR /app

# 先装依赖再拷代码：改代码不会让依赖层失效，重新构建快很多。
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 整个 app/ 都要：main.py、templates/、static/、seed/（含攻略原文与
# 生成出来的模式库）、collection/、rules/、report/、adapters/。
COPY app ./app

# 数据目录：SQLite、视频、抽帧、临时文件。挂 CFS 时挂到这里。
RUN mkdir -p /data

EXPOSE 8080

# 平台有自己的健康检查配置（服务详情 → 健康检查路径填 "/"），
# 这里这份只是兜底，防止容器起不来却一直显示「运行中」。
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os,sys,urllib.request; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8080')+'/', timeout=4).status==200 else 1)"

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --timeout-keep-alive 65"]
