#!/bin/bash
# 启动前端 + 后端
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 安装后端依赖（首次运行，或缺少 WebSocket runtime）
if ! python -c "import fastapi, uvicorn, websockets" 2>/dev/null; then
  echo "安装后端依赖..."
  pip install -r "$SCRIPT_DIR/backend/requirements.txt"
fi

echo "启动 FastAPI 后端 (port 8000)..."
cd "$SCRIPT_DIR/backend"
python server.py &
BACKEND_PID=$!

echo "启动 React 前端 (port 5173)..."
cd "$SCRIPT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ 前端: http://localhost:5173"
echo "✅ 后端: http://localhost:8000"
echo ""
echo "按 Ctrl+C 停止所有服务"

cleanup() {
  kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
  exit 0
}
trap cleanup INT TERM

wait
