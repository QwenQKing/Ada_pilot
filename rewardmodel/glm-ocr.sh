#!/bin/bash
# ========== 配置 ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_SCRIPT="$SCRIPT_DIR/glm_ocr_server.py"
CUDA_DEVICE="${CUDA_DEVICE:-5}"
PORT="${PORT:-4512}"
LOG_FILE="$SCRIPT_DIR/glm_ocr_${PORT}.log"
PID_FILE="$SCRIPT_DIR/glm_ocr_${PORT}.pid"

# ========== 检查是否已在运行 ==========
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[WARN] 服务已在运行 (PID=$OLD_PID)，请先停止后再启动。"
        echo "       停止命令: kill $OLD_PID"
        exit 1
    else
        echo "[INFO] 检测到残留 PID 文件，已清理。"
        rm -f "$PID_FILE"
    fi
fi

# ========== 检查脚本是否存在 ==========
if [ ! -f "$SERVER_SCRIPT" ]; then
    echo "[ERROR] 找不到服务脚本: $SERVER_SCRIPT"
    exit 1
fi

# ========== 启动服务 ==========
echo "[INFO] 启动 GLM-OCR 服务，GPU=$CUDA_DEVICE PORT=$PORT ..."
echo "[INFO] 日志输出到: $LOG_FILE"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE nohup python "$SERVER_SCRIPT" --port $PORT > "$LOG_FILE" 2>&1 &

PID=$!
echo $PID > "$PID_FILE"

# ========== 等待确认进程存活 ==========
sleep 2
if kill -0 "$PID" 2>/dev/null; then
    echo "[OK] 服务已启动 (PID=$PID)"
    echo "[INFO] 查看日志: tail -f $LOG_FILE"
    echo "[INFO] 停止服务: kill \$(cat $PID_FILE)"
else
    echo "[ERROR] 服务启动失败，请查看日志: $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi
