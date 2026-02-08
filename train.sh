#!/bin/bash

# 설정 변수
PYTHON_SCRIPT="train_vits.py"
CONFIG_FILE="configs/ko.json"
MODEL_NAME="ko"
LOG_FILE="train_vits.log"

# 1. 기존 로그 파일 정리
if [ -f "$LOG_FILE" ]; then
    rm "$LOG_FILE"
    echo "🗑️  기존 로그($LOG_FILE) 삭제 완료."
fi

# 2. 학습 시작 (nohup 사용, 표준출력/에러 모두 로그파일로 리다이렉트)
echo "🚀 VITS 8k 학습을 시작합니다..."
# nohup으로 실행하고, 출력(1)과 에러(2)를 모두 LOG_FILE에 저장 (&)
nohup python $PYTHON_SCRIPT -c $CONFIG_FILE -m $MODEL_NAME > $LOG_FILE 2>&1 &

# 3. PID 안내
echo "✅ 실행 완료! (PID: $!)"
echo "📜 로그 확인: tail -f $LOG_FILE"
