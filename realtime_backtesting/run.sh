#!/bin/bash

# 실시간 백테스팅 대시보드 실행 스크립트

echo "=================================="
echo "실시간 비트코인 백테스팅 시뮬레이터"
echo "=================================="
echo ""

# 패키지 확인
echo "1. 패키지 확인 중..."
python3 -c "import streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Streamlit이 설치되지 않았습니다."
    echo "설치하시겠습니까? (y/n)"
    read answer
    if [ "$answer" == "y" ]; then
        pip install -r requirements.txt
    else
        echo "설치를 건너뛰었습니다."
        exit 1
    fi
fi

echo "✅ 패키지 확인 완료"
echo ""

# 대시보드 실행
echo "2. 대시보드 실행 중..."
echo "브라우저가 자동으로 열립니다."
echo "종료하려면 Ctrl+C를 누르세요."
echo ""

streamlit run dashboard.py

echo ""
echo "대시보드를 종료했습니다."
