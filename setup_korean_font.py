#!/usr/bin/env python3
"""
matplotlib 한글 폰트 설정

Mac의 경우: AppleGothic 또는 Apple SD Gothic Neo 사용
Windows의 경우: Malgun Gothic 사용
Linux의 경우: Nanum Gothic 설치 필요
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

def setup_korean_font():
    """
    운영체제에 맞는 한글 폰트 설정
    """
    system = platform.system()

    if system == 'Darwin':  # Mac
        # Mac에서 사용 가능한 한글 폰트 찾기
        available_fonts = [f.name for f in fm.fontManager.ttflist]

        # 우선순위: AppleGothic > Apple SD Gothic Neo > AppleMyungjo
        korean_fonts = ['AppleGothic', 'Apple SD Gothic Neo', 'AppleMyungjo']

        selected_font = None
        for font in korean_fonts:
            if font in available_fonts:
                selected_font = font
                break

        if selected_font:
            plt.rcParams['font.family'] = selected_font
            print(f"✅ Mac 한글 폰트 설정 완료: {selected_font}")
        else:
            print("⚠️ Mac 한글 폰트를 찾을 수 없습니다.")
            print(f"사용 가능한 폰트: {korean_fonts}")

    elif system == 'Windows':  # Windows
        plt.rcParams['font.family'] = 'Malgun Gothic'
        print("✅ Windows 한글 폰트 설정 완료: Malgun Gothic")

    else:  # Linux
        plt.rcParams['font.family'] = 'NanumGothic'
        print("✅ Linux 한글 폰트 설정 완료: NanumGothic")

    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False

    return plt.rcParams['font.family']

if __name__ == "__main__":
    print("=" * 60)
    print("matplotlib 한글 폰트 설정 테스트")
    print("=" * 60)

    # 폰트 설정
    font_name = setup_korean_font()

    # 테스트
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([1, 2, 3], [1, 4, 2], marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('시간 (일)', fontsize=12, fontweight='bold')
    ax.set_ylabel('가격 ($)', fontsize=12, fontweight='bold')
    ax.set_title('한글 폰트 테스트: 비트코인 가격', fontsize=14, fontweight='bold')
    ax.text(2, 3, '한글이 잘 보이나요? ✅', fontsize=11, ha='center')
    ax.grid(True, alpha=0.3)

    plt.savefig('korean_font_test.png', dpi=150, bbox_inches='tight')
    print(f"\n테스트 이미지 저장: korean_font_test.png")
    print(f"사용된 폰트: {font_name}")
    print("\n이미지를 열어서 한글이 제대로 보이는지 확인하세요!")
    plt.close()
