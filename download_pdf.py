import requests
import os

print("=" * 70)
print("PDF 다운로드 스크립트")
print("=" * 70)

# PDF URL (제공하신 상대 경로로부터 전체 URL 구성)
url = "https://www.sciencedirect.com/science/article/pii/S266682702500057X/pdfft?md5=cc19b23f81721dd431457de1253c99f3&pid=1-s2.0-S266682702500057X-main.pdf"

# 파일명
filename = "논문3_S266682702500057X.pdf"

print(f"\n다운로드 URL:")
print(f"  {url}")
print(f"\n저장 파일명: {filename}")

try:
    print("\n다운로드 중...")

    # Headers 설정 (일반 브라우저처럼 보이도록)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }

    # GET 요청
    response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)

    # 상태 코드 확인
    print(f"  응답 상태: {response.status_code}")

    if response.status_code == 200:
        # Content-Type 확인
        content_type = response.headers.get('Content-Type', '')
        print(f"  Content-Type: {content_type}")

        # PDF인지 확인
        if 'pdf' in content_type.lower() or response.content[:4] == b'%PDF':
            # 파일 저장
            with open(filename, 'wb') as f:
                f.write(response.content)

            file_size = os.path.getsize(filename)
            print(f"\n✓ 다운로드 성공!")
            print(f"  파일 크기: {file_size:,} bytes ({file_size/1024:.1f} KB)")
            print(f"  저장 위치: {os.path.abspath(filename)}")
        else:
            print(f"\n✗ PDF가 아닌 것 같습니다.")
            print(f"  받은 내용 (첫 500자):")
            print(f"  {response.text[:500]}")

            # HTML로 저장해서 확인
            with open('response.html', 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"\n  → 응답 내용을 'response.html'에 저장했습니다.")
            print(f"     (로그인 페이지이거나 접근 제한 메시지일 수 있습니다)")

    elif response.status_code == 401:
        print("\n✗ 인증 필요 (401 Unauthorized)")
        print("  → 로그인이 필요한 유료 논문입니다.")
        print("  → 학교/기관 네트워크에 연결하거나 구독 필요")

    elif response.status_code == 403:
        print("\n✗ 접근 거부 (403 Forbidden)")
        print("  → 접근 권한이 없습니다.")
        print("  → 학교/기관 네트워크에서 시도하거나 VPN 사용")

    elif response.status_code == 404:
        print("\n✗ 파일 없음 (404 Not Found)")
        print("  → URL이 잘못되었거나 파일이 삭제되었습니다.")

    else:
        print(f"\n✗ 다운로드 실패")
        print(f"  응답 내용: {response.text[:200]}")

except requests.exceptions.RequestException as e:
    print(f"\n✗ 오류 발생: {e}")

print("\n" + "=" * 70)
print("작업 완료")
print("=" * 70)

print("\n💡 다운로드가 실패한 경우:")
print("  1. 학교/기관 네트워크에 연결되어 있는지 확인")
print("  2. 브라우저에서 로그인 후 쿠키를 사용하는 방법 필요")
print("  3. 또는 브라우저에서 직접 다운로드 (우클릭 → 다른 이름으로 링크 저장)")
