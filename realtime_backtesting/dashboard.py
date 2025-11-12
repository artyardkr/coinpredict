"""
실시간 백테스팅 대시보드 (Streamlit) - 개선된 UI/UX
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

# 로컬 모듈 import
from data_fetcher import RealTimeDataFetcher
from prediction_engine import PredictionEngine
from backtesting_engine import BacktestingEngine, PositionType
import config


# 페이지 설정
st.set_page_config(
    page_title="실시간 비트코인 백테스팅",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS - 깔끔하고 현대적인 디자인
st.markdown("""
<style>
    /* 전체 폰트 */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* 메인 컨테이너 */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }

    /* 카드 스타일 */
    .stApp {
        background: transparent;
    }

    /* 헤더 스타일 */
    h1 {
        color: white;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        margin-bottom: 0.5rem;
    }

    h3 {
        color: white;
        font-weight: 600;
        margin-top: 2rem;
    }

    /* 메트릭 카드 */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.875rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    [data-testid="stMetricDelta"] {
        font-size: 1rem;
    }

    /* 사이드바 */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
    }

    /* 버튼 스타일 */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }

    /* 차트 컨테이너 */
    .js-plotly-plot {
        border-radius: 15px;
        background: white;
        padding: 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }

    /* 데이터프레임 */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* 정보 박스 */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    /* 입력 필드 */
    .stSelectbox, .stSlider, .stNumberInput {
        background: white;
        border-radius: 8px;
    }

    /* 프로그레스 바 부드럽게 */
    .stProgress > div > div {
        transition: width 0.5s ease;
    }

    /* 메트릭 컨테이너 */
    [data-testid="metric-container"] {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    [data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }

    /* 구분선 */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 헤더
st.markdown("<h1>₿ 실시간 비트코인 트레이딩 백테스트</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: rgba(255,255,255,0.8); font-size: 1.1rem; margin-bottom: 2rem;'>AI 기반 트레이딩 전략 시뮬레이터</p>", unsafe_allow_html=True)

# 사이드바 - 설정
st.sidebar.header("⚙️ 전략 설정")

# 전략 선택
strategy = st.sidebar.selectbox(
    "트레이딩 전략",
    ["threshold", "long_only", "long_short"],
    index=0,
    help="threshold: 확신 높을 때만 매수 (추천)\nlong_only: 상승 예측시만 매수\nlong_short: 롱/숏 모두 활용"
)

# Threshold 설정
if strategy == "threshold":
    threshold = st.sidebar.slider(
        "매수 기준 수익률 (%)",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="예측 수익률이 이 값보다 클 때만 매수"
    )
else:
    threshold = 0.0

st.sidebar.markdown("---")
st.sidebar.header("💰 자본 설정")

# 초기 자본
initial_capital = st.sidebar.number_input(
    "초기 자본 ($)",
    min_value=1000,
    max_value=1000000,
    value=10000,
    step=1000
)

# 거래 비용
transaction_cost = st.sidebar.slider(
    "거래 수수료 (%)",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.01
) / 100

st.sidebar.markdown("---")
st.sidebar.header("📊 데이터 설정")

# 데이터 기간
lookback_days = st.sidebar.select_slider(
    "분석 기간",
    options=[30, 90, 180, 365, 730],
    value=365,
    format_func=lambda x: f"{x}일 ({x//30}개월)"
)

# 업데이트 빈도 (부드러운 애니메이션)
update_frequency = st.sidebar.select_slider(
    "업데이트 속도",
    options=[1, 5, 10, 20, 50],
    value=10,
    format_func=lambda x: f"{x}일씩"
)

# 업데이트 간격
update_interval = st.sidebar.slider(
    "업데이트 간격 (초)",
    min_value=0.1,
    max_value=2.0,
    value=0.5,
    step=0.1
)

st.sidebar.markdown("---")

# 컨트롤 버튼
col1, col2 = st.sidebar.columns(2)
start_btn = col1.button("▶️ 시작", use_container_width=True, type="primary")
stop_btn = col2.button("⏸️ 정지", use_container_width=True)
reset_btn = st.sidebar.button("🔄 초기화", use_container_width=True)

st.sidebar.markdown("---")
with st.sidebar.expander("💡 사용 가이드", expanded=False):
    st.markdown("""
    **빠른 시작:**
    1. 전략을 선택하세요
    2. 파라미터를 조정하세요
    3. ▶️ 시작 버튼을 누르세요

    **추천 설정:**
    - 전략: Threshold
    - 매수 기준: 1.0%
    - 초기 자본: $10,000
    """)

# 세션 상태 초기화
if 'initialized' not in st.session_state or reset_btn:
    st.session_state.initialized = True
    st.session_state.running = False
    st.session_state.current_idx = 0
    st.session_state.data = None
    st.session_state.engine = None
    st.session_state.predictions = []
    st.session_state.portfolio_history = []

# 데이터 로드
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(lookback_days):
    """데이터 로드 (캐싱)"""
    fetcher = RealTimeDataFetcher(lookback_days=lookback_days)
    df = fetcher.fetch_historical_data()
    return df

# 메인 화면
if start_btn:
    st.session_state.running = True

if stop_btn:
    st.session_state.running = False

# 데이터 로드
if st.session_state.data is None:
    with st.spinner("🔄 데이터를 불러오는 중..."):
        st.session_state.data = load_data(lookback_days)

        if st.session_state.data is not None and len(st.session_state.data) > 0:
            st.success(f"✅ 데이터 로드 완료: {len(st.session_state.data)}일")
            time.sleep(1)
            st.rerun()
        else:
            st.error("❌ 데이터 로드 실패")
            st.stop()

# 엔진 초기화
if st.session_state.engine is None or reset_btn:
    st.session_state.engine = BacktestingEngine(
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
        strategy=strategy,
        threshold=threshold
    )
    st.session_state.current_idx = 30
    st.session_state.predictions = []
    st.session_state.portfolio_history = []

# 예측 엔진 초기화
prediction_engine = PredictionEngine()

# 대시보드 레이아웃
if st.session_state.data is not None:

    # 상단: 현재 상태 메트릭
    if st.session_state.portfolio_history:
        latest = st.session_state.portfolio_history[-1]
        latest_pred = st.session_state.predictions[-1]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            delta_color = "normal" if latest_pred['predicted_return'] >= 0 else "inverse"
            st.metric(
                "현재 가격",
                f"${latest_pred['actual_price']:,.0f}",
                f"{latest_pred['predicted_return']:+.2f}% 예측"
            )

        with col2:
            st.metric(
                "포트폴리오",
                f"${latest['portfolio_value']:,.0f}",
                f"{latest['total_return']:+.2f}%"
            )

        with col3:
            st.metric(
                "포지션",
                latest['position'],
                f"{latest['num_trades']}회 거래"
            )

        with col4:
            metrics = st.session_state.engine.get_performance_metrics()
            if metrics:
                st.metric(
                    "샤프 비율",
                    f"{metrics['sharpe_ratio']:.2f}",
                    f"MDD {metrics['max_drawdown']:.1f}%"
                )

    st.markdown("---")

    # 중앙: 차트 영역
    tab1, tab2, tab3 = st.tabs(["📈 가격 차트", "💰 포트폴리오", "📊 거래 내역"])

    with tab1:
        price_chart_placeholder = st.empty()

    with tab2:
        portfolio_chart_placeholder = st.empty()

    with tab3:
        trades_placeholder = st.empty()

# 시뮬레이션 루프
if st.session_state.running and st.session_state.data is not None:
    data = st.session_state.data
    engine = st.session_state.engine
    idx = st.session_state.current_idx

    if idx < len(data):
        # 한 번에 여러 일을 처리 (부드러운 업데이트)
        batch_size = min(update_frequency, len(data) - idx)

        for i in range(batch_size):
            if idx + i >= len(data):
                break

            # 현재 시점 데이터
            current_data = data.iloc[:idx + i + 1]
            current_row = data.iloc[idx + i]

            current_price = current_row['Close']
            timestamp = current_row['Date']

            # 예측
            prediction = prediction_engine.predict(current_data)

            if prediction:
                predicted_return = prediction['predicted_return']
                predicted_price = prediction['predicted_price']

                # 백테스팅 업데이트
                status = engine.update(current_price, predicted_return, timestamp)

                # 결과 저장
                st.session_state.predictions.append({
                    'date': timestamp,
                    'actual_price': current_price,
                    'predicted_return': predicted_return,
                    'predicted_price': predicted_price
                })
                st.session_state.portfolio_history.append(status)

            # 차트 업데이트
            pred_df = pd.DataFrame(st.session_state.predictions)
            port_df = pd.DataFrame(st.session_state.portfolio_history)

            # 1. 가격 차트 (탭 1)
            fig_price = make_subplots(
                rows=2, cols=1,
                row_heights=[0.65, 0.35],
                subplot_titles=("", ""),
                vertical_spacing=0.08,
                specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
            )

            # 실제 가격
            fig_price.add_trace(
                go.Scatter(
                    x=pred_df['date'],
                    y=pred_df['actual_price'],
                    name="실제 가격",
                    line=dict(color='#667eea', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(102, 126, 234, 0.1)'
                ),
                row=1, col=1
            )

            # 예측 가격
            fig_price.add_trace(
                go.Scatter(
                    x=pred_df['date'],
                    y=pred_df['predicted_price'],
                    name="AI 예측",
                    line=dict(color='#fbbf24', width=2, dash='dot'),
                    opacity=0.8
                ),
                row=1, col=1
            )

            # 예측 수익률
            colors = ['#10b981' if x > 0 else '#ef4444' for x in pred_df['predicted_return']]
            fig_price.add_trace(
                go.Bar(
                    x=pred_df['date'],
                    y=pred_df['predicted_return'],
                    name="예측 수익률",
                    marker_color=colors,
                    opacity=0.7,
                    showlegend=False
                ),
                row=2, col=1
            )

            # 레이아웃
            fig_price.update_xaxes(showgrid=False, row=1, col=1)
            fig_price.update_xaxes(title_text="날짜", showgrid=False, row=2, col=1)
            fig_price.update_yaxes(title_text="가격 (USD)", showgrid=True, gridcolor='rgba(0,0,0,0.05)', row=1, col=1)
            fig_price.update_yaxes(title_text="수익률 (%)", showgrid=True, gridcolor='rgba(0,0,0,0.05)', row=2, col=1)

            fig_price.update_layout(
                height=600,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=60, r=30, t=30, b=60),
                paper_bgcolor='white',
                plot_bgcolor='white',
                hovermode='x unified',
                transition=dict(duration=300, easing='cubic-in-out')
            )

            price_chart_placeholder.plotly_chart(
                fig_price,
                use_container_width=True,
                key="price_chart",
                config={'displayModeBar': False}
            )

            # 2. 포트폴리오 차트 (탭 2)
            fig_portfolio = go.Figure()

            # 포트폴리오 가치
            fig_portfolio.add_trace(
                go.Scatter(
                    x=port_df['timestamp'],
                    y=port_df['portfolio_value'],
                    name="전략 수익",
                    line=dict(color='#10b981', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(16, 185, 129, 0.1)'
                )
            )

            # Buy-and-Hold 비교
            bnh_values = initial_capital * (current_data['Close'] / current_data['Close'].iloc[0])
            fig_portfolio.add_trace(
                go.Scatter(
                    x=current_data['Date'],
                    y=bnh_values,
                    name="Buy & Hold",
                    line=dict(color='#94a3b8', width=2, dash='dash'),
                    opacity=0.6
                )
            )

            fig_portfolio.update_layout(
                height=500,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                xaxis_title="날짜",
                yaxis_title="포트폴리오 가치 (USD)",
                margin=dict(l=60, r=30, t=30, b=60),
                paper_bgcolor='white',
                plot_bgcolor='white',
                hovermode='x unified',
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
                transition=dict(duration=300, easing='cubic-in-out')
            )

            portfolio_chart_placeholder.plotly_chart(
                fig_portfolio,
                use_container_width=True,
                key="portfolio_chart",
                config={'displayModeBar': False}
            )

            # 3. 거래 내역 (탭 3)
            if engine.trades:
                trades_df = pd.DataFrame(engine.trades[-20:])  # 최근 20개
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
                trades_df['price'] = trades_df['price'].apply(lambda x: f"${x:,.2f}")
                trades_df['pnl'] = trades_df['pnl'].apply(lambda x: f"${x:+,.2f}")

                # 스타일링된 데이터프레임
                trades_placeholder.dataframe(
                    trades_df[['timestamp', 'type', 'price', 'pnl', 'reason']].rename(columns={
                        'timestamp': '시간',
                        'type': '주문 유형',
                        'price': '가격',
                        'pnl': '손익',
                        'reason': '사유'
                    }),
                    use_container_width=True,
                    height=400
                )

            # 프로그레스 바
            progress = (idx + batch_size) / len(data)
            st.progress(progress, text=f"진행률: {progress*100:.1f}% ({idx + batch_size}/{len(data)}일)")

            # 인덱스 증가 (배치 크기만큼)
            st.session_state.current_idx += batch_size

            # 부드러운 자동 진행
            time.sleep(update_interval)
            st.rerun()

    else:
        st.session_state.running = False

        # 최종 결과
        final_metrics = engine.get_performance_metrics()
        st.balloons()

        st.markdown("<h2 style='text-align: center; color: white;'>🎉 시뮬레이션 완료!</h2>", unsafe_allow_html=True)

        # 최종 성과 지표
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("최종 자산", f"${final_metrics['final_value']:,.0f}")

        with col2:
            st.metric("총 수익률", f"{final_metrics['total_return']:+.2f}%")

        with col3:
            st.metric("샤프 비율", f"{final_metrics['sharpe_ratio']:.2f}")

        with col4:
            st.metric("최대 낙폭", f"{final_metrics['max_drawdown']:.2f}%")

        with col5:
            st.metric("승률", f"{final_metrics['win_rate']:.1f}%")

else:
    # 대기 상태
    if not st.session_state.running:
        st.info("⏸️ 시뮬레이션이 준비되었습니다. 왼쪽 사이드바에서 전략을 설정하고 '▶️ 시작' 버튼을 눌러주세요.", icon="ℹ️")

        # 안내 이미지 (선택사항)
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 3rem; background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);'>
                <h3 style='color: #667eea;'>실시간 백테스팅을 시작하세요</h3>
                <p style='color: #6b7280; margin-top: 1rem;'>
                    AI 기반 트레이딩 전략의 성과를 실시간으로 시뮬레이션하고<br>
                    다양한 지표로 성과를 분석할 수 있습니다.
                </p>
            </div>
            """, unsafe_allow_html=True)
