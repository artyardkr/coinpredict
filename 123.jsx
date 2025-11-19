import React, { useState, useEffect, useRef } from 'react';
import { ArrowRight, ArrowLeft, TrendingUp, TrendingDown, Clock, AlertTriangle, Search, Database, MousePointer, Brain, XCircle, MessageSquare, BarChart2, X, Send, Sparkles } from 'lucide-react';

const Presentation = () => {
  const [currentSlide, setCurrentSlide] = useState(0);
  const [isAiPanelOpen, setIsAiPanelOpen] = useState(false);
  const totalSlides = 12;

  const nextSlide = () => {
    if (currentSlide < totalSlides - 1) setCurrentSlide(curr => curr + 1);
  };

  const prevSlide = () => {
    if (currentSlide > 0) setCurrentSlide(curr => curr - 1);
  };

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Prevent slide navigation if typing in AI panel
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      
      if (e.key === 'ArrowRight' || e.key === 'Space') nextSlide();
      if (e.key === 'ArrowLeft') prevSlide();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentSlide]);

  // Theme determination based on slide index (Act 1 vs Act 2)
  const isAct2 = currentSlide >= 10; // Slide 11 starts Act 2 logic conceptually

  return (
    <div className={`w-full h-screen flex flex-col transition-colors duration-700 ${isAct2 ? 'bg-slate-900 text-white' : 'bg-zinc-900 text-white'} overflow-hidden relative font-sans`}>
      
      {/* Progress Bar */}
      <div className="absolute top-0 left-0 h-1 bg-orange-500 transition-all duration-300 z-50" style={{ width: `${((currentSlide + 1) / totalSlides) * 100}%` }}></div>

      {/* AI Feature Button */}
      <button 
        onClick={() => setIsAiPanelOpen(true)}
        className="absolute bottom-6 right-6 z-[60] flex items-center gap-2 bg-gradient-to-r from-purple-600 to-blue-600 text-white px-4 py-3 rounded-full shadow-lg hover:scale-105 transition-transform font-bold animate-pulse"
      >
        <Sparkles size={20} />
        <span>AI 체험</span>
      </button>

      {/* AI Panel Component */}
      {isAiPanelOpen && (
        <AiInteractionPanel onClose={() => setIsAiPanelOpen(false)} isAct2={isAct2} />
      )}

      {/* Slide Content Container */}
      <div className="flex-1 flex items-center justify-center p-8 sm:p-16 relative">
        
        {/* SLIDE 1: Title */}
        {currentSlide === 0 && (
          <div className="text-center animate-fade-in-up space-y-8">
            <div className="inline-block bg-orange-500 text-black font-bold px-4 py-1 rounded-full mb-4 text-sm tracking-wide">
              TEAM PROJECT : BITCOIN ETF
            </div>
            <h1 className="text-5xl md:text-7xl font-black leading-tight tracking-tight">
              <span className="text-gray-400">From</span> <span className="text-orange-500">Ape</span><br />
              <span className="text-gray-400">To</span> <span className="text-blue-400">Intellectual</span>
            </h1>
            <div className="flex justify-center items-center gap-8 text-4xl my-8">
              <span>🦧</span>
              <div className="h-px w-24 bg-gray-600"></div>
              <span>👨‍🎓</span>
            </div>
            <div className="text-gray-500 font-mono mt-12">
              <p>Presenter: Song Seong-won</p>
              <p>2025.11.21</p>
            </div>
          </div>
        )}

        {/* SLIDE 2: 2021, 20 Years Old */}
        {currentSlide === 1 && (
          <div className="flex flex-col md:flex-row items-center gap-12 w-full max-w-5xl">
            <div className="w-full md:w-1/2 flex justify-center">
              {/* Mockup Phone UI */}
              <div className="w-64 h-96 border-4 border-gray-700 rounded-3xl p-4 flex flex-col justify-center items-center bg-gray-800 relative">
                <div className="absolute top-4 w-20 h-1 bg-gray-700 rounded-full"></div>
                <XCircle size={48} className="text-red-500 mb-4" />
                <h3 className="text-xl font-bold mb-2">인증 실패</h3>
                <p className="text-center text-gray-400 text-sm">
                  만 19세 미만은<br/>계좌를 개설할 수 없습니다.
                </p>
                <div className="mt-8 w-full h-10 bg-blue-600 rounded-lg flex items-center justify-center text-sm font-bold opacity-50">
                  확인
                </div>
              </div>
            </div>
            <div className="w-full md:w-1/2 space-y-6">
              <h2 className="text-6xl font-bold text-gray-200">2021년<br/><span className="text-blue-500">스무 살</span></h2>
              <p className="text-2xl text-gray-400 leading-relaxed">
                성인 인증 실패.<br/>
                그래서 <span className="text-white font-bold border-b-2 border-orange-500">누나 계좌</span>를 빌렸습니다.
              </p>
            </div>
          </div>
        )}

        {/* SLIDE 3: Chimpanzee Market */}
        {currentSlide === 2 && (
          <div className="w-full max-w-6xl relative">
            <h2 className="text-4xl md:text-5xl font-bold text-center mb-12">"침팬지도 돈 버는 시장"</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 opacity-90">
              <div className="bg-gray-800 p-6 rounded-lg transform -rotate-2 hover:scale-105 transition-transform border-l-4 border-red-500">
                <p className="text-gray-400 text-sm mb-2">NEWS</p>
                <h3 className="text-xl font-bold">"아무거나 사도 수익... 개미들의 축제"</h3>
              </div>
              <div className="bg-gray-800 p-6 rounded-lg transform rotate-1 hover:scale-105 transition-transform border-l-4 border-green-500">
                <p className="text-gray-400 text-sm mb-2">COMMUNITY</p>
                <h3 className="text-xl font-bold">"도지코인 5분만에 200% 떡상 ㅋㅋㅋ"</h3>
              </div>
              <div className="bg-gray-800 p-6 rounded-lg transform rotate-3 hover:scale-105 transition-transform border-l-4 border-yellow-500">
                <p className="text-gray-400 text-sm mb-2">FINANCE</p>
                <h3 className="text-xl font-bold">"가상화폐 광풍, 20대 투자자 급증"</h3>
              </div>
              <div className="bg-gray-800 p-6 rounded-lg transform -rotate-1 hover:scale-105 transition-transform border-l-4 border-blue-500">
                <p className="text-gray-400 text-sm mb-2">INTERVIEW</p>
                <h3 className="text-xl font-bold">"저 그냥 이름 예쁜 거 샀는데요?"</h3>
              </div>
            </div>
          </div>
        )}

        {/* SLIDE 4: 50% in 5 mins */}
        {currentSlide === 3 && (
          <div className="w-full h-full flex flex-col justify-center items-center relative">
             <div className="absolute inset-0 flex items-center justify-center opacity-10 pointer-events-none">
               <TrendingUp size={400} />
             </div>
             
             <div className="z-10 flex flex-col items-center space-y-8">
                <div className="flex items-center gap-4 text-red-500 animate-pulse">
                  <Clock size={48} />
                  <span className="text-5xl font-mono font-bold">05:00</span>
                </div>
                <h2 className="text-8xl font-black text-green-400 tracking-tighter">
                  +50%
                </h2>
                <p className="text-2xl text-gray-300">수업 시간보다 빠른 수익률</p>
             </div>
          </div>
        )}

        {/* SLIDE 5: Ape Pattern */}
        {currentSlide === 4 && (
          <div className="w-full max-w-4xl">
             <h2 className="text-3xl text-gray-400 mb-12 text-center">전형적인 유인원의 알고리즘</h2>
             <div className="flex flex-col md:flex-row items-center justify-between gap-4 text-center">
                <div className="flex flex-col items-center space-y-4">
                   <div className="w-24 h-24 rounded-full bg-green-600 flex items-center justify-center text-3xl font-bold">Buy</div>
                   <span className="text-xl">1. 산다</span>
                </div>
                <ArrowRight className="hidden md:block text-gray-600" size={32} />
                <div className="flex flex-col items-center space-y-4">
                   <div className="w-24 h-24 rounded-full bg-red-600 flex items-center justify-center text-3xl font-bold">Drop</div>
                   <span className="text-xl">2. 떨어진다</span>
                </div>
                <ArrowRight className="hidden md:block text-gray-600" size={32} />
                <div className="flex flex-col items-center space-y-4">
                   <div className="w-24 h-24 rounded-full bg-blue-600 flex items-center justify-center text-3xl font-bold">Stuck</div>
                   <span className="text-xl">3. 물린다</span>
                </div>
                <ArrowRight className="hidden md:block text-gray-600" size={32} />
                <div className="flex flex-col items-center space-y-4">
                   <div className="w-24 h-24 rounded-full bg-gray-600 flex items-center justify-center text-3xl font-bold">?</div>
                   <span className="text-xl">4. 검색한다</span>
                </div>
             </div>
          </div>
        )}

        {/* SLIDE 6: Crash */}
        {currentSlide === 5 && (
          <div className="w-full h-full flex flex-col justify-center items-center bg-red-900/20 absolute inset-0">
            <h2 className="text-4xl font-light mb-4 text-gray-300">2022년</h2>
            <div className="text-[12rem] font-black text-red-500 leading-none tracking-tighter">
              -60%
            </div>
            <TrendingDown size={64} className="text-red-500 mt-8" />
          </div>
        )}

        {/* SLIDE 7: Contempt */}
        {currentSlide === 6 && (
          <div className="text-center max-w-4xl">
             <h2 className="text-6xl md:text-7xl font-black mb-12 leading-tight">
               "이딴 게<br/>자산이라고?"
             </h2>
             <p className="text-2xl text-gray-400">저는 이 시장을 경멸했습니다.</p>
          </div>
        )}

        {/* SLIDE 8: Observation */}
        {currentSlide === 7 && (
          <div className="w-full max-w-4xl space-y-8">
             <div className="flex justify-between items-end h-64 border-b-2 border-gray-600 pb-2 px-4 relative">
                {/* Fake sideways chart */}
                <svg className="absolute inset-0 h-full w-full p-4" preserveAspectRatio="none">
                  <path d="M0,150 Q100,180 200,140 T400,160 T600,145 T800,155" fill="none" stroke="#666" strokeWidth="3" />
                </svg>
                <span className="text-gray-500 mb-2">2022</span>
                <div className="flex flex-col items-center mb-16">
                   <span className="text-4xl font-bold text-white">?</span>
                   <span className="text-sm text-gray-400">Why?</span>
                </div>
                <div className="flex flex-col items-center mb-32">
                   <span className="text-4xl font-bold text-white">?</span>
                   <span className="text-sm text-gray-400">When?</span>
                </div>
                <div className="flex flex-col items-center mb-20">
                   <span className="text-4xl font-bold text-white">?</span>
                   <span className="text-sm text-gray-400">How?</span>
                </div>
                <span className="text-gray-500 mb-2">2023</span>
             </div>
             <div className="text-center text-xl text-gray-400">
               질문만 쌓여가던 관찰의 시간
             </div>
          </div>
        )}

        {/* SLIDE 9: Turnaround (ETF Approved) */}
        {currentSlide === 8 && (
          <div className="flex flex-col items-center justify-center text-center">
             <div className="text-orange-500 font-bold tracking-widest mb-4">THE TURNING POINT</div>
             <h1 className="text-6xl md:text-8xl font-bold text-white mb-12">
               2024.01.10
             </h1>
             <div className="bg-white text-black px-8 py-4 rounded text-2xl font-bold mb-8 shadow-[0_0_30px_rgba(255,255,255,0.3)]">
               Bitcoin Spot ETF Approved
             </div>
             <div className="flex gap-8 text-gray-400 font-bold text-xl">
                <span>BlackRock</span>
                <span>Fidelity</span>
                <span>Ark Invest</span>
             </div>
          </div>
        )}

        {/* SLIDE 10: Difference */}
        {currentSlide === 9 && (
          <div className="w-full max-w-5xl">
            <h2 className="text-4xl font-bold text-center mb-12">Something Changed</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-0 border border-gray-700 rounded-xl overflow-hidden">
               <div className="bg-gray-800/50 p-8 border-b md:border-b-0 md:border-r border-gray-700">
                  <h3 className="text-gray-400 font-bold mb-6 uppercase tracking-wider">Before (The Ape Era)</h3>
                  <ul className="space-y-4 text-lg">
                    <li className="flex items-center gap-3 text-red-400"><TrendingDown size={20}/> 알트코인 동반 폭락</li>
                    <li className="flex items-center gap-3 text-gray-300"><AlertTriangle size={20}/> 뉴스 한 줄에 급등락</li>
                    <li className="flex items-center gap-3 text-yellow-500">🦧 침팬지도 수익</li>
                  </ul>
               </div>
               <div className="bg-blue-900/20 p-8">
                  <h3 className="text-blue-400 font-bold mb-6 uppercase tracking-wider">After (Institutional)</h3>
                  <ul className="space-y-4 text-lg">
                    <li className="flex items-center gap-3 text-green-400"><TrendingUp size={20}/> 비트코인 독주</li>
                    <li className="flex items-center gap-3 text-gray-300"><Database size={20}/> 기관 자금 유입 (ETF)</li>
                    <li className="flex items-center gap-3 text-blue-400"><Brain size={20}/> 전략과 분석 필요</li>
                  </ul>
               </div>
            </div>
          </div>
        )}

        {/* SLIDE 11: Three Questions */}
        {currentSlide === 10 && (
          <div className="w-full max-w-4xl">
             <h2 className="text-3xl md:text-4xl font-bold mb-12 text-center">증명을 위한 3가지 질문</h2>
             <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-slate-800 p-6 rounded-lg border border-slate-700 hover:border-blue-500 transition-colors">
                   <div className="text-blue-500 text-4xl font-bold mb-4">Q1</div>
                   <h3 className="text-xl font-bold mb-2">시장의 변화</h3>
                   <p className="text-gray-400 text-sm">정말 통계적으로 시장이 달라졌는가?</p>
                </div>
                <div className="bg-slate-800 p-6 rounded-lg border border-slate-700 hover:border-blue-500 transition-colors">
                   <div className="text-blue-500 text-4xl font-bold mb-4">Q2</div>
                   <h3 className="text-xl font-bold mb-2">가격 요인</h3>
                   <p className="text-gray-400 text-sm">가격을 움직이는 드라이버가 바뀌었는가?</p>
                </div>
                <div className="bg-slate-800 p-6 rounded-lg border border-slate-700 hover:border-blue-500 transition-colors">
                   <div className="text-blue-500 text-4xl font-bold mb-4">Q3</div>
                   <h3 className="text-xl font-bold mb-2">예측 가능성</h3>
                   <p className="text-gray-400 text-sm">그렇다면, 이제는 예측할 수 있는가?</p>
                </div>
             </div>
          </div>
        )}

        {/* SLIDE 12: Data Collection */}
        {currentSlide === 11 && (
          <div className="flex flex-col items-center w-full max-w-4xl">
             <div className="flex items-center gap-8 mb-16 w-full justify-center">
                <div className="text-center opacity-50">
                   <div className="text-4xl mb-2">🦧</div>
                   <div className="text-sm">Intuition</div>
                </div>
                <ArrowRight size={32} className="text-gray-600"/>
                <div className="text-center">
                   <div className="text-4xl mb-2">👨‍🎓</div>
                   <div className="text-sm font-bold text-blue-400">Data Driven</div>
                </div>
             </div>

             <h2 className="text-4xl font-bold mb-8">데이터로 말하기 위해</h2>
             
             <div className="grid grid-cols-3 gap-4 w-full text-center">
                <div className="bg-slate-800 p-6 rounded-lg">
                   <div className="text-gray-400 text-sm mb-1">Period</div>
                   <div className="text-3xl font-bold text-white">4.7<span className="text-lg text-blue-500 ml-1">years</span></div>
                   <div className="text-xs text-gray-500 mt-2">내가 물린 그 순간부터</div>
                </div>
                <div className="bg-slate-800 p-6 rounded-lg">
                   <div className="text-gray-400 text-sm mb-1">Samples</div>
                   <div className="text-3xl font-bold text-white">1,715<span className="text-lg text-blue-500 ml-1">days</span></div>
                   <div className="text-xs text-gray-500 mt-2">매일매일 추적</div>
                </div>
                 <div className="bg-slate-800 p-6 rounded-lg">
                   <div className="text-gray-400 text-sm mb-1">Features</div>
                   <div className="text-3xl font-bold text-white">138<span className="text-lg text-blue-500 ml-1">ea</span></div>
                   <div className="text-xs text-gray-500 mt-2">가격, 금리, 온체인...</div>
                </div>
             </div>
          </div>
        )}

      </div>

      {/* Footer Controls */}
      <div className="absolute bottom-0 left-0 w-full p-6 flex justify-between items-center text-gray-500 text-sm z-50">
        <div>Slide {currentSlide + 1} / {totalSlides}</div>
        <div className="flex gap-4">
          <button onClick={prevSlide} className="hover:text-white transition-colors p-2 rounded-full bg-white/5 backdrop-blur-sm"><ArrowLeft size={20} /></button>
          <button onClick={nextSlide} className="hover:text-white transition-colors p-2 rounded-full bg-white/5 backdrop-blur-sm"><ArrowRight size={20} /></button>
        </div>
      </div>
    </div>
  );
};

/* ================================================================
  GEMINI AI INTEGRATION COMPONENTS
  ================================================================
*/

const AiInteractionPanel = ({ onClose, isAct2 }) => {
  const [activeTab, setActiveTab] = useState('persona'); // 'persona' or 'sentiment'

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-md p-4">
      <div className="w-full max-w-2xl bg-slate-900 rounded-2xl border border-slate-700 shadow-2xl overflow-hidden flex flex-col h-[600px]">
        
        {/* Header */}
        <div className="p-4 bg-slate-800 border-b border-slate-700 flex justify-between items-center">
          <div className="flex items-center gap-2">
            <Sparkles className="text-yellow-400" size={20} />
            <h2 className="font-bold text-lg text-white">Gemini Powered Tools</h2>
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-white">
            <X size={24} />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-slate-700">
          <button 
            onClick={() => setActiveTab('persona')}
            className={`flex-1 py-3 font-bold text-sm flex items-center justify-center gap-2 transition-colors ${activeTab === 'persona' ? 'bg-slate-800 text-blue-400 border-b-2 border-blue-400' : 'text-gray-500 hover:bg-slate-800/50'}`}
          >
            <MessageSquare size={16} />
            투자 상담소 (Time Machine)
          </button>
          <button 
            onClick={() => setActiveTab('sentiment')}
            className={`flex-1 py-3 font-bold text-sm flex items-center justify-center gap-2 transition-colors ${activeTab === 'sentiment' ? 'bg-slate-800 text-green-400 border-b-2 border-green-400' : 'text-gray-500 hover:bg-slate-800/50'}`}
          >
            <BarChart2 size={16} />
            뉴스 감성 분석기 (Simulator)
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-hidden">
          {activeTab === 'persona' ? <PersonaChat /> : <SentimentAnalyzer />}
        </div>
        
      </div>
    </div>
  );
};

// Feature 1: Persona Chat (Ape vs Intellectual)
const PersonaChat = () => {
  const [persona, setPersona] = useState('ape'); // 'ape' or 'intellectual'
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([
    { role: 'system', content: '무엇이든 물어보세요. 제가 답변해 드립니다.' }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const chatContainerRef = useRef(null);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const apiKey = ""; // API key injected by environment
      const systemPrompt = persona === 'ape' 
        ? "당신은 2021년 코인 불장에 미쳐있는 20대 한국인 투자자입니다. 일명 '코인 유인원'입니다. 당신은 '가즈아', '화성 갈끄니까', '존버', '돔황챠', '구조대' 같은 인터넷 은어와 유행어를 과도하게 사용합니다. 펀더멘탈이나 기술적 분석은 무시하고 오직 감과 믿음으로 투자합니다. 무조건 긍정적이고 흥분한 상태로 답변하세요. 답변은 짧고 강렬하게 하세요."
        : "당신은 2025년의 냉철한 가상자산 데이터 분석가입니다. 당신은 '송성원 팀장'입니다. 당신은 감정을 배제하고 오직 데이터, 거시경제 지표(금리, 인플레이션), ETF 흐름, 온체인 데이터에 근거해서만 답변합니다. 논리적이고 차분하며 전문적인 용어(상관관계, 구조변화, ElasticNet 등)를 적절히 섞어서 사용하세요. 2021년의 무지성 투자를 경계합니다.";

      const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key=${apiKey}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{ parts: [{ text: input }] }],
          systemInstruction: { parts: [{ text: systemPrompt }] }
        })
      });

      const data = await response.json();
      const replyText = data.candidates?.[0]?.content?.parts?.[0]?.text || "오류가 발생했습니다. 다시 시도해주세요.";

      setMessages(prev => [...prev, { role: 'model', content: replyText }]);
    } catch (error) {
      console.error(error);
      setMessages(prev => [...prev, { role: 'model', content: "통신 오류가 발생했습니다." }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-slate-900">
      <div className="flex gap-2 p-4 justify-center bg-slate-800/50">
        <button 
          onClick={() => setPersona('ape')}
          className={`px-4 py-2 rounded-full text-sm font-bold transition-all ${persona === 'ape' ? 'bg-yellow-500 text-black scale-105 shadow-lg' : 'bg-slate-700 text-gray-400'}`}
        >
          🦧 2021년 유인원
        </button>
        <button 
          onClick={() => setPersona('intellectual')}
          className={`px-4 py-2 rounded-full text-sm font-bold transition-all ${persona === 'intellectual' ? 'bg-blue-600 text-white scale-105 shadow-lg' : 'bg-slate-700 text-gray-400'}`}
        >
          👨‍🎓 2025년 지식인
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4" ref={chatContainerRef}>
        {messages.slice(1).map((msg, idx) => (
          <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] p-3 rounded-lg text-sm leading-relaxed ${
              msg.role === 'user' 
                ? 'bg-slate-700 text-white rounded-br-none' 
                : persona === 'ape' 
                  ? 'bg-yellow-900/50 text-yellow-100 border border-yellow-700 rounded-bl-none'
                  : 'bg-blue-900/50 text-blue-100 border border-blue-700 rounded-bl-none'
            }`}>
              {msg.role === 'model' && (
                 <div className="text-xs font-bold mb-1 opacity-70">
                   {persona === 'ape' ? '🦧 유인원:' : '👨‍🎓 송성원 팀장:'}
                 </div>
              )}
              {msg.content}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start animate-pulse">
             <div className="bg-slate-800 p-3 rounded-lg text-sm text-gray-400">
                답변 생각 중...
             </div>
          </div>
        )}
      </div>

      <div className="p-4 bg-slate-800 border-t border-slate-700 flex gap-2">
        <input 
          type="text" 
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSend()}
          placeholder={persona === 'ape' ? "형, 지금 사도 돼? (질문하기)" : "팀장님, 현재 시장 상황은 어떤가요? (질문하기)"}
          className="flex-1 bg-slate-900 border border-slate-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
        />
        <button 
          onClick={handleSend}
          disabled={isLoading}
          className="bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 text-white p-2 rounded-lg transition-colors"
        >
          <Send size={20} />
        </button>
      </div>
    </div>
  );
};

// Feature 2: Sentiment Analyzer
const SentimentAnalyzer = () => {
  const [headline, setHeadline] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const analyzeSentiment = async () => {
    if (!headline.trim()) return;
    setLoading(true);
    setResult(null);

    try {
      const apiKey = ""; // API key injected by environment
      const prompt = `Analyze the sentiment of this crypto news headline for a Bitcoin investor. Return a JSON object with two fields: "score" (integer between -100 to 100, where -100 is extremely bearish, 0 is neutral, 100 is extremely bullish) and "reason" (a short explanation in Korean). Headline: "${headline}"`;

      const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key=${apiKey}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{ parts: [{ text: prompt }] }],
           generationConfig: { responseMimeType: "application/json" }
        })
      });

      const data = await response.json();
      const jsonText = data.candidates?.[0]?.content?.parts?.[0]?.text;
      setResult(JSON.parse(jsonText));
      
    } catch (error) {
      console.error(error);
      setResult({ score: 0, reason: "분석 중 오류가 발생했습니다." });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-slate-900 p-6 items-center justify-center">
      <div className="w-full max-w-md space-y-6">
        <div className="text-center space-y-2">
           <h3 className="text-2xl font-bold text-white">뉴스 감성 분석기</h3>
           <p className="text-gray-400 text-sm">발표에서 사용된 '감성 지표' 모델을 체험해보세요.<br/>뉴스가 가격에 미칠 영향을 AI가 예측합니다.</p>
        </div>

        <div className="space-y-2">
          <label className="text-sm font-bold text-blue-400 ml-1">News Headline</label>
          <textarea
            value={headline}
            onChange={(e) => setHeadline(e.target.value)}
            placeholder="예: 비트코인 현물 ETF 승인 임박, SEC 위원장 긍정적 발언..."
            className="w-full h-24 bg-slate-800 border border-slate-600 rounded-xl p-4 text-white focus:outline-none focus:border-green-500 resize-none"
          />
        </div>

        <button 
          onClick={analyzeSentiment}
          disabled={loading || !headline}
          className="w-full bg-green-600 hover:bg-green-700 disabled:bg-slate-700 disabled:text-gray-500 text-white font-bold py-3 rounded-xl transition-all flex items-center justify-center gap-2"
        >
          {loading ? 'AI 분석 중...' : '감성 점수 분석하기'}
          {!loading && <BarChart2 size={20} />}
        </button>

        {result && (
          <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 animate-fade-in-up">
             <div className="flex justify-between items-center mb-4">
                <span className="text-gray-400 text-sm font-bold uppercase">Sentiment Score</span>
                <span className={`text-3xl font-black ${result.score > 0 ? 'text-green-400' : result.score < 0 ? 'text-red-500' : 'text-gray-400'}`}>
                   {result.score > 0 ? '+' : ''}{result.score}
                </span>
             </div>
             <div className="h-2 w-full bg-slate-700 rounded-full mb-4 overflow-hidden">
                <div 
                  className={`h-full transition-all duration-1000 ${result.score > 0 ? 'bg-green-500' : 'bg-red-500'}`} 
                  style={{ width: `${Math.abs(result.score)}%`, marginLeft: result.score < 0 ? 0 : '0' }} // Simplified visualization
                ></div>
             </div>
             <p className="text-sm text-gray-300 leading-relaxed">
               <span className="text-blue-400 font-bold">AI 분석: </span>
               {result.reason}
             </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Presentation;