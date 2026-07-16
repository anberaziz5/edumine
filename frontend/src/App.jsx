import { useState, useEffect } from 'react'

const API = import.meta.env.VITE_API_URL

const RISK_COLOR = {
  'High Risk':     { ring:'stroke-red-500',   bg:'bg-red-500/10',   text:'text-red-400',   gradient:'from-red-500 to-rose-600' },
  'Moderate Risk': { ring:'stroke-amber-400', bg:'bg-amber-400/10', text:'text-amber-400', gradient:'from-amber-400 to-orange-500' },
  'Low Risk':      { ring:'stroke-emerald-400',bg:'bg-emerald-400/10',text:'text-emerald-400',gradient:'from-emerald-400 to-teal-500' },
}

function RiskGauge({ pct, label }) {
  const colors = RISK_COLOR[label] || RISK_COLOR['Low Risk']
  const radius = 60;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (pct / 100) * circumference;
  
  const [animatedOffset, setAnimatedOffset] = useState(circumference);
  
  useEffect(() => {
    setAnimatedOffset(circumference); // reset
    const timer = setTimeout(() => setAnimatedOffset(strokeDashoffset), 50);
    return () => clearTimeout(timer);
  }, [pct, strokeDashoffset, circumference]);

  return (
    <div className='flex flex-col items-center justify-center p-6 w-full'>
      <div className='relative w-48 h-48 flex items-center justify-center drop-shadow-2xl'>
        <svg className='absolute inset-0 w-full h-full -rotate-90' viewBox='0 0 140 140'>
          <circle cx='70' cy='70' r={radius} className='stroke-slate-800/50' strokeWidth='12' fill='none' />
          <circle 
            cx='70' cy='70' r={radius} 
            className={`${colors.ring} transition-all duration-1000 ease-out`} 
            strokeWidth='12' 
            fill='none' 
            strokeDasharray={circumference} 
            strokeDashoffset={animatedOffset} 
            strokeLinecap='round'
          />
        </svg>
        <div className='absolute inset-0 flex flex-col items-center justify-center pointer-events-none'>
            <span className={`text-5xl font-black ${colors.text} tracking-tighter drop-shadow-md`}>{pct}<span className='text-2xl text-slate-500'>%</span></span>
        </div>
      </div>
      <div className={`mt-6 px-6 py-2 rounded-full ${colors.bg} ${colors.text} border border-current/20 text-sm font-bold uppercase tracking-widest shadow-lg`}>
        {label}
      </div>
    </div>
  )
}

function ShapBar({ feature, shap, delay }) {
  const positive = shap > 0
  const pct = Math.min(100, Math.abs(shap) * 200)
  const label = feature.replace(/_/g,' ').replace(/\b\w/g, l => l.toUpperCase())
  
  const [width, setWidth] = useState(0)
  
  useEffect(() => {
    setWidth(0);
    const timer = setTimeout(() => setWidth(pct), delay * 100 + 100);
    return () => clearTimeout(timer);
  }, [pct, delay])

  return (
    <div className='mb-4 group'>
      <div className='flex justify-between items-end mb-1.5'>
        <span className='text-sm font-medium text-slate-300 group-hover:text-white transition-colors'>{label}</span>
        <span className={`text-xs font-bold px-2 py-0.5 rounded ${positive ? 'bg-red-500/10 text-red-400' : 'bg-emerald-500/10 text-emerald-400'}`}>
          {positive ? '↑ Increases Risk' : '↓ Reduces Risk'}
        </span>
      </div>
      <div className='h-2.5 bg-slate-800/80 rounded-full overflow-hidden shadow-inner'>
        <div className={`h-full rounded-full ${positive ? 'bg-gradient-to-r from-red-600 to-red-400' : 'bg-gradient-to-r from-emerald-600 to-emerald-400'} transition-all duration-700 ease-out`} style={{width:`${width}%`}} />
      </div>
    </div>
  )
}

const DEFAULTS = {
  total_clicks:40, active_days:10, mean_daily_clicks:4, activity_diversity:3,
  weeks_active:3, click_std:5, max_weekly:20, min_weekly:2,
  num_submissions:1, avg_score:55, min_score:45, early_clicks:15,
  gender_M:0, disability_Y:0, edu_level:2, num_of_prev_attempts:0, studied_credits:60
}

const INPUT_GROUPS = [
  {
    title: 'Engagement Metrics',
    icon: '🖱️',
    fields: [
      {k:'total_clicks', l:'Total Clicks'}, {k:'active_days', l:'Active Days'}, 
      {k:'mean_daily_clicks', l:'Avg Daily Clicks'}, {k:'activity_diversity', l:'Activity Types Used'}, 
      {k:'weeks_active', l:'Weeks Active'}, {k:'early_clicks', l:'Early Clicks (Wk 1-2)'}
    ]
  },
  {
    title: 'Academic Performance',
    icon: '📝',
    fields: [
      {k:'num_submissions', l:'Assessments Submitted'}, {k:'avg_score', l:'Avg Score'}
    ]
  },
  {
    title: 'Student Profile',
    icon: '🎓',
    fields: [
      {k:'edu_level', l:'Education Level (0-4)'}, {k:'studied_credits', l:'Credits Enrolled'}, 
      {k:'num_of_prev_attempts', l:'Previous Attempts'}
    ]
  }
]

export default function App() {
  const [form, setForm] = useState(DEFAULTS)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const analyze = async () => {
    setLoading(true); setError(null)
    try {
      const res = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(form)
      })
      if (!res.ok) throw new Error('Network response was not ok')
      const data = await res.json()
      if (data.error) throw new Error(data.error)
      setResult(data)
    } catch(e) { setError(e.message || 'API unavailable — check that the backend is running.') }
    setLoading(false)
  }

  return (
    <div className='min-h-screen bg-[#0B1120] text-slate-200 selection:bg-cyan-500/30'>
      {/* Background Effects */}
      <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] rounded-full bg-blue-900/20 blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[-5%] w-[50%] h-[50%] rounded-full bg-cyan-900/10 blur-[150px]" />
      </div>

      <header className='relative z-50 border-b border-white/5 bg-[#0B1120]/80 backdrop-blur-xl sticky top-0'>
        <div className='max-w-7xl mx-auto px-6 py-5 flex items-center gap-4'>
          <div className='w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20'>
            <span className='text-white font-bold text-xl'>E</span>
          </div>
          <div>
            <h1 className='text-2xl font-black tracking-tight text-white'>EduMine</h1>
            <p className='text-xs font-medium text-cyan-400/80 uppercase tracking-widest'>Early Warning System</p>
          </div>
        </div>
      </header>

      <main className='relative z-10 max-w-7xl mx-auto px-6 py-12'>
        <div className='grid grid-cols-1 xl:grid-cols-12 gap-8'>
          
          {/* Left Column: Input Form */}
          <div className='xl:col-span-7 space-y-6'>
            <div className='mb-8'>
              <h2 className='text-3xl font-bold text-white mb-3'>Student Analyzer</h2>
              <p className='text-slate-400 text-sm leading-relaxed max-w-2xl'>
                Input LMS engagement metrics from the first 3 weeks to predict the likelihood of student dropout. 
                Powered by an XGBoost model trained on 32,593 Open University records.
              </p>
            </div>

            <div className='bg-slate-900/50 backdrop-blur-md border border-white/5 rounded-3xl p-8 shadow-2xl'>
              <div className='space-y-8'>
                {INPUT_GROUPS.map((group, idx) => (
                  <div key={idx} className='space-y-4'>
                    <h3 className='text-lg font-semibold text-white flex items-center gap-2 border-b border-white/5 pb-2'>
                      <span>{group.icon}</span> {group.title}
                    </h3>
                    <div className='grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-4'>
                      {group.fields.map(({k, l}) => (
                        <div key={k} className='flex flex-col gap-1.5'>
                          <label className='text-xs font-medium text-slate-400'>{l}</label>
                          <input 
                            type='number' 
                            value={form[k]} 
                            onChange={e=>setForm({...form,[k]:+e.target.value})}
                            className='w-full bg-[#0B1120] border border-slate-700/50 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500/50 transition-all shadow-inner' 
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>

              <button 
                onClick={analyze} 
                disabled={loading}
                className='w-full mt-10 py-4 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-2xl font-bold text-white tracking-wide shadow-lg shadow-cyan-500/25 transition-all duration-300 transform hover:-translate-y-0.5 active:translate-y-0 flex justify-center items-center gap-2'
              >
                {loading ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-2 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                    Analyzing Profile...
                  </>
                ) : 'Generate Prediction'}
              </button>
              
              {error && (
                <div className='mt-4 p-4 bg-red-500/10 border border-red-500/20 rounded-xl flex items-start gap-3'>
                  <span className='text-red-400 text-xl leading-none'>⚠️</span>
                  <p className='text-red-400 text-sm font-medium'>{error}</p>
                </div>
              )}
            </div>
          </div>

          {/* Right Column: Results */}
          <div className='xl:col-span-5'>
            <div className='sticky top-28 space-y-6'>
              {result ? (
                <>
                  <div className='bg-slate-900/80 backdrop-blur-xl border border-white/10 rounded-3xl p-8 shadow-2xl relative overflow-hidden'>
                    {/* Decorative glow behind the gauge */}
                    <div className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-48 rounded-full blur-[80px] opacity-20 pointer-events-none ${RISK_COLOR[result.risk_label]?.bg.split('/')[0]}`} />
                    
                    <h3 className='text-center text-sm font-semibold text-slate-400 uppercase tracking-widest mb-2'>Risk Assessment</h3>
                    <RiskGauge pct={result.risk_percent} label={result.risk_label} />
                  </div>

                  <div className='bg-slate-900/50 backdrop-blur-md border border-white/5 rounded-3xl p-8 shadow-xl'>
                    <div className='flex items-center justify-between mb-6'>
                      <h3 className='text-lg font-bold text-white'>Key Risk Drivers</h3>
                      <span className='text-xs font-medium bg-slate-800 px-2.5 py-1 rounded-md text-slate-400'>SHAP Analysis</span>
                    </div>
                    
                    <div className='space-y-2'>
                      {result.top_factors?.map((f, i) => (
                        <ShapBar key={i} feature={f.feature} shap={f.shap} delay={i} />
                      ))}
                    </div>
                  </div>
                </>
              ) : (
                <div className='h-full min-h-[500px] flex flex-col items-center justify-center border-2 border-dashed border-slate-700/50 rounded-3xl bg-slate-900/20 p-10 text-center'>
                  <div className='w-20 h-20 bg-slate-800/50 rounded-full flex items-center justify-center mb-6 shadow-inner'>
                    <span className='text-3xl opacity-50'>📊</span>
                  </div>
                  <h3 className='text-xl font-bold text-white mb-2'>Awaiting Data</h3>
                  <p className='text-slate-500 text-sm max-w-xs'>
                    Fill out the student's engagement metrics and run the analyzer to see the dropout risk prediction and contributing factors.
                  </p>
                </div>
              )}
            </div>
          </div>

        </div>
      </main>
    </div>
  )
}
