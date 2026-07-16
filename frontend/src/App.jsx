import { useState } from 'react'

const API = import.meta.env.VITE_API_URL

const RISK_COLOR = {
  'High Risk':     { ring:'ring-red-500',     bg:'bg-red-500',     text:'text-red-400'     },
  'Moderate Risk': { ring:'ring-yellow-400',  bg:'bg-yellow-400',  text:'text-yellow-300'  },
  'Low Risk':      { ring:'ring-emerald-500', bg:'bg-emerald-500', text:'text-emerald-400' },
}

function RiskGauge({ pct, label }) {
  const colors = RISK_COLOR[label] || RISK_COLOR['Low Risk']
  const deg = (pct / 100) * 180

  return (
    <div className='flex flex-col items-center'>
      <div className={`w-40 h-20 rounded-t-full border-8 ${colors.ring} border-b-0 relative overflow-hidden`}>
        <div className='absolute bottom-0 left-1/2 w-1 h-16 bg-slate-300 origin-bottom transition-transform duration-700'
             style={{transform:`translateX(-50%) rotate(${deg-90}deg)`}} />
      </div>
      <div className={`mt-3 text-3xl font-black ${colors.text}`}>{pct}%</div>
      <div className={`text-sm font-semibold mt-1 ${colors.text}`}>{label}</div>
    </div>
  )
}

function ShapBar({ feature, shap }) {
  const positive = shap > 0
  const pct = Math.min(100, Math.abs(shap) * 200)
  const label = feature.replace(/_/g,' ')

  return (
    <div className='mb-2'>
      <div className='flex justify-between text-xs text-slate-400 mb-0.5'>
        <span>{label}</span>
        <span className={positive ? 'text-red-400' : 'text-emerald-400'}>
          {positive ? '↑ increases risk' : '↓ reduces risk'}
        </span>
      </div>
      <div className='h-2 bg-slate-700 rounded-full overflow-hidden'>
        <div className={`h-full rounded-full ${positive ? 'bg-red-500' : 'bg-emerald-500'}`} style={{width:`${pct}%`}} />
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

const FIELD_LABELS = {
  total_clicks:'Total Clicks', active_days:'Active Days', mean_daily_clicks:'Avg Daily Clicks',
  activity_diversity:'Activity Types Used', weeks_active:'Weeks Active',
  num_submissions:'Assessments Submitted', avg_score:'Avg Score', early_clicks:'Early Clicks (Wk 1-2)',
  edu_level:'Education Level (0-4)', studied_credits:'Credits Enrolled', num_of_prev_attempts:'Previous Attempts',
}

export default function App() {
  const [form, setForm] = useState(DEFAULTS)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const analyze = async () => {
    setLoading(true); setError(null)
    try {
      const res = await fetch(`${API}/predict`,{
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify(form)
      })
      if (!res.ok) throw new Error('Network response was not ok')
      setResult(await res.json())
    } catch(e) { setError('API unavailable.') }
    setLoading(false)
  }

  return (
    <div className='min-h-screen bg-gradient-to-br from-slate-900 via-blue-950 to-slate-900 text-white'>
      <header className='border-b border-white/10 bg-white/5 backdrop-blur-md sticky top-0 z-10'>
        <div className='max-w-6xl mx-auto px-6 py-4 flex items-center gap-4'>
          <span className='text-2xl font-bold bg-gradient-to-r from-blue-400 to-cyan-300 bg-clip-text text-transparent'>📊 EduMine</span>
          <span className='text-slate-400 text-sm'>At-Risk Student Early Warning System</span>
        </div>
      </header>

      <main className='max-w-6xl mx-auto px-6 py-10'>
        <div className='text-center mb-10'>
          <h1 className='text-4xl font-extrabold mb-2 bg-gradient-to-r from-blue-300 to-cyan-200 bg-clip-text text-transparent'>Student Risk Analyzer</h1>
          <p className='text-slate-400 max-w-2xl mx-auto'>Input week-3 LMS engagement metrics to predict dropout risk. Powered by XGBoost trained on 32,593 OULAD students with SHAP explainability.</p>
        </div>

        <div className='grid grid-cols-1 lg:grid-cols-2 gap-8'>
          {/* Input form */}
          <div className='bg-white/5 border border-white/10 rounded-2xl p-6 space-y-3'>
            <h2 className='text-blue-300 font-semibold mb-4'>Student Engagement (Week 1-3 Data)</h2>
            {Object.entries(FIELD_LABELS).map(([k,label])=>(
              <div key={k} className='flex justify-between items-center'>
                <label className='text-slate-300 text-sm'>{label}</label>
                <input type='number' value={form[k]} onChange={e=>setForm({...form,[k]:+e.target.value})}
                  className='w-24 text-right bg-slate-800 border border-slate-700 rounded-lg px-2 py-1 text-sm text-white focus:outline-none focus:border-blue-400' />
              </div>
            ))}
            <button onClick={analyze} disabled={loading}
              className='w-full mt-4 py-2.5 bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 disabled:opacity-40 rounded-xl font-bold transition'>
              {loading ? 'Analyzing...' : 'Predict Risk →'}
            </button>
            {error && <p className='text-red-400 text-sm'>{error}</p>}
          </div>

          {/* Results */}
          <div className='space-y-6'>
            {result ? (<>
              <div className='bg-white/5 border border-white/10 rounded-2xl p-6 flex flex-col items-center'>
                <RiskGauge pct={result.risk_percent} label={result.risk_label} />
              </div>

              <div className='bg-white/5 border border-white/10 rounded-2xl p-6'>
                <h3 className='text-blue-300 font-semibold mb-4'>Why? — Top Contributing Factors</h3>
                {result.top_factors?.map((f,i)=><ShapBar key={i} feature={f.feature} shap={f.shap} />)}
                <p className='text-xs text-slate-500 mt-3'>SHAP values: positive = increases at-risk probability, negative = reduces it</p>
              </div>
            </>) : (
              <div className='h-full flex items-center justify-center border border-white/10 rounded-2xl bg-white/5 min-h-64'>
                <p className='text-slate-500'>Enter student data and click Predict</p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}
