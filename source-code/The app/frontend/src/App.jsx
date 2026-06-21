import React, { useState, useRef, useEffect, useMemo } from 'react';
import { Send, Calculator, MessageSquare, TrendingUp, AlertCircle, Loader2, Sun, Moon, User, Users, Database, BarChart3, LogOut, ChevronDown, Shield } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:5001';

const FACTOR_LABELS = [
  'Leadership Engagement',
  'Cultural Change',
  'Communication',
  'Training',
  'Tools & Techniques',
  'Employee Involvement',
  'Expertise & Skills'
];

const DEFAULT_LEVEL_CHOICES = [
  { level: 1, title: 'Initial', detail: 'No formal practice; ad-hoc or absent.' },
  { level: 2, title: 'Basic', detail: 'Some awareness; inconsistent execution.' },
  { level: 3, title: 'Defined', detail: 'Standardized and repeatable.' },
  { level: 4, title: 'Managed', detail: 'Measured, controlled, and improving.' },
  { level: 5, title: 'Optimized', detail: 'Continuous improvement; best practice.' }
];

const LOCAL_MODEL_OPTIONS = [
  'llama3:8b',
  'llama3:70b',
  'llama3.1:8b',
  'mistral:7b',
  'mixtral:8x7b',
  'gemma2:9b',
  'qwen2.5:7b',
  'phi3:mini'
];

const API_MODEL_CATALOG = [
  {
    provider: 'Google',
    families: [
      {
        family: 'Gemini 3',
        models: ['Gemini 3 Pro', 'Gemini 3 Flash']
      },
      {
        family: 'Gemini 2.5',
        models: ['Gemini 2.5 Pro', 'Gemini 2.5 Flash', 'Gemini 2.5 Flash-Lite']
      }
    ]
  },
  {
    provider: 'OpenAI',
    families: [
      {
        family: 'GPT-5.2',
        models: ['GPT-5.2 Instant', 'GPT-5.2 Thinking', 'GPT-5.2 Pro']
      },
      {
        family: 'GPT-5.1',
        models: ['GPT-5.1 Instant', 'GPT-5.1 Thinking']
      },
      {
        family: 'GPT-4.1',
        models: ['GPT-4.1', 'GPT-4.1 Mini', 'GPT-4.1 Nano']
      },
      {
        family: 'Specialized',
        models: ['Codex', 'Vision', 'Audio']
      }
    ]
  },
  {
    provider: 'Anthropic',
    families: [
      {
        family: 'Claude 4.5',
        models: ['Claude Opus 4.5', 'Claude Sonnet 4.5', 'Claude Haiku 4.5']
      }
    ]
  }
];

const ASSISTED_STEPS = [
  ...FACTOR_LABELS.map((label, index) => ({
    group: 'IL',
    index,
    code: `IL${index + 1}`,
    label: `${label} (Lean)`
  })),
  ...FACTOR_LABELS.map((label, index) => ({
    group: 'IS',
    index,
    code: `IS${index + 1}`,
    label: `${label} (Six Sigma)`
  })),
  ...FACTOR_LABELS.map((label, index) => ({
    group: 'IM',
    index,
    code: `IM${index + 1}`,
    label: `${label} (Maturity)`
  }))
];

const ASSISTED_LABELS = Object.fromEntries(
  ASSISTED_STEPS.map((step) => [step.code, step.label])
);

const INTRO_MESSAGES = [
  {
    role: 'assistant',
    content: 'Welcome to the AI-Driven Lean Six Sigma Performance Engine. Ask about DMAIC, capability, or improvement strategy.'
  },
  {
    role: 'assistant',
    content: 'Start the Guided Assessment for step by step scoring of IL, IS, and IM factors.'
  }
];

const getIntroMessages = () => INTRO_MESSAGES.map((message) => ({ ...message }));

const createEmptyParameters = () => ({
  IL: Array(7).fill(''),
  IS: Array(7).fill(''),
  IM: Array(7).fill('')
});

const buildAssistedPrompt = (stepIndex) => {
  const step = ASSISTED_STEPS[stepIndex];
  const position = stepIndex + 1;
  return `Step ${position} of ${ASSISTED_STEPS.length}: ${step.code} - ${step.label}. Enter a score from 1 to 5.`;
};

const LeanSixSigmaChatbot = () => {
  const [messages, setMessages] = useState(getIntroMessages);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('chat');
  const [themeMode, setThemeMode] = useState('light');
  const [showIntro, setShowIntro] = useState(true);
  const [uiLanguage, setUiLanguage] = useState('en');
  const [loadingMessage, setLoadingMessage] = useState('');
  const [parameters, setParameters] = useState(createEmptyParameters());
  const [panelsCollapsed, setPanelsCollapsed] = useState(false);
  const [assistedAssessment, setAssistedAssessment] = useState({
    active: false,
    stepIndex: 0,
    parameters: createEmptyParameters(),
    review: false
  });
  const [csfLevels, setCsfLevels] = useState({ levels: {}, labels: {}, prescriptions: {} });
  const [calcResult, setCalcResult] = useState(null);
  const [lastCalcParams, setLastCalcParams] = useState(createEmptyParameters());
  const [showModelSettings, setShowModelSettings] = useState(false);
  const [modelConfig, setModelConfig] = useState({
    localModel: 'llama3:8b',
    apiModel: 'gemini-pro',
    apiKey: '',
    useApi: false
  });
  const [apiTestState, setApiTestState] = useState({
    status: 'idle',
    message: '',
    details: null
  });
  const [calcExplanation, setCalcExplanation] = useState('');
  const [calcExplanationLoading, setCalcExplanationLoading] = useState(false);
  const [explainLoadingId, setExplainLoadingId] = useState(null);
  // ===== USER AUTH STATE =====
  const [currentUser, setCurrentUser] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('lss_user') || 'null');
    } catch { return null; }
  });
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const [adminPanel, setAdminPanel] = useState(null); // null | 'users' | 'companies' | 'stats'
 
  const handleLogout = () => {
    localStorage.removeItem('lss_token');
    localStorage.removeItem('lss_user');
    window.location.href = '/login.html';
  };
 
  const isAdmin = currentUser?.role === 'admin';
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);
  const inputRef = useRef(null);
  const panelsCollapsedRef = useRef(false);

  const historyItems = useMemo(
    () => messages.filter((msg) => msg.role === 'user'),
    [messages]
  );
  const hasAnyParameter = useMemo(
    () => Object.values(parameters).some((values) => values.some((value) => value !== '')),
    [parameters]
  );
  const reviewReady = useMemo(() => {
    if (!assistedAssessment.review) return false;
    const groups = ['IL', 'IS', 'IM'];
    return groups.every((group) => {
      const values = assistedAssessment.parameters?.[group] || [];
      if (values.length !== 7) return false;
      return values.every((value) => {
        const parsed = Number(value);
        return Number.isInteger(parsed) && parsed >= 1 && parsed <= 5;
      });
    });
  }, [assistedAssessment.parameters, assistedAssessment.review]);
  const isDark = themeMode === 'dark';
  const themeClass = (darkClass, lightClass) => (isDark ? darkClass : lightClass);
  const activeModelLabel = modelConfig.useApi
    ? `API (${modelConfig.apiModel.trim() || 'gemini-pro'})`
    : `Local (${modelConfig.localModel.trim() || 'llama3:8b'})`;
  const apiModelSelected = modelConfig.apiModel.trim().toLowerCase();
  const apiTestBadge = useMemo(() => {
    if (apiTestState.status === 'idle') return null;
    if (apiTestState.status === 'loading') {
      return { label: 'Testing', className: 'badge badge--neutral' };
    }
    if (apiTestState.status === 'api') {
      return { label: 'API OK', className: 'badge badge--low' };
    }
    if (apiTestState.status === 'fallback') {
      return { label: 'Fallback', className: 'badge badge--mid' };
    }
    return { label: 'Error', className: 'badge badge--high' };
  }, [apiTestState.status]);
  const guidedStatus = assistedAssessment.active
    ? `In progress ${assistedAssessment.stepIndex + 1}/${ASSISTED_STEPS.length}`
    : assistedAssessment.review
      ? 'Review pending'
      : 'Ready';
  const autoCollapseEnabled = !showModelSettings && !assistedAssessment.active && !assistedAssessment.review;

  useEffect(() => {
    if (!loading) {
      setLoadingMessage('');
      return;
    }
    const steps = [
      'Analyzing your question...',
      'Reviewing L6S knowledge base...',
      'Generating response...'
    ];
    let index = 0;
    setLoadingMessage(steps[index]);
    const interval = setInterval(() => {
      index = Math.min(index + 1, steps.length - 1);
      setLoadingMessage(steps[index]);
    }, 1500);
    return () => clearInterval(interval);
  }, [loading]);

  const toAverage = (values) => {
    const nums = values
      .map((value) => Number(value))
      .filter((value) => Number.isFinite(value));
    if (!nums.length) return 0;
    return nums.reduce((sum, value) => sum + value, 0) / nums.length;
  };

  const localAnalysis = useMemo(() => ({
    IL_average: toAverage(parameters.IL),
    IS_average: toAverage(parameters.IS),
    IM_average: toAverage(parameters.IM)
  }), [parameters]);

  const analysis = useMemo(
    () => calcResult?.parameter_analysis || localAnalysis,
    [calcResult, localAnalysis]
  );

  const radarValues = useMemo(() => ([
    { label: 'IL', value: analysis.IL_average },
    { label: 'IS', value: analysis.IS_average },
    { label: 'IM', value: analysis.IM_average }
  ]), [analysis]);

  const radarSize = 180;
  const { radarPoints, radarAxisPoints } = useMemo(() => {
    const radarCenter = radarSize / 2;
    const radarRadius = radarCenter - 12;
    const points = radarValues.map((item, index) => {
      const angle = (Math.PI * 2 * index) / radarValues.length - Math.PI / 2;
      const normalized = Math.max(0, Math.min(1, (item.value || 0) / 5));
      const r = radarRadius * normalized;
      const x = radarCenter + r * Math.cos(angle);
      const y = radarCenter + r * Math.sin(angle);
      return `${x},${y}`;
    }).join(' ');
    const axisPoints = radarValues.map((_, index) => {
      const angle = (Math.PI * 2 * index) / radarValues.length - Math.PI / 2;
      const x = radarCenter + radarRadius * Math.cos(angle);
      const y = radarCenter + radarRadius * Math.sin(angle);
      return `${x},${y}`;
    }).join(' ');
    return { radarPoints: points, radarAxisPoints: axisPoints };
  }, [radarValues]);

  const radarColors = useMemo(() => ({
    grid: isDark ? '#243041' : '#d4dde8',
    stroke: 'var(--accent)',
    fill: isDark ? 'rgba(var(--brand-blue-rgb), 0.32)' : 'rgba(var(--brand-blue-rgb), 0.2)'
  }), [isDark]);

  const formatMetric = (value, digits = 1) => {
    const num = Number(value);
    if (!Number.isFinite(num)) {
      return value ? String(value) : 'N/A';
    }
    return num.toFixed(digits);
  };

  const getGapBadgeClass = (value) => {
    const num = Number(value);
    if (!Number.isFinite(num)) return 'badge--neutral';
    if (num >= 15) return 'badge--high';
    if (num >= 7) return 'badge--mid';
    return 'badge--low';
  };

  const getCategoryToneClass = (category) => {
    const lower = String(category || '').toLowerCase();
    if (lower.includes('financial')) return 'tone-amber';
    if (lower.includes('operational')) return 'tone-sky';
    if (lower.includes('innovation')) return 'tone-teal';
    return 'tone-neutral';
  };

  const getFactorToneClass = (code) => {
    const normalized = String(code || '').toUpperCase();
    if (normalized.startsWith('IL')) return 'tone-amber';
    if (normalized.startsWith('IS')) return 'tone-sky';
    if (normalized.startsWith('IM')) return 'tone-teal';
    return 'tone-neutral';
  };

  const renderRadarSvg = (labels, values, maxValue) => {
    const size = 220;
    const center = size / 2;
    const radius = center - 26;
    const gridLevels = 4;
    const labelStep = Math.max(1, Math.ceil(labels.length / 10));
    const gridColor = isDark ? '#1f2937' : '#e2e8f0';
    const areaStroke = 'var(--accent)';
    const areaFill = isDark ? 'rgba(var(--brand-blue-rgb), 0.26)' : 'rgba(var(--brand-blue-rgb), 0.18)';
    const pointFor = (value, index, scale = 1) => {
      const angle = (Math.PI * 2 * index) / labels.length - Math.PI / 2;
      const r = radius * scale * (value / maxValue);
      const x = center + r * Math.cos(angle);
      const y = center + r * Math.sin(angle);
      return { x, y, angle };
    };

    const polygonPoints = values.map((value, index) => {
      const point = pointFor(value, index, 1);
      return `${point.x},${point.y}`;
    }).join(' ');

    return (
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {Array.from({ length: gridLevels }).map((_, level) => {
          const scale = (level + 1) / gridLevels;
          const gridPoints = labels.map((_, index) => {
            const angle = (Math.PI * 2 * index) / labels.length - Math.PI / 2;
            const r = radius * scale;
            const x = center + r * Math.cos(angle);
            const y = center + r * Math.sin(angle);
            return `${x},${y}`;
          }).join(' ');
          return (
            <polygon
              key={`grid-${scale}`}
              points={gridPoints}
              fill="none"
              stroke={gridColor}
              strokeWidth="1"
            />
          );
        })}
        {labels.map((label, index) => {
          const point = pointFor(maxValue, index, 1);
          return (
            <line
              key={`axis-${label}-${index}`}
              x1={center}
              y1={center}
              x2={point.x}
              y2={point.y}
              stroke={gridColor}
              strokeWidth="1"
            />
          );
        })}
        <polygon points={polygonPoints} fill={areaFill} stroke={areaStroke} strokeWidth="2" />
        {labels.map((label, index) => {
          if (index % labelStep !== 0) return null;
          const point = pointFor(maxValue, index, 1);
          const labelOffset = 10;
          const x = center + (radius + labelOffset) * Math.cos(point.angle);
          const y = center + (radius + labelOffset) * Math.sin(point.angle);
          const anchor = Math.cos(point.angle) > 0.2 ? 'start' : Math.cos(point.angle) < -0.2 ? 'end' : 'middle';
          const baseline = Math.sin(point.angle) > 0.2 ? 'hanging' : Math.sin(point.angle) < -0.2 ? 'auto' : 'middle';
          return (
            <text
              key={`label-${label}-${index}`}
              x={x}
              y={y}
              textAnchor={anchor}
              dominantBaseline={baseline}
              fontSize="9"
              fill={isDark ? '#94a3b8' : '#64748b'}
            >
              {label}
            </text>
          );
        })}
      </svg>
    );
  };

  const renderChartBlock = (chartData, title) => {
    if (!chartData?.labels?.length || !chartData?.datasets?.length) return null;
    const dataset = chartData.datasets[0] || {};
    const values = (dataset.data || []).map((value) => (Number.isFinite(value) ? value : 0));
    const maxValue = Math.max(...values, 1);
    const showRadar = chartData.labels.length >= 3;
    return (
      <div className={`mt-3 rounded-xl border p-4 ${themeClass('border-slate-800 bg-slate-900/80', 'border-slate-200 bg-white')}`}>
        <div className={`text-sm font-semibold ${themeClass('text-slate-100', 'text-slate-900')}`}>
          {title || dataset.label || 'CSF Profile'}
        </div>
        <div className={`mt-3 grid gap-4 ${showRadar ? 'md:grid-cols-[220px_1fr]' : ''}`}>
          {showRadar && (
            <div className="flex items-center justify-center">
              {renderRadarSvg(chartData.labels, values, maxValue)}
            </div>
          )}
          <div className="space-y-2 max-h-72 overflow-y-auto pr-2">
            {chartData.labels.map((label, index) => {
              const value = values[index] ?? 0;
              const width = `${Math.round((value / maxValue) * 100)}%`;
              return (
                <div key={`${label}-${index}`} className="flex items-center gap-2 text-xs">
                  <div className={`w-10 shrink-0 ${themeClass('text-slate-400', 'text-slate-500')}`}>{label}</div>
                  <div className={`h-2 flex-1 rounded-full ${themeClass('bg-slate-800', 'bg-slate-200')}`}>
                    <div className="h-2 rounded-full bg-accent" style={{ width }} />
                  </div>
                  <div className={`w-12 text-right ${themeClass('text-slate-300', 'text-slate-600')}`}>
                    {formatMetric(value, 2)}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    );
  };

  const renderPerformanceTable = (tableData) => {
    if (!tableData) return null;
    const summary = tableData.high_level_summary || {};
    const primary = tableData.primary_criteria || [];
    const subCriteria = tableData.sub_criteria || {};
    const subEntries = Object.entries(subCriteria);

    return (
      <div className={`mt-4 rounded-xl border p-4 ${themeClass('border-slate-800 bg-slate-900/80', 'border-slate-200 bg-white')}`}>
        <div className={`text-sm font-semibold ${themeClass('text-slate-100', 'text-slate-900')}`}>
          Performance Table
        </div>
        <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-3">
          <div className="stat-card stat-card--strategy">
            <div className="stat-card__label">Strategy</div>
            <div className="stat-card__value">{summary.recommended_strategy || 'N/A'}</div>
          </div>
          <div className="stat-card stat-card--current">
            <div className="stat-card__label">Current</div>
            <div className="stat-card__value stat-card__value--big">
              {formatMetric(summary.current_performance)}%
            </div>
          </div>
          <div className="stat-card stat-card--target">
            <div className="stat-card__label">Target</div>
            <div className="stat-card__value stat-card__value--big">
              {formatMetric(summary.target_performance)}%
            </div>
          </div>
        </div>

        {primary.length > 0 && (
          <div className="mt-4 overflow-x-auto">
            <table className="min-w-full text-xs data-table">
              <thead>
                <tr className={themeClass('text-slate-400', 'text-slate-500')}>
                  <th className="px-2 py-2 text-left">Criterion</th>
                  <th className="px-2 py-2 text-right">Weight</th>
                  <th className="px-2 py-2 text-right">Current</th>
                  <th className="px-2 py-2 text-right">Target</th>
                  <th className="px-2 py-2 text-right">Gap</th>
                </tr>
              </thead>
              <tbody>
                {primary.map((row, index) => {
                  const gapClass = getGapBadgeClass(row.gap);
                  const gapValue = formatMetric(row.gap);
                  const gapLabel = gapValue === 'N/A' ? 'N/A' : `${gapValue}%`;
                  return (
                    <tr key={`${row.criterion}-${index}`} className={themeClass('border-slate-800', 'border-slate-200')}>
                      <td className={`px-2 py-2 ${themeClass('text-slate-200', 'text-slate-700')}`}>{row.criterion}</td>
                      <td className={`px-2 py-2 text-right ${themeClass('text-slate-300', 'text-slate-600')}`}>{formatMetric(row.weight, 3)}</td>
                      <td className={`px-2 py-2 text-right ${themeClass('text-slate-300', 'text-slate-600')}`}>{formatMetric(row.real_performance)}%</td>
                      <td className={`px-2 py-2 text-right ${themeClass('text-slate-300', 'text-slate-600')}`}>{formatMetric(row.target_performance)}%</td>
                      <td className={`px-2 py-2 text-right ${themeClass('text-slate-300', 'text-slate-600')}`}>
                        <span className={`badge ${gapClass}`}>{gapLabel}</span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {subEntries.length > 0 && (
          <div className="mt-4 space-y-3">
            {subEntries.map(([category, rows]) => (
              <div key={category} className={`criteria-card ${getCategoryToneClass(category)}`}>
                <div className={`criteria-title text-xs ${themeClass('text-slate-200', 'text-slate-700')}`}>
                  {category}
                </div>
                <div className="mt-2 overflow-x-auto">
                  <table className="min-w-full text-xs data-table data-table--compact">
                    <thead>
                      <tr className={themeClass('text-slate-400', 'text-slate-500')}>
                        <th className="px-2 py-1 text-left">Sub-criterion</th>
                        <th className="px-2 py-1 text-right">Weight</th>
                        <th className="px-2 py-1 text-right">Contribution</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(rows || []).map((row, index) => (
                        <tr key={`${category}-${index}`} className={themeClass('border-slate-800', 'border-slate-200')}>
                          <td className={`px-2 py-1 ${themeClass('text-slate-200', 'text-slate-700')}`}>{row.subcategory}</td>
                          <td className={`px-2 py-1 text-right ${themeClass('text-slate-300', 'text-slate-600')}`}>{formatMetric(row.weight, 3)}</td>
                          <td className={`px-2 py-1 text-right ${themeClass('text-slate-300', 'text-slate-600')}`}>{formatMetric(row.contribution)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  };

  const buildLevelUpActions = (params) => {
    const actions = [];
    const addActions = (prefix, values) => {
      if (!Array.isArray(values)) return;
      values.forEach((value, index) => {
        const parsed = Number(value);
        if (!Number.isFinite(parsed)) return;
        const current = Math.max(1, Math.min(5, Math.round(parsed)));
        const nextLevel = Math.min(current + 1, 5);
        const code = `${prefix}${index + 1}`;
        const actionMap = csfLevels.prescriptions?.[code] || {};
        const nextAction = actionMap[nextLevel] || (current >= 5 ? 'Maintain current level.' : 'No prescription available.');

        actions.push({
          code,
          label: csfLevels.labels?.[code] || code,
          current,
          next: nextLevel,
          action: nextAction
        });
      });
    };

    addActions('IL', params?.IL || []);
    addActions('IS', params?.IS || []);
    addActions('IM', params?.IM || []);
    return actions;
  };

  const renderLevelUpTable = (params) => {
    const actions = buildLevelUpActions(params);
    if (!actions.length) return null;

    return (
      <div className={`mt-4 rounded-xl border p-4 ${themeClass('border-slate-800 bg-slate-900/80', 'border-slate-200 bg-white')}`}>
        <div className={`text-sm font-semibold ${themeClass('text-slate-100', 'text-slate-900')}`}>
          Next-Level Action Table
        </div>
        <div className="mt-3 overflow-x-auto">
          <table className="min-w-full text-xs data-table">
            <thead>
              <tr className={themeClass('text-slate-400', 'text-slate-500')}>
                <th className="px-2 py-2 text-left">Factor</th>
                <th className="px-2 py-2 text-right">Current</th>
                <th className="px-2 py-2 text-right">Next</th>
                <th className="px-2 py-2 text-left">Action to Reach Next Level</th>
              </tr>
            </thead>
            <tbody>
              {actions.map((row, index) => {
                const toneClass = getFactorToneClass(row.code);
                return (
                  <tr key={`${row.code}-${index}`} className={`${themeClass('border-slate-800', 'border-slate-200')} ${toneClass}`}>
                    <td className={`px-2 py-2 ${themeClass('text-slate-200', 'text-slate-700')}`}>
                      <div className={`factor-cell ${toneClass}`}>
                        <span className="factor-tag">{row.code}</span>
                        <span className={`factor-label ${themeClass('text-slate-200', 'text-slate-700')}`}>
                          {row.label}
                        </span>
                      </div>
                    </td>
                    <td className={`px-2 py-2 text-right ${themeClass('text-slate-300', 'text-slate-600')}`}>
                      <span className="level-pill level-pill--current">{row.current}</span>
                    </td>
                    <td className={`px-2 py-2 text-right ${themeClass('text-slate-300', 'text-slate-600')}`}>
                      <span className="level-pill level-pill--next">{row.next}</span>
                    </td>
                    <td className={`px-2 py-2 action-cell ${themeClass('text-slate-300', 'text-slate-600')}`}>
                      {row.action}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  
  const renderTargetActionTable = (table) => {
    if (!table?.rows?.length) return null;

    const title = table.title || 'Target Action Table';
    const normalizeLevel = (value) => {
      if (value === null || value === undefined || value === '') return null;
      const num = Math.round(Number(value));
      if (!Number.isFinite(num) || num < 1 || num > 5) return null;
      return num;
    };

    const grouped = table.rows.reduce((acc, row) => {
      const key = row.factor || row.label || 'Unknown';
      if (!acc[key]) {
        acc[key] = {
          factor: row.factor || key,
          label: row.label || row.factor || key,
          current: null,
          target: null,
          steps: []
        };
      }

      const currentValue = normalizeLevel(row.current);
      if (currentValue !== null && acc[key].current === null) {
        acc[key].current = currentValue;
      }

      const targetValue = normalizeLevel(row.target);
      if (targetValue !== null) {
        acc[key].target = targetValue;
      }

      const levelValue = normalizeLevel(row.level);
      if (levelValue !== null) {
        acc[key].steps.push({
          level: levelValue,
          action: row.action || 'N/A'
        });
      }

      return acc;
    }, {});

    const groups = Object.values(grouped)
      .map((group) => ({
        ...group,
        steps: (group.steps || []).sort((a, b) => (a.level ?? 0) - (b.level ?? 0))
      }))
      .sort((a, b) => `${a.factor}`.localeCompare(`${b.factor}`));

    const showCurrent = groups.some((group) => group.current !== null);
    const needsCurrent = groups.every((group) => group.current === null);

    return (
      <div className={`mt-4 rounded-xl border p-4 ${themeClass('border-slate-800 bg-slate-900/80', 'border-slate-200 bg-white')}`}>
        <div className={`text-sm font-semibold ${themeClass('text-slate-100', 'text-slate-900')}`}>
          {title}
        </div>
        {needsCurrent && (
          <div className={`mt-2 text-xs ${themeClass('text-slate-400', 'text-slate-500')}`}>
            Provide current CSF scores to personalize the steps.
          </div>
        )}
        <div className="mt-3 overflow-x-auto">
          <table className="min-w-full text-xs align-top data-table data-table--compact">
            <thead>
              <tr className={themeClass('text-slate-400', 'text-slate-500')}>
                <th className="px-2 py-2 text-left">Factor</th>
                {showCurrent && <th className="px-2 py-2 text-right">Current</th>}
                <th className="px-2 py-2 text-right">Target</th>
                <th className="px-2 py-2 text-left">Steps to Reach Target</th>
              </tr>
            </thead>
            <tbody>
              {groups.map((group, index) => {
                const toneClass = getFactorToneClass(group.factor);
                return (
                  <tr key={`${group.factor}-${index}`} className={`${themeClass('border-slate-800', 'border-slate-200')} ${toneClass}`}>
                    <td className={`px-2 py-2 ${themeClass('text-slate-200', 'text-slate-700')}`}>
                      <div className={`factor-cell ${toneClass}`}>
                        <span className="factor-tag">{group.factor}</span>
                        <span className={`factor-label ${themeClass('text-slate-200', 'text-slate-700')}`}>
                          {group.label}
                        </span>
                      </div>
                    </td>
                    {showCurrent && (
                      <td className={`px-2 py-2 text-right ${themeClass('text-slate-300', 'text-slate-600')}`}>
                        {group.current === null ? (
                          'N/A'
                        ) : (
                          <span className="level-pill level-pill--current">{group.current}</span>
                        )}
                      </td>
                    )}
                    <td className={`px-2 py-2 text-right ${themeClass('text-slate-300', 'text-slate-600')}`}>
                      {group.target === null ? (
                        'N/A'
                      ) : (
                        <span className="level-pill level-pill--target">{group.target}</span>
                      )}
                    </td>
                    <td className={`px-2 py-2 action-cell ${themeClass('text-slate-300', 'text-slate-600')}`}>
                      {group.steps.length ? (
                        <ol className="list-decimal pl-4 space-y-1">
                          {group.steps.map((step) => (
                            <li key={`${group.factor}-${step.level}`}>
                              <span className="step-chip">L{step.level}</span>{' '}
                              {step.action}
                            </li>
                          ))}
                        </ol>
                      ) : (
                        <span>N/A</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    );
  };



  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', themeMode);
    document.documentElement.style.colorScheme = themeMode;
  }, [themeMode]);

  useEffect(() => {
    if (assistedAssessment.active) return;
    scrollToBottom();
  }, [messages, assistedAssessment.active]);

  useEffect(() => {
    if ((showModelSettings || assistedAssessment.active || assistedAssessment.review)
      && panelsCollapsedRef.current) {
      updatePanelsCollapsed(false);
    }
  }, [showModelSettings, assistedAssessment.active, assistedAssessment.review]);

  useEffect(() => {
    const timer = setTimeout(() => {
      setShowIntro(false);
    }, 3200);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    const loadLevels = async () => {
      try {
        const response = await fetch(`${API_BASE}/csf-levels?lang=${uiLanguage}`);
        if (!response.ok) return;
        const data = await response.json();
        setCsfLevels({
          levels: data.levels || {},
          labels: data.labels || {},
          prescriptions: data.prescriptions || {}
        });
      } catch (error) {
        setCsfLevels({ levels: {}, labels: {}, prescriptions: {} });
      }
    };
    loadLevels();
  }, [uiLanguage]);

  // ===== ADMIN PANEL DATA FETCHING & STATE =====
  const [users, setUsers] = useState([]);
  const [usersLoading, setUsersLoading] = useState(false);
  const [usersError, setUsersError] = useState(null);

  const [newUserUsername, setNewUserUsername] = useState('');
  const [newUserPassword, setNewUserPassword] = useState('');
  const [newUserRole, setNewUserRole] = useState('utilisateur');
  const [registerLoading, setRegisterLoading] = useState(false);
  const [registerError, setRegisterError] = useState(null);
  const [registerSuccess, setRegisterSuccess] = useState(null);

  const [companies, setCompanies] = useState([]);
  const [companiesLoading, setCompaniesLoading] = useState(false);
  const [companiesError, setCompaniesError] = useState(null);
  const [companiesPage, setCompaniesPage] = useState(1);
  const [companiesSearch, setCompaniesSearch] = useState('');
  const [companiesFilterStrategy, setCompaniesFilterStrategy] = useState('all');
  const [companiesFilterCluster, setCompaniesFilterCluster] = useState('all');

  const [statsData, setStatsData] = useState(null);
  const [statsLoading, setStatsLoading] = useState(false);
  const [statsError, setStatsError] = useState(null);

  const fetchUsers = async () => {
    setUsersLoading(true);
    setUsersError(null);
    try {
      const response = await fetch(`${API_BASE}/api/auth/users`, {
        headers: {
          Authorization: 'Bearer ' + localStorage.getItem('lss_token')
        }
      });
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.error || 'Failed to fetch users');
      }
      const data = await response.json();
      setUsers(data);
    } catch (err) {
      setUsersError(err.message);
    } finally {
      setUsersLoading(false);
    }
  };

  const fetchCompanies = async () => {
    setCompaniesLoading(true);
    setCompaniesError(null);
    try {
      const response = await fetch(`${API_BASE}/api/companies`, {
        headers: {
          Authorization: 'Bearer ' + localStorage.getItem('lss_token')
        }
      });
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.error || 'Failed to fetch companies');
      }
      const data = await response.json();
      setCompanies(data);
    } catch (err) {
      setCompaniesError(err.message);
    } finally {
      setCompaniesLoading(false);
    }
  };

  const fetchStats = async () => {
    setStatsLoading(true);
    setStatsError(null);
    try {
      const response = await fetch(`${API_BASE}/api/model-stats`, {
        headers: {
          Authorization: 'Bearer ' + localStorage.getItem('lss_token')
        }
      });
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.error || 'Failed to fetch stats');
      }
      const data = await response.json();
      setStatsData(data);
    } catch (err) {
      setStatsError(err.message);
    } finally {
      setStatsLoading(false);
    }
  };

  useEffect(() => {
    if (adminPanel === 'users') {
      fetchUsers();
    } else if (adminPanel === 'companies') {
      fetchCompanies();
    } else if (adminPanel === 'stats') {
      fetchStats();
    }
  }, [adminPanel]);

  const handleRegisterUser = async (e) => {
    e.preventDefault();
    if (!newUserUsername.trim() || !newUserPassword) {
      setRegisterError('Please fill in all fields');
      return;
    }
    setRegisterLoading(true);
    setRegisterError(null);
    setRegisterSuccess(null);
    try {
      const response = await fetch(`${API_BASE}/api/auth/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: 'Bearer ' + localStorage.getItem('lss_token')
        },
        body: JSON.stringify({
          username: newUserUsername.trim(),
          password: newUserPassword,
          role: newUserRole
        })
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Registration failed');
      }
      setRegisterSuccess(`User "${newUserUsername}" registered successfully.`);
      setNewUserUsername('');
      setNewUserPassword('');
      setNewUserRole('utilisateur');
      fetchUsers();
    } catch (err) {
      setRegisterError(err.message);
    } finally {
      setRegisterLoading(false);
    }
  };

  const handleDeleteUser = async (userId, username) => {
    if (!window.confirm(`Are you sure you want to delete user "${username}"?`)) {
      return;
    }
    try {
      const response = await fetch(`${API_BASE}/api/auth/users/${userId}`, {
        method: 'DELETE',
        headers: {
          Authorization: 'Bearer ' + localStorage.getItem('lss_token')
        }
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Failed to delete user');
      }
      fetchUsers();
    } catch (err) {
      alert(err.message);
    }
  };

  const detectLanguage = (text) => {
    if (/[\u0600-\u06FF]/.test(text)) return 'Arabic';
    const lower = text.toLowerCase();
    const frenchMarkers = [
      'bonjour',
      'merci',
      's il',
      's il vous plait',
      'pourquoi',
      'comment',
      'avec',
      'dans',
      'amelior',
      'processus',
      'qualite'
    ];
    if (frenchMarkers.some((marker) => lower.includes(marker))) return 'French';
    return 'English';
  };

  const updateModelConfig = (updates, options = {}) => {
    setModelConfig((prev) => ({ ...prev, ...updates }));
    if (!options.keepApiTestState) {
      setApiTestState((prev) => (prev.status === 'idle'
        ? prev
        : { status: 'idle', message: '', details: null }));
    }
  };

  const buildModelConfigPayload = () => {
    const localModel = modelConfig.localModel.trim() || 'llama3:8b';
    const apiModel = modelConfig.apiModel.trim() || 'gemini-pro';
    const apiKey = modelConfig.apiKey.trim();

    const payload = {
      local_model: localModel,
      api_model: apiModel,
      use_api: modelConfig.useApi
    };

    if (apiKey) {
      payload.api_key = apiKey;
    }

    return payload;
  };

  const runApiTest = async () => {
    if (apiTestState.status === 'loading') return;
    setApiTestState({ status: 'loading', message: 'Testing API model...', details: null });

    try {
      const modelConfigPayload = buildModelConfigPayload();
      const response = await fetch(`${API_BASE}/llm-test`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          modelConfig: {
            ...modelConfigPayload,
            use_api: true
          }
        })
      });

      const rawText = await response.text();
      let data = null;
      try {
        data = rawText ? JSON.parse(rawText) : null;
      } catch (parseError) {
        const preview = rawText.trim().slice(0, 180);
        const hint = preview ? `Non-JSON response from ${API_BASE}/llm-test: ${preview}` : 'Empty response from backend.';
        throw new Error(hint);
      }

      if (!response.ok) {
        const errorMessage = data?.message || `Request failed (${response.status}).`;
        throw new Error(errorMessage);
      }

      const status = data?.status || (data?.ok ? 'api' : 'error');
      const details = data?.model_info || null;
      const usedModel = details?.used_model || '';
      const errorDetail = details?.error || '';
      let message = data?.message || '';

      if (status === 'api') {
        message = `API OK using ${usedModel || modelConfig.apiModel || 'external model'}.`;
      } else if (status === 'fallback') {
        message = errorDetail
          ? `API failed: ${errorDetail}. Local fallback used (${usedModel || modelConfig.localModel || 'local model'}).`
          : `API failed; local fallback used (${usedModel || modelConfig.localModel || 'local model'}).`;
        updateModelConfig({ useApi: false }, { keepApiTestState: true });
      } else if (status === 'error') {
        message = errorDetail ? `API error: ${errorDetail}` : 'API test failed.';
        updateModelConfig({ useApi: false }, { keepApiTestState: true });
      } else if (!message) {
        message = 'API test failed.';
      }

      setApiTestState({ status, message, details });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setApiTestState({ status: 'error', message: errorMessage, details: null });
    }
  };

  const formatScoreList = (prefix, values) => {
    if (!Array.isArray(values)) return 'N/A';
    const items = values
      .map((value, index) => {
        if (value === '' || value === null || value === undefined) return null;
        const parsed = Number(value);
        if (!Number.isFinite(parsed)) return null;
        const level = Math.round(parsed);
        if (level < 1 || level > 5) return null;
        return { key: `${prefix}${index + 1}`, value: level };
      })
      .filter(Boolean);
    if (!items.length) return 'N/A';
    return items.map(({ key, value }) => `${key}=${value}`).join(', ');
  };

  const buildScoreMap = (params) => {
    const scores = {};
    const addScores = (prefix, values) => {
      if (!Array.isArray(values)) return;
      values.forEach((value, index) => {
        if (value === '' || value === null || value === undefined) return;
        const parsed = Number(value);
        if (!Number.isFinite(parsed)) return;
        const level = Math.round(parsed);
        if (level < 1 || level > 5) return;
        scores[`${prefix}${index + 1}`] = level;
      });
    };

    addScores('IL', params?.IL || []);
    addScores('IS', params?.IS || []);
    addScores('M', params?.IM || []);
    return scores;
  };

  const buildScoreContext = (params) => {
    const scores = buildScoreMap(params);
    const order = [
      ...Array.from({ length: 7 }, (_, i) => `IL${i + 1}`),
      ...Array.from({ length: 7 }, (_, i) => `IS${i + 1}`),
      ...Array.from({ length: 7 }, (_, i) => `M${i + 1}`)
    ];
    const items = order
      .filter((key) => Number.isFinite(scores[key]))
      .map((key) => `${key}=${scores[key]}`);
    if (!items.length) return '';
    const label = uiLanguage === 'fr' ? 'Scores CSF actuels' : 'Current CSF scores';
    return `${label}: ${items.join(', ')}`;
  };

  const buildExplanationPayload = (params, result) => {
    const analysis = result?.parameter_analysis || {
      IL_average: toAverage(params?.IL || []),
      IS_average: toAverage(params?.IS || []),
      IM_average: toAverage(params?.IM || [])
    };

    return {
      language: uiLanguage === 'fr' ? 'French' : 'English',
      scores: buildScoreMap(params),
      prediction: result?.prediction ?? null,
      confidence: result?.confidence ?? null,
      strategy: analysis?.strategy || result?.parameter_analysis?.strategy || null,
      recommendations: result?.recommendations || [],
      averages: {
        IL: analysis.IL_average,
        IS: analysis.IS_average,
        IM: analysis.IM_average
      },
      chart: result?.chart || null,
      performance_table: result?.performance_table || null
    };
  };

  const requestExplanation = async (params, result) => {
    const payload = buildExplanationPayload(params, result);
    const explainMessage = `__EXPLAIN_RESULT__\n${JSON.stringify(payload)}`;
    const response = await fetch(`${API_BASE}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: explainMessage,
        conversationHistory: [],
        modelConfig: buildModelConfigPayload()
      })
    });

    let data = null;
    try {
      data = await response.json();
    } catch (parseError) {
      throw new Error('Invalid JSON response from backend.');
    }

    if (!response.ok) {
      const errorMessage = data?.error || `Request failed (${response.status}).`;
      throw new Error(errorMessage);
    }

    const answer = typeof data?.answer === 'string' ? data.answer.trim() : '';
    if (!answer) {
      throw new Error('Empty response from backend.');
    }

    return answer;
  };

  const startNewConversation = () => {
    setMessages(getIntroMessages());
    setInput('');
    setCalcResult(null);
    setActiveTab('chat');
    setAssistedAssessment({
      active: false,
      stepIndex: 0,
      parameters: createEmptyParameters(),
      review: false
    });
  };

  const submitCalculation = async (paramSet, addToChat = true) => {
    setCalcExplanation('');
    setCalcExplanationLoading(false);
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/calculate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ parameters: paramSet, language: uiLanguage })
      });

      let data = null;
      try {
        data = await response.json();
      } catch (parseError) {
        throw new Error('Invalid JSON response from backend.');
      }
      if (!response.ok) {
        const errorMessage = data?.error || data?.detail || `Request failed (${response.status}).`;
        throw new Error(errorMessage);
      }
      setCalcResult(data);
      setLastCalcParams(paramSet);

      if (addToChat) {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `Analysis Complete!\n\nPredicted Performance: ${data.prediction?.toFixed(2) || 'N/A'}%\nConfidence: ${data.confidence || 'N/A'}\n\nRecommendations:\n${data.recommendations?.map((r, i) => `${i + 1}. ${r}`).join('\n') || 'Process optimization suggestions based on your parameters.'}`,
          type: 'calculation',
          chart: data.chart,
          performance_table: data.performance_table,
          calcSnapshot: {
            parameters: paramSet,
            result: data
          }
        }]);
      }

      return data;
    } catch (error) {
      const fallbackAnalysis = {
        IL_average: toAverage(paramSet.IL),
        IS_average: toAverage(paramSet.IS),
        IM_average: toAverage(paramSet.IM)
      };
      const fallback = {
        prediction: 'Demo Mode',
        confidence: 'High',
        recommendations: [
          'Focus on improving Implementation parameters (IM)',
          'Strengthen Strategic alignment (IS)',
          'Enhance Leadership support (IL)',
          'Consider cross-functional team training'
        ],
        parameter_analysis: fallbackAnalysis
      };
      setCalcResult(fallback);
      setLastCalcParams(paramSet);

      if (addToChat) {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: 'Demo Analysis Complete! Connect to your backend for real predictions.',
          type: 'calculation',
          calcSnapshot: {
            parameters: paramSet,
            result: fallback
          }
        }]);
      }

      return fallback;
    } finally {
      setLoading(false);
    }
  };

  const getAssistedStepLabel = (step) => {
    return csfLevels.labels?.[step.code] || step.label;
  };

  const getAssistedChoices = (step) => {
    const levelMap = csfLevels.levels?.[step.code] || {};
    return DEFAULT_LEVEL_CHOICES.map((choice) => ({
      ...choice,
      detail: levelMap[choice.level] || choice.detail
    }));
  };

  const updateAssistedReviewScore = (group, index, value) => {
    const parsed = Number(value);
    if (!Number.isInteger(parsed) || parsed < 1 || parsed > 5) {
      return;
    }
    setAssistedAssessment((prev) => ({
      ...prev,
      parameters: {
        ...prev.parameters,
        [group]: prev.parameters[group].map((v, i) => (i === index ? String(parsed) : v))
      }
    }));
    setParameters((prev) => ({
      ...prev,
      [group]: prev[group].map((v, i) => (i === index ? String(parsed) : v))
    }));
  };

  const confirmAssistedAssessment = async () => {
    if (loading || !reviewReady) return;
    const updatedParams = assistedAssessment.parameters;
    setAssistedAssessment({
      active: false,
      stepIndex: 0,
      parameters: updatedParams,
      review: false
    });
    setParameters(updatedParams);
    setMessages(prev => [...prev, {
      role: 'assistant',
      content: 'Thanks! I am calculating your predicted performance now.'
    }]);
    await submitCalculation(updatedParams, true);
    setActiveTab('chat');
  };

  const startAssistedAssessment = () => {
    if (loading) return;
    const emptyParams = createEmptyParameters();
    setAssistedAssessment({
      active: true,
      stepIndex: 0,
      parameters: emptyParams,
      review: false
    });
    setMessages(prev => [...prev, {
      role: 'assistant',
      content: `Guided assessment started. Choose the best option for each level. You can also type a number from 1 to 5. Type "cancel" to stop.\n\n${buildAssistedPrompt(0)}`
    }]);
  };

  const cancelAssistedAssessment = () => {
    setAssistedAssessment({
      active: false,
      stepIndex: 0,
      parameters: createEmptyParameters(),
      review: false
    });
    setMessages(prev => [...prev, {
      role: 'assistant',
      content: 'Guided assessment canceled. You can start again anytime.'
    }]);
  };

  const applyAssistedScore = async (score) => {
    const currentStep = ASSISTED_STEPS[assistedAssessment.stepIndex];
    const updatedParams = {
      ...assistedAssessment.parameters,
      [currentStep.group]: assistedAssessment.parameters[currentStep.group].map((value, index) =>
        index === currentStep.index ? String(score) : value
      )
    };

    const nextStepIndex = assistedAssessment.stepIndex + 1;
    if (nextStepIndex < ASSISTED_STEPS.length) {
      setAssistedAssessment({
        active: true,
        stepIndex: nextStepIndex,
        parameters: updatedParams,
        review: false
      });
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: buildAssistedPrompt(nextStepIndex)
      }]);
      return;
    }

    setAssistedAssessment({
      active: false,
      stepIndex: 0,
      parameters: updatedParams,
      review: true
    });
    setParameters(updatedParams);
    setMessages(prev => [...prev, {
      role: 'assistant',
      content: 'Review your answers below. You can edit any score before we calculate the results.'
    }]);
  };

  const handleAssistedInput = async (text) => {
    const normalized = text.trim().toLowerCase();
    if (['cancel', 'stop', 'exit'].includes(normalized)) {
      cancelAssistedAssessment();
      return;
    }

    const currentStep = ASSISTED_STEPS[assistedAssessment.stepIndex];
    const parsed = Number(normalized);
    if (!Number.isInteger(parsed) || parsed < 1 || parsed > 5) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Please enter a whole number between 1 and 5 for ${currentStep.code} - ${getAssistedStepLabel(currentStep)}.`
      }]);
      return;
    }

    await applyAssistedScore(parsed);
  };

  const handleAssistedChoice = async (score) => {
    const step = ASSISTED_STEPS[assistedAssessment.stepIndex];
    const label = getAssistedStepLabel(step);
    setMessages(prev => [...prev, {
      role: 'user',
      content: `${step.code} = ${score} (${label})`
    }]);
    await applyAssistedScore(score);
  };

  const sendUserMessage = async (rawText, options = {}) => {
    const { clearInput = true } = options;
    const messageText = typeof rawText === 'string' ? rawText.trim() : '';
    if (!messageText) return;

    if (assistedAssessment.active) {
      setMessages(prev => [...prev, { role: 'user', content: messageText }]);
      if (clearInput) {
        setInput('');
      }
      await handleAssistedInput(messageText);
      return;
    }

    const detectedLanguage = uiLanguage === 'fr' ? 'fr' : 'en';

    const userMessage = { role: 'user', content: messageText };
    setMessages(prev => [...prev, userMessage]);
    if (clearInput) {
      setInput('');
    }
    setLoading(true);
    setLoadingMessage('Analyzing your question...');

    try {
      const historyPayload = [...messages, userMessage]
        .slice(-10)
        .map(({ role, content }) => ({ role, content }));
      const modelConfigPayload = buildModelConfigPayload();
      const scoreContext = assistedAssessment.review || assistedAssessment.active
        ? buildScoreContext(assistedAssessment.parameters)
        : buildScoreContext(lastCalcParams) || buildScoreContext(parameters);
      const queryPayload = scoreContext ? `${messageText}

${scoreContext}` : messageText;
      const response = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: queryPayload,
          language: detectedLanguage,
          conversationHistory: historyPayload,
          modelConfig: modelConfigPayload
        })
      });

      let data = null;
      try {
        data = await response.json();
      } catch (parseError) {
        throw new Error('Invalid JSON response from backend.');
      }

      if (!response.ok) {
        const errorMessage = data?.error || `Request failed (${response.status}).`;
        throw new Error(errorMessage);
      }

      const answer = typeof data?.answer === 'string' ? data.answer.trim() : '';
      if (!answer) {
        throw new Error('Empty response from backend.');
      }

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: answer,
        sources: data.sources || [],
        graph_context: data.graph_context
      }]);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Backend error: ${errorMessage} Please ensure your backend API is running at ${API_BASE}.`
      }]);
    } finally {
      setLoading(false);
      setLoadingMessage('');
    }
  };

  const handleSendMessage = async () => {
    await sendUserMessage(input);
  };

  const getHistoryMeta = (content) => {
    const text = String(content || '').trim();
    const scoreMatch = text.match(/^((IL|IS|IM)\d+)\s*=\s*([1-5])\b/i);
    if (!scoreMatch) {
      return {
        kind: 'prompt',
        tone: 'tone-neutral',
        title: text || 'Message',
        hint: 'Tap to reuse in chat',
        tags: ['Prompt']
      };
    }

    const code = scoreMatch[1].toUpperCase();
    const level = Number(scoreMatch[3]);
    const label = csfLevels.labels?.[code] || ASSISTED_LABELS[code];
    return {
      kind: 'score',
      tone: getFactorToneClass(code),
      title: label || text,
      hint: `Level ${level} score captured`,
      tags: ['Score', code, `L${level}`]
    };
  };

  const handleHistoryInsert = (content) => {
    const text = String(content || '');
    if (!text.trim()) return;
    setActiveTab('chat');
    setInput(text);
    setTimeout(() => {
      inputRef.current?.focus();
    }, 0);
  };

  const handleHistorySend = (content) => {
    const text = String(content || '');
    if (!text.trim()) return;
    setActiveTab('chat');
    void sendUserMessage(text, { clearInput: false });
  };

  const updatePanelsCollapsed = (next) => {
    panelsCollapsedRef.current = next;
    setPanelsCollapsed(next);
  };

  const handleMessagesScroll = (event) => {
    if (!autoCollapseEnabled) return;
    const scrollTop = event.currentTarget.scrollTop;
    const current = panelsCollapsedRef.current;
    if (!current && scrollTop > 64) {
      updatePanelsCollapsed(true);
    } else if (current && scrollTop < 20) {
      updatePanelsCollapsed(false);
    }
  };

  const handleExplainCalculation = async (params, result, messageIndex = null) => {
    if (!result) return;

    if (messageIndex !== null) {
      setExplainLoadingId(messageIndex);
    } else {
      setCalcExplanationLoading(true);
      setCalcExplanation('');
    }

    try {
      const explanation = await requestExplanation(params, result);
      if (messageIndex !== null) {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: explanation,
          type: 'explanation'
        }]);
      } else {
        setCalcExplanation(explanation);
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      if (messageIndex !== null) {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `Explanation error: ${errorMessage}`
        }]);
      } else {
        setCalcExplanation(`Explanation error: ${errorMessage}`);
      }
    } finally {
      if (messageIndex !== null) {
        setExplainLoadingId(null);
      } else {
        setCalcExplanationLoading(false);
      }
    }
  };

  const handleParameterChange = (category, index, value) => {
    if (value === '') {
      setParameters(prev => ({
        ...prev,
        [category]: prev[category].map((v, i) => i === index ? value : v)
      }));
      return;
    }

    const parsed = Number(value);
    if (!Number.isInteger(parsed) || parsed < 1 || parsed > 5) {
      return;
    }

    setParameters(prev => ({
      ...prev,
      [category]: prev[category].map((v, i) => i === index ? String(parsed) : v)
    }));
  };

  const calculatePredictions = async () => {
    await submitCalculation(parameters, true);
    setActiveTab('chat');
  };

  const isGuidedFullScreen = assistedAssessment.active || assistedAssessment.review;
  const guidedPanel = (
    <div className={`panel assistant-card p-4 guided-panel${isGuidedFullScreen ? ' is-fullscreen' : ''}`}>
      <div className="guided-panel__content">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="section-label">Guided Assessment</div>
            <div className="mt-1 text-base font-semibold">
              Step by step scoring for IL, IS, and IM
            </div>
            <div className="mt-1 text-sm text-muted">
              The assistant will ask for IL, IS, and IM scores using levels 1-5, then calculate your performance.
            </div>
          </div>
          <div className="flex items-center gap-2">
            {assistedAssessment.active ? (
              <>
                <div className="chip">
                  In progress: {assistedAssessment.stepIndex + 1}/{ASSISTED_STEPS.length}
                </div>
                <button
                  type="button"
                  onClick={cancelAssistedAssessment}
                  className="button-secondary"
                >
                  Cancel
                </button>
              </>
            ) : assistedAssessment.review ? (
              <>
                <div className="chip">Review pending</div>
                <button
                  type="button"
                  onClick={cancelAssistedAssessment}
                  className="button-secondary"
                >
                  Clear
                </button>
              </>
            ) : (
              <button
                type="button"
                onClick={startAssistedAssessment}
                disabled={loading}
                className="button-primary"
              >
                Start Guided Assessment
              </button>
            )}
          </div>
        </div>
        {assistedAssessment.active && (
          <div className="panel panel--soft mt-4 p-4">
            {(() => {
              const step = ASSISTED_STEPS[assistedAssessment.stepIndex];
              const choices = getAssistedChoices(step);
              return (
                <div>
                  <div className="section-label">{step.code}</div>
                  <div className="mt-1 text-lg font-semibold">{getAssistedStepLabel(step)}</div>
                  <div className="mt-1 text-sm text-muted">
                    Choose the level that best matches your current situation.
                  </div>
                  <div className="mt-4 grid grid-cols-1 gap-3">
                    {choices.map((choice) => (
                      <button
                        key={`${step.code}-${choice.level}`}
                        type="button"
                        onClick={() => handleAssistedChoice(choice.level)}
                        className="choice-button"
                      >
                        <div className="flex items-center justify-between">
                          <div className="text-sm font-semibold choice-title">
                            Level {choice.level} - {choice.title}
                          </div>
                          <div className="text-xs text-muted">
                            Select
                          </div>
                        </div>
                        <div className="mt-2 text-sm text-muted">
                          {choice.detail}
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              );
            })()}
          </div>
        )}
        {assistedAssessment.review && (
          <div className="panel panel--soft mt-4 p-4">
            <div className="section-label">Review Answers</div>
            <div className="mt-1 text-sm text-muted">
              Adjust any score before calculation.
            </div>
            <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-3">
              {[
                { key: 'IL', title: 'Implementation of Lean (IL)', tone: 'text-accent' },
                { key: 'IS', title: 'Implementation of Six Sigma (IS)', tone: 'text-accent-2' },
                { key: 'IM', title: 'Maturity Levels (IM)', tone: 'text-accent-3' }
              ].map((section) => (
                <div key={section.key} className="space-y-3">
                  <div className={`text-sm font-semibold ${section.tone}`}>{section.title}</div>
                  {Array.from({ length: 7 }).map((_, idx) => {
                    const code = `${section.key}${idx + 1}`;
                    const label = csfLevels.labels?.[code] || code;
                    const value = assistedAssessment.parameters?.[section.key]?.[idx] || '';
                    return (
                      <div key={code} className="flex items-center justify-between gap-3">
                        <div>
                          <div className="text-xs font-semibold">{code}</div>
                          <div className="text-xs text-muted">{label}</div>
                        </div>
                        <select
                          value={value}
                          onChange={(e) => updateAssistedReviewScore(section.key, idx, e.target.value)}
                          className="input-field input-field--compact w-24"
                        >
                          <option value="" disabled>
                            --
                          </option>
                          {[1, 2, 3, 4, 5].map((level) => (
                            <option key={level} value={level}>
                              Level {level}
                            </option>
                          ))}
                        </select>
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
            <div className="mt-4 flex flex-wrap gap-2">
              <button
                type="button"
                onClick={confirmAssistedAssessment}
                disabled={!reviewReady || loading}
                className="button-primary"
              >
                Confirm & Calculate
              </button>
              <button
                type="button"
                onClick={startAssistedAssessment}
                disabled={loading}
                className="button-secondary"
              >
                Restart Guided Assessment
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="app-shell" data-theme={themeMode}>
      <div className="ambient-layer">
        <span className="ambient-blob ambient-blob--a" />
        <span className="ambient-blob ambient-blob--b" />
        <span className="ambient-blob ambient-blob--c" />
        <span className="ambient-grid" />
      </div>
      <div className="app-frame">
        {isGuidedFullScreen && (
          <div className="guided-overlay">
            {guidedPanel}
          </div>
        )}
        {showIntro && (
          <div className="intro-screen">
            <div className="intro-orb intro-orb--a" />
            <div className="intro-orb intro-orb--b" />
            <div className="intro-orb intro-orb--c" />
            <div className="intro-logo text-center">
              <div className="intro-kicker">AI-Driven Lean Six Sigma</div>
              <div className="intro-title">Performance Engine</div>
              <div className="intro-subtitle">IA-Guided industrial performance management</div>
              <div className="intro-dots">
                <span className="intro-dot" />
                <span className="intro-dot" />
                <span className="intro-dot" />
              </div>
            </div>
          </div>
        )}

        <header className="topbar panel">
          <div className="max-w-7xl mx-auto px-4 py-3">
            <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
              <div className="flex items-center gap-4">
                <div className="brand-mark">
                  <TrendingUp className="w-6 h-6" />
                </div>
                <div>
                  <div className="brand-kicker">AI-Driven Lean Six Sigma</div>
                  <h1 className="brand-title">Operational Excellence Performance Engine</h1>
                  <p className="brand-subtitle">
                    Industrial engine to drive, secure, and succeed in Operational Excellence initiatives
                  </p>
                  <div className="brand-chips">
                    <span className="chip">GraphRAG</span>
                    <span className="chip">Guided</span>
                    <span className="chip">Predictive</span>
                  </div>
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-3" style={{ position: 'relative', zIndex: 9999 }}>
                <div className="mode-switch">
                  <button type="button" onClick={() => setActiveTab('chat')} aria-pressed={activeTab === 'chat'} className={`mode-button ${activeTab === 'chat' ? 'is-active' : ''}`}>
                    <MessageSquare className="w-4 h-4" /> Chat
                  </button>
                  <button type="button" onClick={() => setActiveTab('calculator')} aria-pressed={activeTab === 'calculator'} className={`mode-button ${activeTab === 'calculator' ? 'is-active' : ''}`}>
                    <Calculator className="w-4 h-4" /> Calculator
                  </button>
                </div>
                <button type="button" onClick={() => setThemeMode(isDark ? 'light' : 'dark')} className="button-ghost">
                  {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
                  {isDark ? 'Light' : 'Dark'}
                </button>

                {currentUser && (
                  <button
                    type="button"
                    onClick={() => setUserMenuOpen(true)}
                    className="button-ghost"
                    style={{
                      display: 'flex', alignItems: 'center', gap: '8px',
                      padding: '6px 12px', borderRadius: '20px',
                      background: isAdmin
                        ? 'linear-gradient(135deg, rgba(239,68,68,0.15), rgba(168,85,247,0.15))'
                        : 'linear-gradient(135deg, rgba(59,130,246,0.15), rgba(14,165,233,0.15))',
                      border: '1px solid',
                      borderColor: isAdmin ? 'rgba(239,68,68,0.3)' : 'rgba(59,130,246,0.3)',
                      cursor: 'pointer'
                    }}
                  >
                    {isAdmin ? <Shield className="w-4 h-4" /> : <User className="w-4 h-4" />}
                    <span style={{ fontSize: '13px', fontWeight: 600 }}>{currentUser.username}</span>
                    <span style={{
                      fontSize: '10px', padding: '2px 8px', borderRadius: '10px',
                      fontWeight: 700, background: isAdmin ? '#ef4444' : '#3b82f6',
                      color: 'white', textTransform: 'uppercase'
                    }}>
                      {isAdmin ? 'Admin' : 'User'}
                    </span>
                  </button>
                )}
              </div>
            </div>
          </div>
        </header>
        {/* Main Content */}
        <div className="flex-1 overflow-hidden min-h-0">
        <div className="flex h-full min-h-0 gap-4 px-4 pb-6 pt-3">
          <aside className="sidebar panel hidden md:flex md:w-72 flex-col min-h-0">
            <div className="px-4 py-4 border-b panel-divider">
              <div className="section-label">Logbook</div>
              <div className="mt-2 text-sm font-semibold">Conversation History</div>
              <button
                type="button"
                onClick={startNewConversation}
                className="button-primary mt-4 w-full"
              >
                New Conversation
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-3 space-y-2 min-h-0">
              {historyItems.length === 0 ? (
                <div className="text-sm text-muted">No messages yet.</div>
              ) : (
                historyItems
                  .slice()
                  .reverse()
                  .map((msg, idx) => {
                    const meta = getHistoryMeta(msg.content);
                    return (
                      <div key={`${msg.content}-${idx}`} className={`history-item ${meta.tone}`}>
                        <button
                          type="button"
                          onClick={() => handleHistoryInsert(msg.content)}
                          className="history-main"
                          aria-label={`Insert ${meta.title}`}
                        >
                          <div className="history-meta">
                            {meta.tags.map((tag) => (
                              <span key={tag} className="history-pill">{tag}</span>
                            ))}
                          </div>
                          <div className="history-title truncate" dir="auto" title={meta.title}>
                            {meta.title}
                          </div>
                          <div className="history-snippet truncate" dir="auto">
                            {meta.hint}
                          </div>
                        </button>
                        <div className="history-actions">
                          <button
                            type="button"
                            onClick={() => handleHistorySend(msg.content)}
                            className="history-action"
                            aria-label="Send again"
                          >
                            <Send className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    );
                  })
              )}
            </div>
          </aside>

          <div className="flex-1 overflow-hidden min-h-0">
            {activeTab === 'chat' ? (
              <div className="flex flex-col h-full max-w-4xl mx-auto min-h-0">
                <div className="px-4 pt-4">
                  <div className={`chat-panels ${panelsCollapsed ? 'is-collapsed' : ''}`}>
                    <div className="chat-panels__peek panel">
                      <div className="chat-panels__peek-inner">
                        <div>
                          <div className="section-label">Quick Panels</div>
                          <div className="chat-panels__peek-meta">
                            <span className="history-pill tone-amber">LLM: {activeModelLabel}</span>
                            <span className="history-pill tone-sky">Guided: {guidedStatus}</span>
                          </div>
                        </div>
                        <button
                          type="button"
                          onClick={() => updatePanelsCollapsed(false)}
                          className="button-secondary"
                        >
                          Show
                        </button>
                      </div>
                    </div>
                    <div className="chat-panels__stack">
                      <div className="panel p-4">
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div>
                        <div className="section-label">Model Settings</div>
                        <div className="mt-1 text-base font-semibold">LLM configuration</div>
                        <div className="mt-1 text-sm text-muted">
                          Active: {activeModelLabel}
                        </div>
                      </div>
                      <div className="flex flex-wrap items-center gap-2">
                        <button
                          type="button"
                          onClick={() => setShowModelSettings((prev) => !prev)}
                          className="button-secondary"
                        >
                          {showModelSettings ? 'Hide' : 'Configure'}
                        </button>
                        <button
                          type="button"
                          onClick={() => updatePanelsCollapsed(true)}
                          className="button-ghost"
                        >
                          Minimize
                        </button>
                      </div>
                    </div>
                    {showModelSettings && (
                      <div className="mt-4 grid gap-4 md:grid-cols-2">
                        <div className="space-y-2">
                          <div className="section-label">Mode</div>
                          <div className="mode-switch">
                            <button
                              type="button"
                              onClick={() => updateModelConfig({ useApi: false })}
                              className={`mode-button ${modelConfig.useApi ? '' : 'is-active'}`}
                            >
                              Local
                            </button>
                            <button
                              type="button"
                              onClick={() => updateModelConfig({ useApi: true })}
                              className={`mode-button ${modelConfig.useApi ? 'is-active' : ''}`}
                            >
                              API
                            </button>
                          </div>
                          <div className="text-xs text-muted">
                            Local uses Ollama; API uses your provided key.
                          </div>
                        </div>
                        <div className="space-y-2">
                          <label className="section-label" htmlFor="local-model">
                            Local model
                          </label>
                          <input
                            id="local-model"
                            type="text"
                            value={modelConfig.localModel}
                            onChange={(e) => updateModelConfig({ localModel: e.target.value })}
                            placeholder="llama3:8b"
                            list="local-model-options"
                            className="input-field input-field--compact"
                          />
                          <datalist id="local-model-options">
                            {LOCAL_MODEL_OPTIONS.map((model) => (
                              <option key={model} value={model} />
                            ))}
                          </datalist>
                          <div className="text-xs text-muted">
                            Choose from suggestions or enter a custom Ollama model.
                          </div>
                        </div>
                        {modelConfig.useApi && (
                          <div className="space-y-2">
                            <label className="section-label" htmlFor="api-model">
                              API model
                            </label>
                            <input
                              id="api-model"
                              type="text"
                              value={modelConfig.apiModel}
                              onChange={(e) => updateModelConfig({ apiModel: e.target.value })}
                              placeholder="gemini-pro"
                              className="input-field input-field--compact"
                            />
                            <div className="text-xs text-muted">
                              Pick from the catalog below or enter a custom API model.
                            </div>
                            <div className="model-catalog">
                              {API_MODEL_CATALOG.map((group) => (
                                <div key={group.provider} className="model-catalog__group">
                                  <div className="model-catalog__provider">{group.provider}</div>
                                  <div className="model-catalog__families">
                                    {group.families.map((family) => (
                                      <div key={`${group.provider}-${family.family}`} className="model-catalog__family">
                                        <div className="model-catalog__family-name">{family.family}</div>
                                        <div className="model-catalog__options">
                                          {family.models.map((model) => {
                                            const isActive = apiModelSelected === model.toLowerCase();
                                            return (
                                              <button
                                                key={`${group.provider}-${family.family}-${model}`}
                                                type="button"
                                                onClick={() => updateModelConfig({ apiModel: model })}
                                                aria-pressed={isActive}
                                                className={`model-chip${isActive ? ' is-active' : ''}`}
                                              >
                                                {model}
                                              </button>
                                            );
                                          })}
                                        </div>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                        {modelConfig.useApi && (
                          <div className="space-y-2">
                            <label className="section-label" htmlFor="api-key">
                              API key
                            </label>
                            <input
                              id="api-key"
                              type="password"
                              value={modelConfig.apiKey}
                              onChange={(e) => updateModelConfig({ apiKey: e.target.value })}
                              placeholder="Paste API key"
                              className="input-field input-field--compact"
                            />
                            <div className="text-xs text-muted">
                              Key stays in this browser session. Leave blank to use server env.
                            </div>
                            <div className="flex flex-wrap items-center gap-2">
                              <button
                                type="button"
                                onClick={runApiTest}
                                disabled={apiTestState.status === 'loading'}
                                className="button-secondary"
                              >
                                Test API
                              </button>
                              {apiTestBadge && (
                                <div className={`${apiTestBadge.className} inline-flex items-center gap-2`}>
                                  {apiTestState.status === 'loading' && (
                                    <Loader2 className="w-3 h-3 animate-spin" />
                                  )}
                                  <span>{apiTestBadge.label}</span>
                                </div>
                              )}
                            </div>
                            {apiTestState.message && (
                              <div className="text-xs text-muted">
                                {apiTestState.message}
                              </div>
                            )}
                            {apiTestState.details && (
                              <div className="text-xs text-muted">
                                Used: {apiTestState.details.used || 'none'}
                                {apiTestState.details.used_model ? ` (${apiTestState.details.used_model})` : ''}
                                {apiTestState.details.active_model ? ` | Active: ${apiTestState.details.active_model}` : ''}
                              </div>
                            )}
                            {apiTestState.details?.sample && (
                              <div className="text-xs text-muted">
                                Sample: "{apiTestState.details.sample}"
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                  {!isGuidedFullScreen && guidedPanel}
                </div>
                </div>
              </div>

                {/* Messages */}
                <div
                  ref={messagesContainerRef}
                  onScroll={handleMessagesScroll}
                  className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0"
                >
                  {messages.map((msg, idx) => {
                    const chartData = msg.chart || msg.graph_context?.chart;
                    const performanceTable = msg.performance_table || msg.graph_context?.performance_table;
                    const targetActionTable = msg.target_action_table || msg.graph_context?.target_action_table;
                    return (
                      <div
                        key={idx}
                        className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                      >
                        <div
                          className={`max-w-2xl message-bubble ${
                            msg.role === 'user' ? 'message-bubble--user' : 'message-bubble--assistant'
                          }`}
                        >
                          <p className="whitespace-pre-wrap" dir="auto">{msg.content}</p>
                          {msg.sources && msg.sources.length > 0 && (
                            <div className="mt-2 pt-2 border-t panel-divider">
                              <p className="text-xs text-muted">
                                Sources: {msg.sources.join(', ')}
                              </p>
                            </div>
                          )}
                          {msg.type === 'calculation' && msg.calcSnapshot && (
                            <div className="mt-3">
                              <button
                                type="button"
                                onClick={() => handleExplainCalculation(
                                  msg.calcSnapshot.parameters,
                                  msg.calcSnapshot.result,
                                  idx
                                )}
                                disabled={explainLoadingId === idx || loading}
                                className="button-secondary"
                              >
                                {explainLoadingId === idx ? 'Explaining...' : 'Explain this result'}
                              </button>
                            </div>
                          )}
                          {msg.role !== 'user' && renderChartBlock(chartData)}
                          {msg.role !== 'user' && renderPerformanceTable(performanceTable)}
                          {msg.role !== 'user' && renderTargetActionTable(targetActionTable)}
                          {msg.role !== 'user' && msg.calcSnapshot && renderLevelUpTable(msg.calcSnapshot.parameters)}
                        </div>
                      </div>
                    );
                  })}
                  {loading && (
                    <div className="flex justify-start">
                      <div className="message-bubble message-bubble--assistant">
                        <div className="flex items-center gap-2">
                          <Loader2 className="w-5 h-5 animate-spin text-muted" />
                          <span className="text-sm text-muted">
                            {loadingMessage || 'Thinking...'}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                  <div ref={messagesEndRef} />
                </div>

                {/* Input */}
                <div className="input-bar panel">
                  <div className="flex gap-2 max-w-4xl mx-auto">
                    <input
                      ref={inputRef}
                      type="text"
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                      placeholder={
                        assistedAssessment.active
                          ? buildAssistedPrompt(assistedAssessment.stepIndex)
                          : 'Ask about DMAIC, process capability, or start a guided assessment.'
                      }
                      className="input-field input-field--chat flex-1"
                      disabled={loading}
                      dir="auto"
                    />
                    <button
                      type="button"
                      onClick={handleSendMessage}
                      disabled={loading || !input.trim()}
                      className="button-primary send-button"
                    >
                      <Send className="w-5 h-5" />
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              <div className="h-full overflow-y-auto p-6 min-h-0">
                <div
                  className={`max-w-5xl mx-auto rounded-xl shadow-xl p-6 border ${themeClass(
                    'bg-slate-900/80 border-slate-800',
                    'bg-white border-slate-200'
                  )}`}
                >
                  <h2 className={`text-2xl font-bold mb-6 ${themeClass('text-slate-100', 'text-slate-900')}`}>
                    Parameter Calculator
                  </h2>
                  
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                    {/* IL Parameters */}
                    <div className="space-y-4">
                      <h3 className="text-lg font-semibold flex items-center gap-2 text-accent">
                        <AlertCircle className="w-5 h-5" />
                        Implementation of Lean (IL)
                      </h3>
                      {[1, 2, 3, 4, 5, 6, 7].map(i => (
                        <div key={i}>
                          <label className={`block text-sm font-medium mb-1 ${themeClass('text-slate-300', 'text-slate-700')}`}>
                            IL{i}
                          </label>
                          <input
                            type="number"
                            min="1"
                            max="5"
                            step="1"
                            inputMode="numeric"
                            value={parameters.IL[i - 1]}
                            onChange={(e) => handleParameterChange('IL', i - 1, e.target.value)}
                            placeholder="1 - 5"
                            className="input-field input-field--compact"
                          />
                        </div>
                      ))}
                    </div>

                    {/* IS Parameters */}
                    <div className="space-y-4">
                      <h3 className="text-lg font-semibold flex items-center gap-2 text-accent-2">
                        <TrendingUp className="w-5 h-5" />
                        Implementation of Six Sigma (IS)
                      </h3>
                      {[1, 2, 3, 4, 5, 6, 7].map(i => (
                        <div key={i}>
                          <label className={`block text-sm font-medium mb-1 ${themeClass('text-slate-300', 'text-slate-700')}`}>
                            IS{i}
                          </label>
                          <input
                            type="number"
                            min="1"
                            max="5"
                            step="1"
                            inputMode="numeric"
                            value={parameters.IS[i - 1]}
                            onChange={(e) => handleParameterChange('IS', i - 1, e.target.value)}
                            placeholder="1 - 5"
                            className="input-field input-field--compact"
                          />
                        </div>
                      ))}
                    </div>

                    {/* IM Parameters */}
                    <div className="space-y-4">
                      <h3 className="text-lg font-semibold flex items-center gap-2 text-accent-3">
                        <Calculator className="w-5 h-5" />
                        Maturity Levels (IM)
                      </h3>
                      {[1, 2, 3, 4, 5, 6, 7].map(i => (
                        <div key={i}>
                          <label className={`block text-sm font-medium mb-1 ${themeClass('text-slate-300', 'text-slate-700')}`}>
                            IM{i}
                          </label>
                          <input
                            type="number"
                            min="1"
                            max="5"
                            step="1"
                            inputMode="numeric"
                            value={parameters.IM[i - 1]}
                            onChange={(e) => handleParameterChange('IM', i - 1, e.target.value)}
                            placeholder="1 - 5"
                            className="input-field input-field--compact"
                          />
                        </div>
                      ))}
                    </div>
                  </div>

                  <button
                    onClick={calculatePredictions}
                    disabled={loading}
                    className="w-full py-4 cta-button text-slate-900 text-lg font-semibold rounded-xl disabled:opacity-50 disabled:cursor-not-allowed transition flex items-center justify-center gap-2"
                  >
                    {loading ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        Calculating...
                      </>
                    ) : (
                      <>
                        <TrendingUp className="w-5 h-5" />
                        Calculate Predictions & Get Recommendations
                      </>
                    )}
                  </button>

                  {calcResult && (
                    <div className={`mt-8 rounded-xl border p-5 ${themeClass(
                      'border-slate-800 bg-slate-900/70',
                      'border-slate-200 bg-white'
                    )}`}>
                      <div className="flex flex-col gap-4">
                        <div>
                          <div className={`text-sm ${themeClass('text-slate-400', 'text-slate-500')}`}>Predicted Performance</div>
                          <div className={`text-3xl font-semibold ${themeClass('text-slate-100', 'text-slate-900')}`}>
                            {typeof calcResult.prediction === 'number'
                              ? `${calcResult.prediction.toFixed(1)}%`
                              : calcResult.prediction}
                          </div>
                        </div>

                        <div className={`h-3 w-full overflow-hidden rounded-full ${themeClass('bg-slate-800', 'bg-slate-200')}`}>
                          <div
                            className="h-full rounded-full bg-accent transition-all"
                            style={{
                              width: `${Math.min(
                                Math.max(Number(calcResult.prediction) || 0, 0),
                                100
                              )}%`
                            }}
                          />
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-[200px_1fr] gap-6">
                          <div className={`rounded-lg p-4 ${themeClass('bg-slate-900/80', 'bg-slate-50')}`}>
                            <div className={`text-xs uppercase tracking-wide ${themeClass('text-slate-400', 'text-slate-500')}`}>
                              Radar Summary
                            </div>
                            <div className="mt-3 flex items-center justify-center">
                              <svg width={radarSize} height={radarSize}>
                                <polygon
                                  points={radarAxisPoints}
                                  fill="none"
                                  stroke={radarColors.grid}
                                  strokeWidth="1"
                                />
                                <polygon
                                  points={radarPoints}
                                  fill={radarColors.fill}
                                  stroke={radarColors.stroke}
                                  strokeWidth="2"
                                />
                              </svg>
                            </div>
                            <div className={`mt-3 grid grid-cols-3 gap-2 text-xs ${themeClass('text-slate-400', 'text-slate-500')}`}>
                              {radarValues.map((item) => (
                                <div key={item.label} className={`text-center ${themeClass('text-slate-400', 'text-slate-500')}`}>
                                  {item.label}
                                </div>
                              ))}
                            </div>
                          </div>

                          <div className="grid grid-cols-3 gap-4">
                            {[
                            { label: 'IL', value: analysis.IL_average, color: 'bg-accent' },
                            { label: 'IS', value: analysis.IS_average, color: 'bg-accent-2' },
                            { label: 'IM', value: analysis.IM_average, color: 'bg-accent-3' }
                          ].map((item) => (
                              <div
                                key={item.label}
                                className={`rounded-lg p-3 shadow-sm ${themeClass('bg-slate-900/80', 'bg-slate-50')}`}
                              >
                                <div className={`text-xs uppercase tracking-wide ${themeClass('text-slate-400', 'text-slate-500')}`}>
                                  {item.label} Avg
                                </div>
                                <div className="mt-2 flex items-end gap-2">
                                  <div
                                    className={`w-8 rounded-md ${item.color}`}
                                    style={{ height: `${(item.value || 0) * 18}px` }}
                                  />
                                  <div className={`text-lg font-semibold ${themeClass('text-slate-200', 'text-slate-700')}`}>
                                    {(item.value || 0).toFixed(2)}
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>

                        {hasAnyParameter && (
                          <div className="mt-6">
                            <div className={`text-xs uppercase tracking-[0.3em] ${themeClass('text-slate-400', 'text-slate-500')}`}>
                              Parameter Profile
                            </div>
                            <div className="mt-4 space-y-4">
                              {[
                                { label: 'Lean (IL)', key: 'IL', color: 'bg-accent', values: parameters.IL },
                                { label: 'Six Sigma (IS)', key: 'IS', color: 'bg-accent-2', values: parameters.IS },
                                { label: 'Maturity (IM)', key: 'IM', color: 'bg-accent-3', values: parameters.IM }
                              ].map((group) => (
                                <div
                                  key={group.key}
                                  className={`rounded-lg p-3 shadow-sm ${themeClass('bg-slate-900/80', 'bg-slate-50')}`}
                                >
                                  <div className={`flex items-center justify-between text-xs ${themeClass('text-slate-400', 'text-slate-500')}`}>
                                    <span className={`font-semibold ${themeClass('text-slate-200', 'text-slate-700')}`}>
                                      {group.label}
                                    </span>
                                    <span>1-5</span>
                                  </div>
                                  <div className="mt-3 grid grid-cols-7 gap-2">
                                    {group.values.map((value, idx) => {
                                      const normalized = Math.max(0, Math.min(1, (Number(value) || 0) / 5));
                                      return (
                                        <div key={`${group.key}-${idx}`} className="flex flex-col items-center gap-2">
                                          <div className={`flex h-20 w-full items-end rounded-lg ${themeClass('bg-slate-800', 'bg-slate-200')}`}>
                                            <div
                                              className={`w-full rounded-lg ${group.color} transition-all`}
                                              style={{ height: `${normalized * 100}%` }}
                                            />
                                          </div>
                                          <div className={`text-[11px] ${themeClass('text-slate-400', 'text-slate-500')}`}>
                                            {group.key}{idx + 1}
                                          </div>
                                        </div>
                                      );
                                    })}
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                        {renderChartBlock(calcResult?.chart, 'Detailed CSF Profile')}
                        {renderPerformanceTable(calcResult?.performance_table)}
                        {renderLevelUpTable(lastCalcParams)}
                        <div className={`mt-6 rounded-xl border p-4 ${themeClass('border-slate-800 bg-slate-900/80', 'border-slate-200 bg-white')}`}>
                          <div className="flex flex-wrap items-center justify-between gap-3">
                            <div>
                              <div className={`text-xs uppercase tracking-wide ${themeClass('text-slate-400', 'text-slate-500')}`}>
                                AI Explanation
                              </div>
                              <div className={`mt-1 text-sm font-semibold ${themeClass('text-slate-100', 'text-slate-900')}`}>
                                Clarify the assessment results
                              </div>
                            </div>
                            <button
                              type="button"
                              onClick={() => handleExplainCalculation(lastCalcParams, calcResult)}
                              disabled={calcExplanationLoading || loading}
                              className="button-secondary"
                            >
                              {calcExplanationLoading ? 'Explaining...' : 'Explain results'}
                            </button>
                          </div>
                          {calcExplanation && (
                            <div className={`mt-4 rounded-lg p-4 text-sm leading-relaxed ${themeClass('bg-slate-900/80 text-slate-100', 'bg-slate-50 text-slate-700')}`}>
                              <p className="whitespace-pre-wrap" dir="auto">{calcExplanation}</p>
                            </div>
                          )}
                          {!calcExplanation && !calcExplanationLoading && (
                            <div className={`mt-3 text-xs ${themeClass('text-slate-400', 'text-slate-500')}`}>
                              Generate a plain-language explanation with tailored recommendations.
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
    {/* === ADMIN/USER SIDEBAR === */}
    {userMenuOpen && (
      <>
        <div
          onClick={() => { setUserMenuOpen(false); setAdminPanel(null); }}
          style={{
            position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)',
            zIndex: 9998, backdropFilter: 'blur(2px)'
          }}
        />
        <div style={{
          position: 'fixed', top: 0, right: 0, bottom: 0, width: '400px',
          background: isDark ? '#0f172a' : '#ffffff',
          borderLeft: '1px solid', borderColor: isDark ? '#334155' : '#e2e8f0',
          boxShadow: '-10px 0 30px rgba(0,0,0,0.3)',
          zIndex: 9999, display: 'flex', flexDirection: 'column',
          animation: 'slideIn 0.3s ease'
        }}>
          {/* Header */}
          <div style={{
            padding: '20px 24px',
            background: isAdmin
              ? 'linear-gradient(135deg, #ef4444, #a855f7)'
              : 'linear-gradient(135deg, #3b82f6, #0ea5e9)',
            color: 'white'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                {isAdmin ? <Shield className="w-6 h-6" /> : <User className="w-6 h-6" />}
                <div>
                  <div style={{ fontSize: '11px', opacity: 0.8, textTransform: 'uppercase', letterSpacing: '1px' }}>
                    {isAdmin ? 'Administrator' : 'User'}
                  </div>
                  <div style={{ fontSize: '18px', fontWeight: 700 }}>{currentUser?.username}</div>
                </div>
              </div>
              <button
                onClick={() => { setUserMenuOpen(false); setAdminPanel(null); }}
                style={{
                  background: 'rgba(255,255,255,0.2)', border: 'none',
                  color: 'white', width: '32px', height: '32px',
                  borderRadius: '50%', cursor: 'pointer', fontSize: '18px',
                  display: 'flex', alignItems: 'center', justifyContent: 'center'
                }}
              >
                ×
              </button>
            </div>
          </div>
 
          {/* Content area */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '20px' }}>
            {adminPanel === null && (
              <>
                {isAdmin && (
                  <>
                    <div style={{
                      fontSize: '11px', textTransform: 'uppercase', letterSpacing: '1px',
                      color: isDark ? '#94a3b8' : '#64748b', marginBottom: '12px', fontWeight: 600
                    }}>
                      Admin Panel
                    </div>
                    {[
                      { id: 'users', icon: Users, label: 'Manage Users', desc: 'Add, edit, or remove users' },
                      { id: 'companies', icon: Database, label: 'Manage Companies', desc: 'Edit the 156 cases database' },
                      { id: 'stats', icon: BarChart3, label: 'Model Statistics', desc: 'View accuracy & metrics' }
                    ].map(item => (
                      <button
                        key={item.id}
                        onClick={() => setAdminPanel(item.id)}
                        style={{
                          display: 'flex', alignItems: 'center', gap: '14px',
                          width: '100%', padding: '14px 16px', marginBottom: '8px',
                          background: isDark ? '#1e293b' : '#f8fafc',
                          border: '1px solid', borderColor: isDark ? '#334155' : '#e2e8f0',
                          borderRadius: '10px', cursor: 'pointer', textAlign: 'left',
                          color: isDark ? '#e2e8f0' : '#0f172a', transition: 'all 0.15s'
                        }}
                        onMouseEnter={(e) => e.currentTarget.style.background = isDark ? '#334155' : '#f1f5f9'}
                        onMouseLeave={(e) => e.currentTarget.style.background = isDark ? '#1e293b' : '#f8fafc'}
                      >
                        <item.icon className="w-5 h-5" />
                        <div style={{ flex: 1 }}>
                          <div style={{ fontWeight: 600, fontSize: '14px' }}>{item.label}</div>
                          <div style={{ fontSize: '12px', opacity: 0.7, marginTop: '2px' }}>{item.desc}</div>
                        </div>
                      </button>
                    ))}
                    <div style={{ height: '1px', background: isDark ? '#334155' : '#e2e8f0', margin: '20px 0' }} />
                  </>
                )}
 
                <div style={{
                  fontSize: '11px', textTransform: 'uppercase', letterSpacing: '1px',
                  color: isDark ? '#94a3b8' : '#64748b', marginBottom: '12px', fontWeight: 600
                }}>
                  Account
                </div>
                <button
                  onClick={() => alert('Profile coming soon')}
                  style={{
                    display: 'flex', alignItems: 'center', gap: '14px',
                    width: '100%', padding: '14px 16px', marginBottom: '8px',
                    background: isDark ? '#1e293b' : '#f8fafc',
                    border: '1px solid', borderColor: isDark ? '#334155' : '#e2e8f0',
                    borderRadius: '10px', cursor: 'pointer', textAlign: 'left',
                    color: isDark ? '#e2e8f0' : '#0f172a'
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.background = isDark ? '#334155' : '#f1f5f9'}
                  onMouseLeave={(e) => e.currentTarget.style.background = isDark ? '#1e293b' : '#f8fafc'}
                >
                  <User className="w-5 h-5" />
                  <span style={{ fontWeight: 600, fontSize: '14px' }}>My Profile</span>
                </button>
                <button
                  onClick={handleLogout}
                  style={{
                    display: 'flex', alignItems: 'center', gap: '14px',
                    width: '100%', padding: '14px 16px',
                    background: 'rgba(239,68,68,0.1)',
                    border: '1px solid rgba(239,68,68,0.3)',
                    borderRadius: '10px', cursor: 'pointer', textAlign: 'left',
                    color: '#ef4444', fontWeight: 600, fontSize: '14px'
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(239,68,68,0.2)'}
                  onMouseLeave={(e) => e.currentTarget.style.background = 'rgba(239,68,68,0.1)'}
                >
                  <LogOut className="w-5 h-5" /> Logout
                </button>
              </>
            )}
 
            {adminPanel === 'users' && (
              <div>
                <button onClick={() => setAdminPanel(null)} style={{
                  background: 'transparent', border: 'none', color: isDark ? '#94a3b8' : '#64748b',
                  fontSize: '13px', cursor: 'pointer', marginBottom: '16px', padding: 0,
                  display: 'flex', alignItems: 'center', gap: '4px'
                }}>← Back</button>
                <h3 style={{ fontSize: '18px', fontWeight: 700, marginBottom: '16px', color: isDark ? '#fff' : '#0f172a' }}>
                  User Management
                </h3>
                <p style={{ fontSize: '14px', color: isDark ? '#94a3b8' : '#64748b', marginBottom: '20px' }}>
                  Add, view, or remove users and assign roles.
                </p>

                {/* Registration Form Card */}
                <div style={{
                  padding: '20px',
                  background: isDark ? '#1e293b' : '#f8fafc',
                  borderRadius: '12px',
                  border: '1px solid',
                  borderColor: isDark ? '#334155' : '#e2e8f0',
                  marginBottom: '24px'
                }}>
                  <h4 style={{ fontSize: '14px', fontWeight: 600, color: isDark ? '#f1f5f9' : '#1e293b', marginBottom: '14px' }}>
                    Create New User
                  </h4>
                  <form onSubmit={handleRegisterUser} style={{ display: 'grid', gap: '12px' }}>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                      <div>
                        <label style={{ fontSize: '11px', fontWeight: 600, color: isDark ? '#94a3b8' : '#64748b', display: 'block', marginBottom: '4px' }}>
                          Username
                        </label>
                        <input
                          type="text"
                          value={newUserUsername}
                          onChange={(e) => setNewUserUsername(e.target.value)}
                          placeholder="e.g. johndoe"
                          style={{
                            width: '100%',
                            padding: '8px 12px',
                            background: isDark ? '#0f172a' : '#fff',
                            border: '1px solid',
                            borderColor: isDark ? '#334155' : '#cbd5e1',
                            borderRadius: '6px',
                            color: isDark ? '#fff' : '#0f172a',
                            fontSize: '13px'
                          }}
                        />
                      </div>
                      <div>
                        <label style={{ fontSize: '11px', fontWeight: 600, color: isDark ? '#94a3b8' : '#64748b', display: 'block', marginBottom: '4px' }}>
                          Password
                        </label>
                        <input
                          type="password"
                          value={newUserPassword}
                          onChange={(e) => setNewUserPassword(e.target.value)}
                          placeholder="••••••••"
                          style={{
                            width: '100%',
                            padding: '8px 12px',
                            background: isDark ? '#0f172a' : '#fff',
                            border: '1px solid',
                            borderColor: isDark ? '#334155' : '#cbd5e1',
                            borderRadius: '6px',
                            color: isDark ? '#fff' : '#0f172a',
                            fontSize: '13px'
                          }}
                        />
                      </div>
                    </div>
                    <div>
                      <label style={{ fontSize: '11px', fontWeight: 600, color: isDark ? '#94a3b8' : '#64748b', display: 'block', marginBottom: '4px' }}>
                        Role
                      </label>
                      <select
                        value={newUserRole}
                        onChange={(e) => setNewUserRole(e.target.value)}
                        style={{
                          width: '100%',
                          padding: '8px 12px',
                          background: isDark ? '#0f172a' : '#fff',
                          border: '1px solid',
                          borderColor: isDark ? '#334155' : '#cbd5e1',
                          borderRadius: '6px',
                          color: isDark ? '#fff' : '#0f172a',
                          fontSize: '13px'
                        }}
                      >
                        <option value="utilisateur">Utilisateur (User)</option>
                        <option value="admin">Admin (Administrator)</option>
                      </select>
                    </div>
                    {registerError && (
                      <div style={{ color: '#ef4444', fontSize: '12px', marginTop: '4px' }}>
                        ⚠️ {registerError}
                      </div>
                    )}
                    {registerSuccess && (
                      <div style={{ color: '#10b981', fontSize: '12px', marginTop: '4px' }}>
                        {registerSuccess}
                      </div>
                    )}
                    <button
                      type="submit"
                      disabled={registerLoading}
                      style={{
                        padding: '10px 16px',
                        background: 'var(--accent, #3b82f6)',
                        color: 'white',
                        border: 'none',
                        borderRadius: '6px',
                        fontWeight: 600,
                        fontSize: '13px',
                        cursor: 'pointer',
                        marginTop: '4px',
                        transition: 'opacity 0.15s'
                      }}
                      onMouseEnter={(e) => e.currentTarget.style.opacity = '0.9'}
                      onMouseLeave={(e) => e.currentTarget.style.opacity = '1'}
                    >
                      {registerLoading ? 'Creating...' : 'Register User'}
                    </button>
                  </form>
                </div>

                {/* User List Container */}
                <h4 style={{ fontSize: '14px', fontWeight: 600, color: isDark ? '#f1f5f9' : '#1e293b', marginBottom: '12px' }}>
                  Registered Users
                </h4>
                {usersLoading ? (
                  <div style={{ textAlign: 'center', padding: '20px', color: isDark ? '#94a3b8' : '#64748b' }}>
                    <Loader2 className="w-5 h-5 animate-spin mx-auto mb-2" />
                    Loading users...
                  </div>
                ) : usersError ? (
                  <div style={{ padding: '16px', background: 'rgba(239,68,68,0.1)', color: '#ef4444', borderRadius: '8px', fontSize: '13px' }}>
                    Failed to load users: {usersError}
                  </div>
                ) : users.length === 0 ? (
                  <div style={{ textAlign: 'center', padding: '20px', color: isDark ? '#94a3b8' : '#64748b', fontSize: '13px' }}>
                    No users registered.
                  </div>
                ) : (
                  <div style={{ display: 'grid', gap: '10px' }}>
                    {users.map((u) => (
                      <div
                        key={u.id}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          padding: '12px 16px',
                          background: isDark ? '#1e293b' : '#f8fafc',
                          border: '1px solid',
                          borderColor: isDark ? '#334155' : '#e2e8f0',
                          borderRadius: '8px'
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                          <div style={{
                            width: '32px',
                            height: '32px',
                            borderRadius: '50%',
                            background: u.role === 'admin' ? '#ef444422' : '#3b82f622',
                            color: u.role === 'admin' ? '#ef4444' : '#3b82f6',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontWeight: 600,
                            fontSize: '14px'
                          }}>
                            {u.username.substring(0, 2).toUpperCase()}
                          </div>
                          <div>
                            <div style={{ fontWeight: 600, fontSize: '14px', color: isDark ? '#fff' : '#0f172a' }}>
                              {u.username}
                              {currentUser?.id === u.id && (
                                <span style={{ fontSize: '10px', opacity: 0.6, marginLeft: '6px' }}>(You)</span>
                              )}
                            </div>
                            <div style={{ marginTop: '2px' }}>
                              <span style={{
                                fontSize: '10px',
                                textTransform: 'uppercase',
                                fontWeight: 700,
                                padding: '2px 6px',
                                borderRadius: '4px',
                                background: u.role === 'admin' ? 'rgba(239,68,68,0.15)' : 'rgba(59,130,246,0.15)',
                                color: u.role === 'admin' ? '#ef4444' : '#3b82f6'
                              }}>
                                {u.role}
                              </span>
                            </div>
                          </div>
                        </div>

                        {/* Delete Button */}
                        <button
                          onClick={() => handleDeleteUser(u.id, u.username)}
                          disabled={currentUser?.id === u.id}
                          style={{
                            background: 'transparent',
                            border: 'none',
                            color: currentUser?.id === u.id ? (isDark ? '#475569' : '#cbd5e1') : '#ef4444',
                            cursor: currentUser?.id === u.id ? 'not-allowed' : 'pointer',
                            padding: '6px',
                            borderRadius: '6px',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            transition: 'background 0.15s'
                          }}
                          onMouseEnter={(e) => {
                            if (currentUser?.id !== u.id) e.currentTarget.style.background = 'rgba(239,68,68,0.1)';
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.background = 'transparent';
                          }}
                          title={currentUser?.id === u.id ? "You cannot delete yourself" : "Delete User"}
                        >
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {adminPanel === 'companies' && (
              <div>
                <button onClick={() => setAdminPanel(null)} style={{
                  background: 'transparent', border: 'none', color: isDark ? '#94a3b8' : '#64748b',
                  fontSize: '13px', cursor: 'pointer', marginBottom: '16px', padding: 0
                }}>← Back</button>
                <h3 style={{ fontSize: '18px', fontWeight: 700, marginBottom: '16px', color: isDark ? '#fff' : '#0f172a' }}>
                  Companies Database
                </h3>
                <p style={{ fontSize: '14px', color: isDark ? '#94a3b8' : '#64748b', marginBottom: '20px' }}>
                  Browse and analyze the 156 companies used to train the model.
                </p>

                {/* Filters Row */}
                <div style={{
                  display: 'flex',
                  gap: '12px',
                  marginBottom: '20px',
                  flexWrap: 'wrap'
                }}>
                  <input
                    type="text"
                    value={companiesSearch}
                    onChange={(e) => {
                      setCompaniesSearch(e.target.value);
                      setCompaniesPage(1);
                    }}
                    placeholder="Search by strategy or estimation..."
                    style={{
                      flex: 1,
                      minWidth: '200px',
                      padding: '8px 12px',
                      background: isDark ? '#1e293b' : '#fff',
                      border: '1px solid',
                      borderColor: isDark ? '#334155' : '#cbd5e1',
                      borderRadius: '6px',
                      color: isDark ? '#fff' : '#0f172a',
                      fontSize: '13px'
                    }}
                  />
                  <select
                    value={companiesFilterStrategy}
                    onChange={(e) => {
                      setCompaniesFilterStrategy(e.target.value);
                      setCompaniesPage(1);
                    }}
                    style={{
                      padding: '8px 12px',
                      background: isDark ? '#1e293b' : '#fff',
                      border: '1px solid',
                      borderColor: isDark ? '#334155' : '#cbd5e1',
                      borderRadius: '6px',
                      color: isDark ? '#fff' : '#0f172a',
                      fontSize: '13px'
                    }}
                  >
                    <option value="all">All Strategies</option>
                    <option value="SS then LM">SS then LM</option>
                    <option value="LM then SS">LM then SS</option>
                    <option value="SS and LM parallel">SS and LM parallel</option>
                    <option value="No orientation">No orientation</option>
                  </select>
                  <select
                    value={companiesFilterCluster}
                    onChange={(e) => {
                      setCompaniesFilterCluster(e.target.value);
                      setCompaniesPage(1);
                    }}
                    style={{
                      padding: '8px 12px',
                      background: isDark ? '#1e293b' : '#fff',
                      border: '1px solid',
                      borderColor: isDark ? '#334155' : '#cbd5e1',
                      borderRadius: '6px',
                      color: isDark ? '#fff' : '#0f172a',
                      fontSize: '13px'
                    }}
                  >
                    <option value="all">All Clusters</option>
                    <option value="0">Cluster 0</option>
                    <option value="1">Cluster 1</option>
                    <option value="2">Cluster 2</option>
                    <option value="3">Cluster 3</option>
                  </select>
                </div>

                {companiesLoading ? (
                  <div style={{ textAlign: 'center', padding: '40px', color: isDark ? '#94a3b8' : '#64748b' }}>
                    <Loader2 className="w-8 h-8 animate-spin mx-auto mb-2" />
                    Loading companies database...
                  </div>
                ) : companiesError ? (
                  <div style={{ padding: '16px', background: 'rgba(239,68,68,0.1)', color: '#ef4444', borderRadius: '8px', fontSize: '13px' }}>
                    Failed to load database: {companiesError}
                  </div>
                ) : (
                  <div>
                    {/* Table View */}
                    {(() => {
                      // Filter companies
                      const filtered = companies.filter((c, idx) => {
                        const searchLower = companiesSearch.toLowerCase();
                        const matchesSearch =
                          (c.strategie && c.strategie.toLowerCase().includes(searchLower)) ||
                          (c.estimations && c.estimations.toLowerCase().includes(searchLower)) ||
                          `#${idx + 1}`.includes(searchLower);
                        
                        const matchesStrategy =
                          companiesFilterStrategy === 'all' ||
                          c.strategie === companiesFilterStrategy;
                        
                        const matchesCluster =
                          companiesFilterCluster === 'all' ||
                          String(c.cluster_id) === companiesFilterCluster;

                        return matchesSearch && matchesStrategy && matchesCluster;
                      });

                      const itemsPerPage = 12;
                      const totalPages = Math.ceil(filtered.length / itemsPerPage) || 1;
                      const currentPage = Math.min(companiesPage, totalPages);
                      const startIndex = (currentPage - 1) * itemsPerPage;
                      const paginated = filtered.slice(startIndex, startIndex + itemsPerPage);

                      // Helper to compute averages
                      const getAverageScore = (comp, regex) => {
                        const keys = Object.keys(comp).filter(k => regex.test(k));
                        if (keys.length === 0) return 0;
                        const sum = keys.reduce((acc, k) => acc + comp[k], 0);
                        return (sum / keys.length).toFixed(1);
                      };

                      return (
                        <>
                          <div style={{ fontSize: '12px', color: isDark ? '#94a3b8' : '#64748b', marginBottom: '10px', fontWeight: 500 }}>
                            Showing {filtered.length === 0 ? 0 : startIndex + 1} - {Math.min(startIndex + itemsPerPage, filtered.length)} of {filtered.length} companies
                          </div>
                          
                          <div style={{ overflowX: 'auto', border: '1px solid', borderColor: isDark ? '#334155' : '#e2e8f0', borderRadius: '8px' }}>
                            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px', textAlign: 'left' }}>
                              <thead>
                                <tr style={{ background: isDark ? '#1e293b' : '#f1f5f9', borderBottom: '1px solid', borderColor: isDark ? '#334155' : '#e2e8f0' }}>
                                  <th style={{ padding: '12px 16px', color: isDark ? '#94a3b8' : '#475569', fontWeight: 600 }}>ID</th>
                                  <th style={{ padding: '12px 16px', color: isDark ? '#94a3b8' : '#475569', fontWeight: 600 }}>Lean Avg (IL)</th>
                                  <th style={{ padding: '12px 16px', color: isDark ? '#94a3b8' : '#475569', fontWeight: 600 }}>Six Sigma Avg (IS)</th>
                                  <th style={{ padding: '12px 16px', color: isDark ? '#94a3b8' : '#475569', fontWeight: 600 }}>Maturity Avg (M)</th>
                                  <th style={{ padding: '12px 16px', color: isDark ? '#94a3b8' : '#475569', fontWeight: 600 }}>Recommended Strategy</th>
                                  <th style={{ padding: '12px 16px', color: isDark ? '#94a3b8' : '#475569', fontWeight: 600 }}>Estimation</th>
                                  <th style={{ padding: '12px 16px', color: isDark ? '#94a3b8' : '#475569', fontWeight: 600 }}>Cluster</th>
                                </tr>
                              </thead>
                              <tbody>
                                {paginated.map((c, i) => {
                                  const companyId = startIndex + i + 1;
                                  return (
                                    <tr
                                      key={i}
                                      style={{
                                        borderBottom: '1px solid',
                                        borderColor: isDark ? '#334155' : '#e2e8f0',
                                        background: isDark ? (i % 2 === 0 ? '#0f172a' : '#1e293b') : (i % 2 === 0 ? '#fff' : '#f8fafc'),
                                        transition: 'background 0.15s'
                                      }}
                                    >
                                      <td style={{ padding: '12px 16px', fontWeight: 600, color: isDark ? '#fff' : '#0f172a' }}>#{companyId}</td>
                                      <td style={{ padding: '12px 16px' }}>
                                        <span style={{
                                          padding: '2px 6px', borderRadius: '4px', background: 'rgba(59,130,246,0.1)', color: '#3b82f6', fontWeight: 600
                                        }}>{getAverageScore(c, /^IL\d/)}</span>
                                      </td>
                                      <td style={{ padding: '12px 16px' }}>
                                        <span style={{
                                          padding: '2px 6px', borderRadius: '4px', background: 'rgba(16,185,129,0.1)', color: '#10b981', fontWeight: 600
                                        }}>{getAverageScore(c, /^IS\d/)}</span>
                                      </td>
                                      <td style={{ padding: '12px 16px' }}>
                                        <span style={{
                                          padding: '2px 6px', borderRadius: '4px', background: 'rgba(245,158,11,0.1)', color: '#f59e0b', fontWeight: 600
                                        }}>{getAverageScore(c, /^M\d/)}</span>
                                      </td>
                                      <td style={{ padding: '12px 16px', fontWeight: 500, color: isDark ? '#e2e8f0' : '#334155' }}>{c.strategie}</td>
                                      <td style={{ padding: '12px 16px', color: '#8b5cf6', fontWeight: 600 }}>{c.estimations}</td>
                                      <td style={{ padding: '12px 16px' }}>
                                        <span style={{
                                          padding: '2px 6px', borderRadius: '4px',
                                          background: isDark ? '#334155' : '#e2e8f0',
                                          color: isDark ? '#f1f5f9' : '#475569',
                                          fontSize: '11px',
                                          fontWeight: 600
                                        }}>C{c.cluster_id}</span>
                                      </td>
                                    </tr>
                                  );
                                })}
                                {paginated.length === 0 && (
                                  <tr>
                                    <td colSpan="7" style={{ padding: '30px', textAlign: 'center', color: isDark ? '#94a3b8' : '#64748b' }}>
                                      No companies match the filters.
                                    </td>
                                  </tr>
                                )}
                              </tbody>
                            </table>
                          </div>

                          {/* Pagination controls */}
                          {totalPages > 1 && (
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '12px', marginTop: '20px' }}>
                              <button
                                onClick={() => setCompaniesPage(prev => Math.max(prev - 1, 1))}
                                disabled={currentPage === 1}
                                style={{
                                  padding: '6px 12px',
                                  background: currentPage === 1 ? 'transparent' : (isDark ? '#1e293b' : '#fff'),
                                  border: '1px solid',
                                  borderColor: isDark ? '#334155' : '#cbd5e1',
                                  borderRadius: '6px',
                                  color: currentPage === 1 ? (isDark ? '#475569' : '#cbd5e1') : (isDark ? '#fff' : '#0f172a'),
                                  cursor: currentPage === 1 ? 'not-allowed' : 'pointer',
                                  fontSize: '13px'
                                }}
                              >
                                Previous
                              </button>
                              <span style={{ fontSize: '13px', color: isDark ? '#94a3b8' : '#64748b' }}>
                                Page {currentPage} of {totalPages}
                              </span>
                              <button
                                onClick={() => setCompaniesPage(prev => Math.min(prev + 1, totalPages))}
                                disabled={currentPage === totalPages}
                                style={{
                                  padding: '6px 12px',
                                  background: currentPage === totalPages ? 'transparent' : (isDark ? '#1e293b' : '#fff'),
                                  border: '1px solid',
                                  borderColor: isDark ? '#334155' : '#cbd5e1',
                                  borderRadius: '6px',
                                  color: currentPage === totalPages ? (isDark ? '#475569' : '#cbd5e1') : (isDark ? '#fff' : '#0f172a'),
                                  cursor: currentPage === totalPages ? 'not-allowed' : 'pointer',
                                  fontSize: '13px'
                                }}
                              >
                                Next
                              </button>
                            </div>
                          )}
                        </>
                      );
                    })()}
                  </div>
                )}
              </div>
            )}

            {adminPanel === 'stats' && (
              <div>
                <button onClick={() => setAdminPanel(null)} style={{
                  background: 'transparent', border: 'none', color: isDark ? '#94a3b8' : '#64748b',
                  fontSize: '13px', cursor: 'pointer', marginBottom: '16px', padding: 0
                }}>← Back</button>
                <h3 style={{ fontSize: '18px', fontWeight: 700, marginBottom: '16px', color: isDark ? '#fff' : '#0f172a' }}>
                  Model Statistics
                </h3>
                
                {statsLoading ? (
                  <div style={{ textAlign: 'center', padding: '40px', color: isDark ? '#94a3b8' : '#64748b' }}>
                    <Loader2 className="w-8 h-8 animate-spin mx-auto mb-2" />
                    Loading statistics...
                  </div>
                ) : statsError ? (
                  <div style={{ padding: '16px', background: 'rgba(239,68,68,0.1)', color: '#ef4444', borderRadius: '8px', fontSize: '13px', marginBottom: '16px' }}>
                    Failed to load real metrics: {statsError}. Showing defaults instead.
                  </div>
                ) : null}

                <div style={{ display: 'grid', gap: '12px' }}>
                  {[
                    { label: 'Training Samples', value: statsData?.trainingSamples ?? '156', color: '#3b82f6' },
                    { label: 'Random Forest Trees', value: statsData?.rfTrees ?? '100', color: '#10b981' },
                    { label: 'CSF Factors', value: statsData?.csfFactors ?? '21', color: '#f59e0b' },
                    { label: 'Strategies', value: statsData?.strategies ?? '4', color: '#8b5cf6' }
                  ].map(stat => (
                    <div key={stat.label} style={{
                      padding: '16px', background: isDark ? '#1e293b' : '#f8fafc',
                      borderRadius: '10px', border: '1px solid', borderColor: isDark ? '#334155' : '#e2e8f0'
                    }}>
                      <div style={{ fontSize: '12px', color: isDark ? '#94a3b8' : '#64748b', textTransform: 'uppercase', letterSpacing: '0.5px' }}>{stat.label}</div>
                      <div style={{ fontSize: '24px', fontWeight: 700, color: stat.color, marginTop: '4px' }}>{stat.value}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </>
    )}
    </div>
  );
};

export default LeanSixSigmaChatbot;



