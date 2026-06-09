import express from 'express';
import { spawn } from 'child_process';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 5001;
const BACKEND_DIR = path.join(__dirname, '..');
const DATA_PATH = path.join(BACKEND_DIR, 'data', 'unified_lss_data_final.json');
const PYTHON_SCRIPT = path.join(__dirname, 'l6s_api_optimized.py');

const pythonExeCandidates = [
  process.env.PYTHON_EXE,
  path.join(__dirname, '..', '..', '..', '.venv', 'Scripts', 'python.exe'),
  path.join(__dirname, '..', '..', '..', '.venv', 'bin', 'python'),
  path.join(__dirname, '..', '..', '.venv', 'Scripts', 'python.exe'),
  path.join(__dirname, '..', '..', '.venv', 'bin', 'python')
].filter(Boolean);
const pythonExePath = pythonExeCandidates.find(candidate => fs.existsSync(candidate)) || 'python';

const parseJsonFromStdout = (raw) => {
  try {
    return JSON.parse(raw);
  } catch (error) {
    const start = raw.lastIndexOf('{');
    const end = raw.lastIndexOf('}');
    if (start !== -1 && end !== -1 && end > start) {
      return JSON.parse(raw.slice(start, end + 1));
    }
    throw error;
  }
};

const runPython = ({ message, useSidebarValues = false, sidebarScores, conversationHistory, modelConfig, language }) => {
  return new Promise((resolve, reject) => {
    const args = [
      PYTHON_SCRIPT,
      '--message', message,
      '--use-sidebar-values', useSidebarValues ? 'true' : 'false'
    ];

    if (modelConfig) {
      if (modelConfig.local_model) {
        args.push('--local-model', modelConfig.local_model);
      }
      if (modelConfig.api_model) {
        args.push('--api-model', modelConfig.api_model);
      }
      if (modelConfig.temperature !== undefined) {
        args.push('--temperature', modelConfig.temperature.toString());
      }
      if (modelConfig.api_key) {
        args.push('--api-key', modelConfig.api_key);
      }
      if (modelConfig.use_api !== undefined) {
        args.push('--use-api', modelConfig.use_api ? 'true' : 'false');
      }
    }

    if (language) {
      args.push('--language', String(language));
    }

    if (sidebarScores && useSidebarValues) {
      args.push('--sidebar-scores', JSON.stringify(sidebarScores));
    }

    if (conversationHistory && Array.isArray(conversationHistory)) {
      args.push('--conversation-history', JSON.stringify(conversationHistory));
    }

    const pythonProcess = spawn(pythonExePath, args, { stdio: ['pipe', 'pipe', 'pipe'], cwd: BACKEND_DIR });
    let response = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
      response += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (stderr) {
        console.error(`Python error: ${stderr}`);
      }
      if (code !== 0) {
        let message = `Python script failed with code: ${code}`;
        try {
          const parsed = parseJsonFromStdout(response);
          if (parsed?.content) {
            message = parsed.content;
          }
        } catch (error) {
          // Keep generic message if output is not JSON.
        }
        return reject(new Error(message));
      }

      try {
        const result = parseJsonFromStdout(response);
        resolve(result);
      } catch (error) {
        console.error('Error parsing Python response:', error);
        console.error('Response was:', response);
        reject(new Error('Invalid response from Python script'));
      }
    });
  });
};

const parseLevelValue = (value) => {
  if (value === '' || value === null || value === undefined) return null;
  const num = Number(value);
  if (!Number.isFinite(num)) return null;
  const intVal = Math.round(num);
  if (intVal < 1 || intVal > 5) return null;
  return intVal;
};

const averageOf = (values) => {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
};

const buildSidebarScores = (parameters) => {
  const scores = {};
  const addScores = (prefix, values) => {
    if (!Array.isArray(values)) return;
    values.forEach((value, index) => {
      const parsed = parseLevelValue(value);
      if (parsed !== null) {
        scores[`${prefix}${index + 1}`] = parsed;
      }
    });
  };

  addScores('IL', parameters?.IL);
  addScores('IS', parameters?.IS);
  addScores('M', parameters?.IM);
  return scores;
};

const extractRecommendationLines = (text) => {
  if (!text) return [];
  return text
    .split('\n')
    .map(line => line.trim())
    .filter(line => line.startsWith('- '))
    .map(line => line.slice(2).trim());
};

const extractMetric = (text, pattern) => {
  if (!text) return null;
  const match = text.match(pattern);
  if (!match) return null;
  const num = Number(match[1]);
  return Number.isFinite(num) ? num : null;
};

// Enable CORS for all routes
app.use(cors());
// Middleware to parse JSON bodies
app.use(express.json({ limit: '10mb' }));

// Serve static files from the Vite build directory
app.use(express.static(path.join(__dirname, 'dist')));

// Endpoint to handle chat messages
app.post('/api/chat', (req, res) => {
  const { message, useSidebarValues, sidebarScores, conversationHistory, modelConfig, language } = req.body;

  // Validate that we have a message
  if (!message) {
    return res.status(400).json({ error: 'Message is required' });
  }

  runPython({ message, useSidebarValues, sidebarScores, conversationHistory, modelConfig, language })
    .then((result) => {
      let structuredOutput = undefined;

      if (result.type) {
        let responseType = result.type;
        let responseData = {};

        if (result.type === 'text_with_chart') {
          responseType = 'chart';
          responseData = result.chart || result.dataframe || {};

          if (responseData.labels && Array.isArray(responseData.labels) &&
              responseData.labels.some(label =>
                typeof label === 'string' &&
                (label.startsWith('IL') || label.startsWith('IS') || label.startsWith('M'))
              )) {
            if (!responseData.datasets) {
              if (result.dataframe && result.dataframe.csf_values) {
                const csfKeys = Object.keys(result.dataframe.csf_values);
                const csfValues = Object.values(result.dataframe.csf_values);

                responseData = {
                  labels: csfKeys,
                  datasets: [
                    {
                      label: 'CSF Profile',
                      data: csfValues,
                      backgroundColor: 'rgba(54, 162, 235, 0.2)',
                      borderColor: 'rgba(54, 162, 235, 1)',
                      borderWidth: 1
                    }
                  ]
                };
              }
            }
          }
        } else if (result.type === 'text') {
          responseType = 'text';
          responseData = result.dataframe || {};
        } else {
          responseType = result.type;
          responseData = result.dataframe || result.chart || {};
        }

        structuredOutput = {
          type: responseType,
          data: responseData,
          confidence: result.dataframe?.confidence || 95,
          sampleSize: result.dataframe?.sample_size || result.dataframe?.sampleSize || 156
        };
      }

      const formattedResponse = {
        content: result.content,
        structuredOutput: structuredOutput
      };

      if (result.performance_table) {
        formattedResponse.performanceTable = result.performance_table;
      }
      res.json(formattedResponse);
    })
    .catch((error) => {
      res.status(500).json({ error: error.message });
    });
});

app.post('/query', async (req, res) => {
  const { query, conversationHistory, modelConfig, language } = req.body || {};

  if (!query) {
    return res.status(400).json({ error: 'Query is required' });
  }

  try {
    const result = await runPython({
      message: query,
      useSidebarValues: false,
      conversationHistory,
      modelConfig,
      language
    });

    const confidence = typeof result.dataframe?.confidence === 'number'
      ? Math.min(1, result.dataframe.confidence / 100)
      : 0.7;

    res.json({
      answer: result.content || '',
      sources: [],
      graph_context: {
        dataframe: result.dataframe || null,
        chart: result.chart || null,
        performance_table: result.performance_table || null,
        target_action_table: result.target_action_table || null
      },
      confidence,
      rag_mode: 'l6s',
      entities: []
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/llm-test', async (req, res) => {
  const { modelConfig } = req.body || {};
  const testConfig = { ...(modelConfig || {}), use_api: true };
  const testMessage = `__LLM_TEST__${JSON.stringify({ mode: 'api' })}`;

  try {
    const result = await runPython({
      message: testMessage,
      useSidebarValues: false,
      conversationHistory: [],
      modelConfig: testConfig
    });

    const modelInfo = result?.model_info || {};
    const used = modelInfo.used || 'none';
    const ok = used === 'api';
    const status = ok ? 'api' : used === 'local' ? 'fallback' : 'error';
    const message = modelInfo.error || result?.content || (ok ? 'API responded.' : 'API test failed.');

    res.json({
      ok,
      status,
      message,
      model_info: modelInfo
    });
  } catch (error) {
    res.status(500).json({ ok: false, status: 'error', message: error.message });
  }
});

app.post('/calculate', async (req, res) => {
  const { parameters, language } = req.body || {};

  const ilValues = Array.isArray(parameters?.IL)
    ? parameters.IL.map(parseLevelValue).filter(value => value !== null)
    : [];
  const isValues = Array.isArray(parameters?.IS)
    ? parameters.IS.map(parseLevelValue).filter(value => value !== null)
    : [];
  const imValues = Array.isArray(parameters?.IM)
    ? parameters.IM.map(parseLevelValue).filter(value => value !== null)
    : [];

  if (!ilValues.length && !isValues.length && !imValues.length) {
    return res.status(422).json({ detail: 'At least one parameter value must be provided.' });
  }

  const sidebarScores = buildSidebarScores(parameters);

  try {
    const result = await runPython({
      message: 'calculate performance',
      useSidebarValues: true,
      sidebarScores,
      language
    });

    let prediction = typeof result.dataframe?.prediction === 'number' ? result.dataframe.prediction : null;
    if (prediction === null) {
      prediction = extractMetric(result.content, /Predicted performance:\s*([0-9.]+)/i)
        ?? extractMetric(result.content, /Performance[^0-9]*([0-9.]+)%/i);
    }

    let confidenceValue = typeof result.dataframe?.confidence === 'number'
      ? result.dataframe.confidence
      : extractMetric(result.content, /Confidence level:\s*([0-9.]+)/i);

    const strategy = result.dataframe?.strategy
      || (result.content?.match(/Recommended strategy:\s*([^\n]+)/i)?.[1] || null);

    const recommendations = [
      ...(strategy ? [`Recommended strategy: ${strategy}`] : []),
      ...extractRecommendationLines(result.content)
    ];

    let confidence = 'Low';
    if (typeof confidenceValue === 'number') {
      confidence = confidenceValue >= 70 ? 'High' : confidenceValue >= 50 ? 'Medium' : 'Low';
    }

    res.json({
      prediction: typeof prediction === 'number' ? prediction : 0,
      confidence,
      recommendations,
      parameter_analysis: {
        IL_average: averageOf(ilValues),
        IS_average: averageOf(isValues),
        IM_average: averageOf(imValues),
        model_used: true,
        strategy,
        strategy_source: 'l6s'
      },
      assistant_summary: null,
      chart: result.chart || null,
      performance_table: result.performance_table || null
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/csf-levels', (req, res) => {
  const lang = String(req.query.lang || 'fr').toLowerCase();
  if (lang.startsWith('en')) {
    runPython({
      message: `__CSF_LEVELS__${JSON.stringify({ lang: 'en' })}`,
      useSidebarValues: false,
      conversationHistory: []
    })
      .then((result) => {
        if (result?.csf_levels) {
          return res.json(result.csf_levels);
        }
        return res.status(500).json({ error: 'Failed to load translated CSF levels.' });
      })
      .catch((error) => {
        res.status(500).json({ error: error.message });
      });
    return;
  }

  if (!fs.existsSync(DATA_PATH)) {
    return res.status(500).json({ error: `Missing data file at ${DATA_PATH}` });
  }

  try {
    const data = JSON.parse(fs.readFileSync(DATA_PATH, 'utf-8'));
    const csf = data.critical_success_factors || {};
    const factors = [
      ...(csf.lean || []),
      ...(csf.six_sigma || []),
      ...(csf.maturity_levels || [])
    ];

    const factorMapping = {
      L_CSFs1: 'IL1', L_CSFs2: 'IL2', L_CSFs3: 'IL3', L_CSFs4: 'IL4', L_CSFs5: 'IL5', L_CSFs6: 'IL6', L_CSFs7: 'IL7',
      S_CSFs1: 'IS1', S_CSFs2: 'IS2', S_CSFs3: 'IS3', S_CSFs4: 'IS4', S_CSFs5: 'IS5', S_CSFs6: 'IS6', S_CSFs7: 'IS7',
      M_CSFs1: 'IM1', M_CSFs2: 'IM2', M_CSFs3: 'IM3', M_CSFs4: 'IM4', M_CSFs5: 'IM5', M_CSFs6: 'IM6', M_CSFs7: 'IM7'
    };

    const levels = {};
    const labels = {};
    const prescriptions = {};

    factors.forEach((factor) => {
      const mapped = factorMapping[factor.id];
      if (!mapped) return;
      levels[mapped] = {};
      prescriptions[mapped] = {};
      (factor.levels || []).forEach((level) => {
        levels[mapped][level.level] = level.description;
        prescriptions[mapped][level.level] = level.prescription || 'No prescription available';
      });
      const suffix = mapped.startsWith('IL')
        ? ' (Lean)'
        : mapped.startsWith('IS')
          ? ' (Six Sigma)'
          : ' (Maturity)';
      labels[mapped] = `${factor.category || factor.factor || mapped}${suffix}`;
    });

    res.json({ levels, labels, prescriptions });
  } catch (error) {
    res.status(500).json({ error: 'Failed to parse CSF levels data.' });
  }
});

// Endpoint to get CSF statistics (matching the lss-analytics API format)
app.get('/api/statistics', (req, res) => {
  // This would call the Python backend to get actual statistics
  // For now, return mock data in the expected format
  res.json({
    IL1: { mean: 3.2, min: 1, max: 5, std: 0.8, count: 156 },
    IL2: { mean: 3.1, min: 1, max: 5, std: 0.9, count: 156 },
    IL3: { mean: 2.9, min: 1, max: 5, std: 0.7, count: 156 },
    IL4: { mean: 3.4, min: 1, max: 5, std: 0.8, count: 156 },
    IL5: { mean: 3.0, min: 1, max: 5, std: 0.9, count: 156 },
    IL6: { mean: 3.3, min: 1, max: 5, std: 0.7, count: 156 },
    IL7: { mean: 2.8, min: 1, max: 5, std: 0.8, count: 156 },
    IS1: { mean: 3.1, min: 1, max: 5, std: 0.9, count: 156 },
    IS2: { mean: 3.2, min: 1, max: 5, std: 0.8, count: 156 },
    IS3: { mean: 2.9, min: 1, max: 5, std: 0.7, count: 156 },
    IS4: { mean: 3.3, min: 1, max: 5, std: 0.8, count: 156 },
    IS5: { mean: 3.0, min: 1, max: 5, std: 0.9, count: 156 },
    IS6: { mean: 3.4, min: 1, max: 5, std: 0.7, count: 156 },
    IS7: { mean: 2.8, min: 1, max: 5, std: 0.8, count: 156 },
    M1: { mean: 4.2, min: 1, max: 7, std: 1.1, count: 156 },
    M2: { mean: 4.1, min: 1, max: 7, std: 1.0, count: 156 },
    M3: { mean: 4.3, min: 1, max: 7, std: 0.9, count: 156 },
    M4: { mean: 4.5, min: 1, max: 7, std: 0.8, count: 156 },
    M5: { mean: 4.0, min: 1, max: 7, std: 1.1, count: 156 },
    M6: { mean: 4.4, min: 1, max: 7, std: 0.9, count: 156 },
    M7: { mean: 4.2, min: 1, max: 7, std: 1.0, count: 156 }
  });
});

// Catch-all handler for single-page application (for all non-API routes)
app.get(/^(?!\/api\/).*$/, (req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`L6S-UI server is running on http://localhost:${PORT}`);
});
