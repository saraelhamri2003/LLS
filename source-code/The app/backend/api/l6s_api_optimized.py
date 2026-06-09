#!/usr/bin/env python3
"""
L6S API for Node.js Integration with Process-Level Caching
This file provides the same backend functionality as the neo.py Streamlit app
to ensure identical responses for the React frontend, but with proper caching to improve performance.
This version caches the brain instance at the process level to avoid reinstantiation.
"""

import sys
import json
import argparse
import os
import re
import numpy as np
from pathlib import Path
import atexit
import time

BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR.parent
CORE_DIR = BACKEND_DIR / "core"
DATA_DIR = BACKEND_DIR / "data"
UTILS_DIR = BACKEND_DIR / "utils"
UNIFIED_JSON_PATH = DATA_DIR / "unified_lss_data_final.json"

# Global variable to cache the brain instance across requests within the same process
_cached_brain = None
_translation_cache = {}

# Global variable to cache the prompt templates
_cached_prompt_templates = None

def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)

def load_prompt_templates():
    """Load prompt templates from JSON file"""
    global _cached_prompt_templates

    if _cached_prompt_templates is not None:
        return _cached_prompt_templates

    prompt_template_paths = [
        UTILS_DIR / "l6s_prompt_templates.json",
        BACKEND_DIR / "l6s_prompt_templates.json",
    ]

    for prompt_template_path in prompt_template_paths:
        if prompt_template_path.exists():
            try:
                with open(prompt_template_path, 'r', encoding='utf-8') as f:
                    _cached_prompt_templates = json.load(f)
                return _cached_prompt_templates
            except Exception as e:
                print(f"Error loading prompt templates: {e}", file=sys.stderr)
                return None

    searched = ", ".join(str(path) for path in prompt_template_paths)
    print(f"Prompt templates file not found. Searched: {searched}", file=sys.stderr)
    return None


def _normalize_lang(lang):
    if not lang:
        return "fr"
    lower = str(lang).strip().lower()
    if lower.startswith("en"):
        return "en"
    if lower.startswith("fr"):
        return "fr"
    return lower


def _looks_french(text):
    if not text or not isinstance(text, str):
        return False
    lower = text.lower()
    if any(ch in lower for ch in "àâäçéèêëîïôöùûüÿœ"):
        return True
    tokens = re.findall(r"[a-zA-Z']+", lower)
    french_markers = {
        'le', 'la', 'les', 'des', 'un', 'une', 'du', 'au', 'aux',
        'et', 'ou', 'dans', 'sur', 'pour', 'avec', 'sans', 'par',
        'comment', 'pourquoi', 'quelle', 'quel', 'quels', 'quelles',
        'merci', 'bonjour', 'bonsoir', 'niveau', 'signifie'
    }
    french_count = sum(1 for token in tokens if token in french_markers)
    return french_count >= 2


def _strip_translation_preamble(text):
    if not text or not isinstance(text, str):
        return text
    cleaned = text.strip()
    cleaned = re.sub(r'^(here\s+is\s+the\s+translation\s*:\s*)', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'^(translation\s*:\s*)', '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def translate_text(text, target_lang, brain):
    if not text or not isinstance(text, str):
        return text
    lang = _normalize_lang(target_lang)
    if lang == "fr":
        return text
    cache_key = (lang, text)
    if cache_key in _translation_cache:
        return _translation_cache[cache_key]
    if not brain or not getattr(brain, "llm_manager", None) or not brain.llm_manager.is_available():
        return text

    def invoke_translation(prompt):
        try:
            manager = brain.llm_manager
            available = manager.get_available_models() if hasattr(manager, "get_available_models") else {}
            if available.get("local", {}).get("available", False):
                return manager.invoke_local_only(prompt)
            return manager.invoke(prompt)
        except Exception:
            return ""

    prompt = (
        "You are a professional translator. Translate the following text to English. "
        "Keep Lean, Six Sigma, Kaizen, DMAIC, and CSF codes unchanged. "
        "Return only the translation. If the text is already English, return it unchanged.\n\n"
        f"{text}"
    )
    translated = invoke_translation(prompt)
    translated = str(translated).strip().strip('"') if translated else ""
    translated = _strip_translation_preamble(translated)
    if translated and not _looks_french(translated):
        _translation_cache[cache_key] = translated
        return translated

    strict_prompt = (
        "Translate the following text to English. Do NOT include any French words or phrases. "
        "Keep Lean, Six Sigma, Kaizen, DMAIC, and CSF codes unchanged. "
        "Return only the translation.\n\n"
        f"{text}"
    )
    translated = invoke_translation(strict_prompt)
    translated = str(translated).strip().strip('"') if translated else ""
    translated = _strip_translation_preamble(translated)
    if translated:
        _translation_cache[cache_key] = translated
        return translated

    return text


def finalize_response(response, is_french, brain):
    if not isinstance(response, dict):
        return response
    if not is_french and response.get("content"):
        response["content"] = translate_text(response["content"], "en", brain)
    return response

def get_brain_instance(local_model="llama3:8b", api_model="gemini-pro", temperature=0.3, api_key=None, use_api=False):
    """Get a cached instance of the L6SBrain, creating it if necessary"""
    global _cached_brain

    if _cached_brain is not None:
        return _cached_brain

    parent_dir = BACKEND_DIR

    # Add the parent directory to sys.path to import the modules
    for path in (str(parent_dir), str(CORE_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)

    # Change working directory to parent directory for proper file paths
    original_cwd = os.getcwd()
    os.chdir(parent_dir)

    try:
        # Capture all stdout to prevent debug outputs from interfering with JSON response
        import io
        from contextlib import redirect_stdout
        stdout_capture = io.StringIO()

        # Dynamically load the L6SBrain class from either neo.py or link.py
        import importlib.util

        # Check which file to use
        neo_candidates = [CORE_DIR / "neo.py", parent_dir / "neo.py"]
        link_candidates = [CORE_DIR / "link.py", parent_dir / "link.py"]

        neo_path = next((path for path in neo_candidates if path.exists()), None)
        link_path = next((path for path in link_candidates if path.exists()), None)

        module_path = None
        if neo_path:
            module_path = neo_path
            module_name = "neo"
        elif link_path:
            module_path = link_path
            module_name = "link"
        else:
            print(json.dumps({
                "content": "Error: Neither core/neo.py nor core/link.py found in backend directory",
                "type": "text",
                "chart": None,
                "dataframe": None
            }))
            sys.exit(1)

        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)

        # Suppress any output during module loading
        with redirect_stdout(stdout_capture):
            spec.loader.exec_module(module)

        # Get the L6SBrain class
        L6SBrainClass = module.L6SBrain

        # Initialize the L6SBrain (this loads the models and data) with model parameters
        with redirect_stdout(stdout_capture):
            brain = L6SBrainClass(
                local_model=local_model,
                api_model=api_model,
                temperature=temperature,
                api_key=api_key,
                use_api=use_api
            )

        if not brain.data_loaded:
            print(json.dumps({
                "content": f"Error: Could not load L6S data. Please ensure {UNIFIED_JSON_PATH} exists and is valid.",
                "type": "text",
                "chart": None,
                "dataframe": None
            }))
            sys.exit(1)

        # Cache the brain instance
        _cached_brain = brain
        print("Successfully initialized and cached L6SBrain instance", file=sys.stderr)

        return brain

    except Exception as e:
        print(json.dumps({
            "content": f"Error initializing L6SBrain: {str(e)}",
            "type": "text",
            "chart": None,
            "dataframe": None
        }))
        sys.exit(1)
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


def process_request(message, use_sidebar_values, sidebar_scores, conversation_history, model_config=None, language=None):
    """Process a single request using the cached brain instance"""
    # Get or create the brain instance with model configuration
    if model_config is None:
        model_config = {}

    # Extract model parameters from the config
    local_model = model_config.get('local_model', 'llama3:8b')
    api_model = model_config.get('api_model', 'gemini-pro')
    temperature = model_config.get('temperature', 0.3)
    api_key = model_config.get('api_key', None)
    use_api = model_config.get('use_api', False)

    brain = get_brain_instance(
        local_model=local_model,
        api_model=api_model,
        temperature=temperature,
        api_key=api_key,
        use_api=use_api
    )

    def invoke_llm(prompt):
        if not brain or not getattr(brain, "llm_manager", None) or not brain.llm_manager.is_available():
            raise RuntimeError("LLM is not configured.")
        if use_api:
            available = brain.llm_manager.get_available_models().get("api", {}).get("available", False)
            if not available:
                local_available = brain.llm_manager.get_available_models().get("local", {}).get("available", False)
                if local_available:
                    return brain.llm_manager.invoke_local_only(prompt)
                raise RuntimeError("API model is not available. Check API key and model name.")
            try:
                return brain.llm_manager.invoke_api_only(prompt)
            except Exception:
                local_available = brain.llm_manager.get_available_models().get("local", {}).get("available", False)
                if local_available:
                    return brain.llm_manager.invoke_local_only(prompt)
                raise
        return brain.llm.invoke(prompt)

    csf_levels_prefix = "__CSF_LEVELS__"
    if message.startswith(csf_levels_prefix):
        payload_text = message[len(csf_levels_prefix):].strip()
        payload = {}
        if payload_text:
            try:
                payload = json.loads(payload_text)
            except Exception:
                payload = {}
        target_lang = _normalize_lang(payload.get("lang", "fr"))

        levels = {}
        labels = {}
        prescriptions = {}

        if UNIFIED_JSON_PATH.exists():
            try:
                with open(UNIFIED_JSON_PATH, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
            except Exception:
                json_data = {}
        else:
            json_data = {}

        csf = json_data.get("critical_success_factors", {}) if isinstance(json_data, dict) else {}
        factors = [
            *(csf.get("lean", []) if isinstance(csf.get("lean"), list) else []),
            *(csf.get("six_sigma", []) if isinstance(csf.get("six_sigma"), list) else []),
            *(csf.get("maturity_levels", []) if isinstance(csf.get("maturity_levels"), list) else [])
        ]

        factor_mapping = {
            'L_CSFs1': 'IL1', 'L_CSFs2': 'IL2', 'L_CSFs3': 'IL3', 'L_CSFs4': 'IL4',
            'L_CSFs5': 'IL5', 'L_CSFs6': 'IL6', 'L_CSFs7': 'IL7',
            'S_CSFs1': 'IS1', 'S_CSFs2': 'IS2', 'S_CSFs3': 'IS3', 'S_CSFs4': 'IS4',
            'S_CSFs5': 'IS5', 'S_CSFs6': 'IS6', 'S_CSFs7': 'IS7',
            'M_CSFs1': 'IM1', 'M_CSFs2': 'IM2', 'M_CSFs3': 'IM3', 'M_CSFs4': 'IM4',
            'M_CSFs5': 'IM5', 'M_CSFs6': 'IM6', 'M_CSFs7': 'IM7'
        }

        for factor in factors:
            mapped = factor_mapping.get(factor.get("id"))
            if not mapped:
                continue
            levels[mapped] = {}
            prescriptions[mapped] = {}
            for level in factor.get("levels", []) or []:
                description = level.get("description", "")
                prescription = level.get("prescription") or "No prescription available"
                if target_lang == "en":
                    description = translate_text(description, "en", brain)
                    prescription = translate_text(prescription, "en", brain)
                levels[mapped][level.get("level")] = description
                prescriptions[mapped][level.get("level")] = prescription

            suffix = mapped.startswith('IL') and ' (Lean)' or mapped.startswith('IS') and ' (Six Sigma)' or ' (Maturity)'
            if target_lang == "fr":
                label_base = factor.get("factor") or factor.get("category") or mapped
            else:
                label_base = factor.get("category") or factor.get("factor") or mapped
            labels[mapped] = f"{label_base}{suffix}"

        return {
            "content": "",
            "type": "csf_levels",
            "csf_levels": {
                "levels": levels,
                "labels": labels,
                "prescriptions": prescriptions
            }
        }

    llm_test_prefix = "__LLM_TEST__"
    if message.startswith(llm_test_prefix):
        payload_text = message[len(llm_test_prefix):].strip()
        payload = {}
        if payload_text:
            try:
                payload = json.loads(payload_text)
            except Exception:
                payload = {}

        requested_mode = str(payload.get("mode", "api")).lower()
        prompt = payload.get("prompt") or "Reply with the single word: OK."
        available_models = brain.get_available_models() if brain else {
            "local": {"available": False, "model": None},
            "api": {"available": False, "model": None},
            "active": "none"
        }

        used_mode = "none"
        error_message = None
        sample = None

        try:
            if not getattr(brain, "llm_manager", None) or not brain.llm_manager.is_available():
                raise RuntimeError("LLM is not configured.")

            if requested_mode == "api":
                try:
                    response_text = brain.llm_manager.invoke_api_only(prompt)
                    used_mode = "api"
                except Exception as exc:
                    error_message = str(exc)
                    response_text = brain.llm_manager.invoke_local_only(prompt)
                    used_mode = "local"
            elif requested_mode == "local":
                response_text = brain.llm_manager.invoke_local_only(prompt)
                used_mode = "local"
            else:
                response_text = brain.llm_manager.invoke(prompt)
                used_mode = brain.llm_manager.active

            if response_text:
                sample = str(response_text).strip().replace("\n", " ")
                if len(sample) > 160:
                    sample = sample[:160] + "..."
        except Exception as exc:
            if error_message is None:
                error_message = str(exc)

        used_model = None
        if used_mode == "api":
            used_model = available_models.get("api", {}).get("model")
        elif used_mode == "local":
            used_model = available_models.get("local", {}).get("model")

        model_info = {
            "requested": requested_mode,
            "used": used_mode,
            "used_model": used_model,
            "active": brain.llm_manager.active if getattr(brain, "llm_manager", None) else available_models.get("active", "none"),
            "active_model": brain.get_active_model_info() if brain else "none",
            "api_available": available_models.get("api", {}).get("available", False),
            "local_available": available_models.get("local", {}).get("available", False),
            "api_model": available_models.get("api", {}).get("model"),
            "local_model": available_models.get("local", {}).get("model"),
            "error": error_message,
            "sample": sample
        }

        if used_mode == "api" and not error_message:
            content = "LLM test succeeded."
        elif used_mode == "local":
            content = "LLM test completed with local fallback."
        else:
            content = "LLM test failed."
        return {
            "content": content,
            "type": "text",
            "chart": None,
            "dataframe": None,
            "model_info": model_info
        }
    
    # Check if message is in French using simple language detection (after message is defined)
    # Enhanced to include common typos (like 'e veux' instead of 'je veux')
    message_lower = message.lower()
    requested_lang = _normalize_lang(language) if language else None

    # Count French indicators to make detection more robust
    french_indicators = [
        'bonjour', 'merci', 's\'il vous plaît', 'svp', 'comment', 'quelle', 'quelle est',
        'aidez', 'sujet', 'stratégie', 'performance', 'objectif', 'but', ' Lean', 'sigma',
        'facteur', 'critique', 'succès', 'organisation', 'améliorer', 'améliorer la',
        'quelle est la', 'quelle est la stratégie', 'conseil', 'recommandation', 'analyse',
        'faut', 'doit', 'quel', 'quelle', 'souhaite', 'voudrais', 'voulez-vous',
        'je veux', 'je souhaite', 'atteindre', 'parametres', 'concrets', 'minimales'
    ]

    # Enhanced English detection
    english_indicators = [
        'i want', 'i need', 'how to', 'what about', 'can you', 'please help',
        'achieve', 'reach', 'target', 'goal', 'performance of', 'get to',
        'help me', 'show me', 'tell me', 'want to', 'need to'
    ]

    # Check for partial matches and common typos
    french_score = 0
    for indicator in french_indicators:
        if indicator in message_lower:
            french_score += 1

    # Also check for common typos and partial patterns
    if any(typo in message_lower for typo in [' e veux', 'e veux ', ' veux', 'e veut', ' veux ']):
        french_score += 1  # Boost score for likely "je veux" typos
    if 'atteindr' in message_lower:  # Likely "atteindre"
        french_score += 1
    if 'parametr' in message_lower:  # Likely "parametres"
        french_score += 1
    if 'concret' in message_lower:  # Likely "concrets"
        french_score += 1
    if 'minimale' in message_lower:  # Likely "minimales"
        french_score += 1

    # Count English indicators
    english_score = 0
    for indicator in english_indicators:
        if indicator in message_lower:
            english_score += 1

    # Override French detection if English indicators are strong
    if english_score >= 2:  # Strong English signal
        is_french = False
    elif french_score > 0 and english_score == 0:
        is_french = True
    else:
        is_french = False  # Default to English when uncertain

    # Honor explicit language selection from the client when provided.
    if requested_lang == "fr":
        is_french = True
    elif requested_lang == "en":
        is_french = False

    # Add debug output temporarily to see what's being detected
    # print(f"DEBUG - Message: {message[:50]}...", file=sys.stderr)
    # print(f"DEBUG - English score: {english_score}, French score: {french_score}", file=sys.stderr)
    # print(f"DEBUG - Detected as French: {is_french}", file=sys.stderr)

    # Convert string values to appropriate types
    use_sidebar_values = use_sidebar_values == 'true'
    sidebar_scores = json.loads(sidebar_scores) if sidebar_scores else {}
    conversation_history = json.loads(conversation_history) if conversation_history else []

    explain_prefix = "__EXPLAIN_RESULT__"
    if message.startswith(explain_prefix):
        payload_text = message[len(explain_prefix):].strip()
        payload = {}
        if payload_text:
            try:
                payload = json.loads(payload_text)
            except Exception:
                payload = {}

        payload_language = str(payload.get("language", "")).lower()
        if payload_language.startswith("fr"):
            is_french = True
        elif payload_language.startswith("en"):
            is_french = False

        raw_scores = payload.get("scores", {}) if isinstance(payload, dict) else {}
        cleaned_scores = {}
        if isinstance(raw_scores, dict):
            for key, value in raw_scores.items():
                key_str = str(key).upper().replace(" ", "")
                if key_str in brain.feature_names:
                    try:
                        val = int(round(float(value)))
                    except Exception:
                        continue
                    if 1 <= val <= 5:
                        cleaned_scores[key_str] = val

        graph_context = ""
        if cleaned_scores:
            graph_context = brain.generate_graph_rag_context(
                "Explain assessment results",
                cleaned_scores
            )

        def format_chart(chart):
            if not isinstance(chart, dict):
                return "N/A"
            labels = chart.get("labels") or []
            datasets = chart.get("datasets") or []
            if not labels or not datasets:
                return "N/A"
            data = datasets[0].get("data") if isinstance(datasets[0], dict) else []
            if not data:
                return "N/A"
            lines = []
            for idx, label in enumerate(labels):
                value = data[idx] if idx < len(data) else "N/A"
                lines.append(f"{label}: {value}")
            return "\n".join(lines) if lines else "N/A"

        def format_performance_table(table):
            if not isinstance(table, dict):
                return "N/A"
            lines = []
            summary = table.get("high_level_summary", {}) or {}
            if summary:
                lines.append(
                    "Summary: strategy={strategy}, current={current}%, target={target}%".format(
                        strategy=summary.get("recommended_strategy", "N/A"),
                        current=summary.get("current_performance", "N/A"),
                        target=summary.get("target_performance", "N/A")
                    )
                )
            primary = table.get("primary_criteria", []) or []
            if primary:
                lines.append("Primary criteria:")
                for row in primary:
                    lines.append(
                        "- {criterion}: weight={weight}, current={current}%, target={target}%, gap={gap}%".format(
                            criterion=row.get("criterion", "N/A"),
                            weight=row.get("weight", "N/A"),
                            current=row.get("real_performance", "N/A"),
                            target=row.get("target_performance", "N/A"),
                            gap=row.get("gap", "N/A")
                        )
                    )
            sub = table.get("sub_criteria", {}) or {}
            if sub:
                lines.append("Sub-criteria:")
                for category, rows in sub.items():
                    lines.append(f"{category}:")
                    for row in rows or []:
                        lines.append(
                            "- {subcategory}: weight={weight}, contribution={contribution}%".format(
                                subcategory=row.get("subcategory", "N/A"),
                                weight=row.get("weight", "N/A"),
                                contribution=row.get("contribution", "N/A")
                            )
                        )
            return "\n".join(lines) if lines else "N/A"

        prediction = payload.get("prediction", "N/A")
        confidence = payload.get("confidence", "N/A")
        strategy = payload.get("strategy", "N/A")
        averages = payload.get("averages", {}) if isinstance(payload, dict) else {}
        recommendations = payload.get("recommendations", []) if isinstance(payload, dict) else []
        if isinstance(recommendations, list):
            recommendations_text = "\n".join(f"- {rec}" for rec in recommendations) if recommendations else "N/A"
        else:
            recommendations_text = str(recommendations) if recommendations is not None else "N/A"
        chart_text = format_chart(payload.get("chart") if isinstance(payload, dict) else None)
        perf_table_text = format_performance_table(payload.get("performance_table") if isinstance(payload, dict) else None)

        actions_lines = []
        if cleaned_scores and hasattr(brain, "csf_level_prescriptions"):
            for factor, current in cleaned_scores.items():
                next_level = min(current + 1, 5)
                prescription = brain.csf_level_prescriptions.get(factor, {}).get(
                    next_level, "No prescription available."
                )
                actions_lines.append(f"{factor}: {current} -> {next_level} | {prescription}")
        actions_text = "\n".join(actions_lines) if actions_lines else "N/A"

        if is_french:
            lang_instruction = "Veuillez repondre en francais."
        else:
            lang_instruction = "Please respond in English."

        prompt = f"""You are a Lean Six Sigma expert. Provide a clear, insight-driven explanation of the assessment results.
- Use the Graph RAG context to extract at least 2 evidence-based insights.
- Interpret the results; do not just restate values.
- Provide 3 to 5 concrete recommendations tied to specific factors and gaps.
- Reference the action table to explain how to move to the next level.

{lang_instruction}

Results:
Predicted performance: {prediction}
Confidence: {confidence}
Recommended strategy: {strategy}
IL average: {averages.get('IL', 'N/A')}
IS average: {averages.get('IS', 'N/A')}
IM average: {averages.get('IM', 'N/A')}
Model recommendations:
{recommendations_text}

CSF profile:
{chart_text}

Performance table:
{perf_table_text}

Action table (current -> next level with prescription):
{actions_text}

Graph RAG context:
{graph_context}
"""

        try:
            response_content = invoke_llm(prompt)
        except Exception as llm_error:
            if use_api:
                if is_french:
                    response_content = f"Erreur modele API: {llm_error}. Verifiez votre cle API et le nom du modele."
                else:
                    response_content = f"API model error: {llm_error}. Please verify your API key and model name."
            else:
                response_content = (
                    "LLM is not configured. Provide CSF values for predictions, or configure "
                    "OLLAMA_BASE_URL or an API key for a supported model."
                )

        return {
            "content": str(response_content).strip(),
            "type": "text",
            "chart": None,
            "dataframe": None
        }

    # Extract scores from both message and sidebar
    extracted_scores = brain.extract_scores_from_text(message)

    if use_sidebar_values and sidebar_scores:
        # Sidebar scores take precedence over extracted scores
        extracted_scores.update(sidebar_scores)

    # Extract scores from conversation history to maintain context
    historical_scores = {}
    for conv_item in conversation_history:
        if isinstance(conv_item, dict) and 'content' in conv_item:
            hist_content = conv_item['content']
            hist_scores = brain.extract_scores_from_text(hist_content)
            historical_scores.update(hist_scores)

            # Also look for CSF patterns in the content more broadly
            # Check if this message contained CSF scores even if extract_scores_from_text missed them
            csf_pattern = r'\b([IM][LS]\d+)\s*(?:=|:|is|are)\s*(\d+)\b'
            csf_matches = re.findall(csf_pattern, hist_content.upper())
            for factor, value in csf_matches:
                try:
                    value_int = int(value)
                    if 1 <= value_int <= 5:
                        historical_scores[factor] = value_int
                except:
                    pass  # Skip if value is not a valid integer
        elif isinstance(conv_item, str):  # In case conversation history is just strings
            hist_scores = brain.extract_scores_from_text(conv_item)
            historical_scores.update(hist_scores)

            # Also look for CSF patterns in the content more broadly
            csf_pattern = r'\b([IM][LS]\d+)\s*(?:=|:|is|are)\s*(\d+)\b'
            csf_matches = re.findall(csf_pattern, conv_item.upper())
            for factor, value in csf_matches:
                try:
                    value_int = int(value)
                    if 1 <= value_int <= 5:
                        historical_scores[factor] = value_int
                except:
                    pass  # Skip if value is not a valid integer

    # Combine historical scores with current extracted scores
    # Current scores take precedence over historical ones
    historical_scores.update(extracted_scores)
    all_available_scores = historical_scores

    # Check if it's an L6S-related query
    # Enhance detection for parameter explanation requests (e.g., "what does IL1 mean?")
    is_l6s_query = brain.is_l6s_related(message)

    # Check if asking about specific parameters (IL1, IS2, M3, etc.)
    param_pattern = r'\b([IM][LS]\d+|M\d+)\b'
    param_matches = re.findall(param_pattern, message.upper())

    # If asking about specific parameters, treat as L6S-related even if is_l6s_related says otherwise
    if not is_l6s_query and param_matches:
        is_l6s_query = True

    if not is_l6s_query:
        response_content = "I'm the L6S (Lean Six Sigma) Expert Advisor. I can help you with Lean, Six Sigma, and performance optimization queries."
        if is_french:
            response_content = "Je suis l'expert-conseil L6S (Lean Six Sigma). Je peux vous aider avec les questions relatives à Lean, Six Sigma et à l'optimisation de la performance."
        response = {
            "content": response_content,
            "type": "text",
            "chart": None,
            "dataframe": None
        }

        # ADD PERFORMANCE CHARTS AND TABLE FOR APPROPRIATE QUERIES
        if any(keyword in message.lower() for keyword in ['performance table', 'analytics', 'breakdown', 'overview', 'analysis', 'performance analytics']):
            try:
                perf_radar_chart = brain.generate_performance_breakdown_chart()
                perf_bar_chart = brain.generate_subcriteria_barchart()

                # Get performance analytics context text
                perf_analytics_context = brain.generate_performance_analytics_context(message)

                if perf_analytics_context:
                    # Append performance analytics to the response content
                    response["content"] += f"\n\n{perf_analytics_context}"

                if perf_radar_chart or perf_bar_chart:
                    # Add chart data to response
                    if perf_radar_chart:
                        if "performance_charts" not in response:
                            response["performance_charts"] = {}
                        response["performance_charts"]["radar_chart"] = perf_radar_chart.to_dict() if hasattr(perf_radar_chart, 'to_dict') else str(perf_radar_chart)
                    if perf_bar_chart:
                        if "performance_charts" not in response:
                            response["performance_charts"] = {}
                        response["performance_charts"]["bar_chart"] = perf_bar_chart.to_dict() if hasattr(perf_bar_chart, 'to_dict') else str(perf_bar_chart)
            except Exception as e:
                print(f"Error in performance analytics: {e}", file=sys.stderr)  # Log error for debugging
                pass  # Ignore errors in chart generation

        return finalize_response(response, is_french, brain)

    # PRIORITY 1: Check for target performance goals FIRST
    target_perf = brain.analyze_goal(message)

    # Also check for goal patterns manually with enhanced patterns
    if not target_perf:
        # Enhanced patterns to catch more natural language
        patterns = [
            r'(?:i\s+)?(?:want|need|would\s+like)\s+(?:a\s+)?(?:performance\s+of\s+)?(\d+)(?:%)?',
            r'(?:achieve|target|get|reach|attain|hit)\s*(\d+)(?:%)?',
            r'(\d+)(?:%)?(?:\s+performance)?',
            r'(?:goal|aim)(?:\s+is|\s+of)?\s*(\d+)(?:%)?',
            r'(?:how\s+about|what\s+about)\s*(\d+)(?:%)?'
        ]

        for pattern in patterns:
            match = re.search(pattern, message_lower, re.IGNORECASE)
            if match:
                perf = float(match.group(1))
                if 30 <= perf <= 100:
                    target_perf = perf
                    break

    # Also check for French goal patterns manually if the built-in function doesn't catch them
    if not target_perf and is_french:
        # Look for French patterns like "atteindre [number]%", "atteindre [number]", etc.
        # Enhanced to handle typos like "e veux" instead of "je veux"
        french_goal_patterns = [
            r'(?:atteindr|atteindre|atteint|atteints|atteinte)\s*(\d+)(?:%)?',  # Includes "atteindr" typo
            r'(?:vouloir|souhaite|veux|veulent| e veux|e veux)\s*(?:atteindr|atteindre|atteint)\s*(\d+)(?:%)?',  # Includes "e veux" typo
            r'(?:comment|comment faire pour)\s*(?:atteindr|atteindre|atteint)\s*(\d+)(?:%)?',
            r'(?:objectif|but|cible)\s*(?:est|de|à)\s*(\d+)(?:%)?',
            r'(?:je veux|je souhaite|on veut|on souhaite|voudrais|vouloir|souhaiter)\s*(?:atteindr|atteindre|atteint)\s*(\d+)(?:%)?'  # More explicit French goal patterns
        ]

        for pattern in french_goal_patterns:
            match = re.search(pattern, message.lower())
            if match:
                perf = float(match.group(1))
                if 30 <= perf <= 100:
                    target_perf = perf
                    break

    # PRIORITY 2: If target_perf is found, process immediately
    if target_perf:
        try:
            # Get reverse recommendation (concrete scores)
            rec = brain.get_reverse_recommendation(target_perf)

            # Validate that rec has the expected structure
            if not rec or 'means' not in rec or not rec['means']:
                raise ValueError("Invalid recommendation result")

            # Get descriptions from the brain instance
            csf_descriptions = getattr(brain, 'CSF_DESCRIPTIONS', {})
            strategy_descriptions = getattr(brain, 'STRATEGY_DESCRIPTIONS', {})

            # Define default CSF descriptions to ensure they're always available
            default_csf_descriptions = {
                'IL1': 'Leadership Engagement (Lean)',
                'IL2': 'Cultural Change (Lean)',
                'IL3': 'Communication (Lean)',
                'IL4': 'Training (Lean)',
                'IL5': 'Tools & Techniques (Lean)',
                'IL6': 'Employee Involvement (Lean)',
                'IL7': 'Expertise & Skills (Lean)',
                'IS1': 'Leadership Engagement (Six Sigma)',
                'IS2': 'Cultural Change (Six Sigma)',
                'IS3': 'Communication (Six Sigma)',
                'IS4': 'Training (Six Sigma)',
                'IS5': 'Tools & Techniques (Six Sigma)',
                'IS6': 'Employee Involvement (Six Sigma)',
                'IS7': 'Expertise & Skills (Six Sigma)',
                'M1': 'Leadership Engagement (Maturity)',
                'M2': 'Cultural Change (Maturity)',
                'M3': 'Communication (Maturity)',
                'M4': 'Training (Maturity)',
                'M5': 'Tools & Techniques (Maturity)',
                'M6': 'Employee Involvement (Maturity)',
                'M7': 'Expertise & Skills (Maturity)'
            }

            # Check if user has provided current scores in this or previous messages
            has_current_scores = bool(all_available_scores)

            # Format response with detailed structure like in neo.py
            if is_french:
                if has_current_scores:
                    response_content = f"""Feuille de route personnalisée vers {target_perf}% de Performance
{rec["message"]}

Stratégie Recommandée: {rec["strategy"]} {strategy_descriptions.get(rec["strategy"], "")}

Performance moyenne attendue: {rec['avg_perf']:.1f}%
                """.strip()
                else:
                    response_content = f"""Feuille de route vers {target_perf}% de Performance
{rec["message"]}

Stratégie Recommandée: {rec["strategy"]} {strategy_descriptions.get(rec["strategy"], "")}

Performance moyenne attendue: {rec['avg_perf']:.1f}%
                """.strip()
            else:
                if has_current_scores:
                    response_content = f"""Personalized roadmap to {target_perf}% Performance
{rec["message"]}

Recommended Strategy: {rec["strategy"]} {strategy_descriptions.get(rec["strategy"], "")}

Expected average performance: {rec['avg_perf']:.1f}%
                """.strip()
                else:
                    response_content = f"""Roadmap to {target_perf}% Performance
{rec["message"]}

Recommended Strategy: {rec["strategy"]} {strategy_descriptions.get(rec["strategy"], "")}

Expected average performance: {rec['avg_perf']:.1f}%
                """.strip()

            # Compare current scores with target scores if current scores are provided
            if has_current_scores:
                if is_french:
                    response_content += f"\n\nComparaison avec vos scores actuels:"
                else:
                    response_content += f"\n\nComparison with your current scores:"

                # Calculate and display improvements needed
                improvements_needed = []
                for factor, current_value in all_available_scores.items():
                    target_value = rec["means"].get(factor)
                    if target_value is not None:
                        target_value = int(min(target_value, 5) if factor.startswith('M') else target_value)  # Cap maturity at 5
                        improvement = target_value - current_value
                        if improvement > 0:
                            if is_french:
                                desc = csf_descriptions.get(factor) or default_csf_descriptions.get(factor) or factor
                            else:
                                desc = default_csf_descriptions.get(factor) or csf_descriptions.get(factor) or factor
                            improvements_needed.append({
                                'factor': factor,
                                'current': current_value,
                                'target': target_value,
                                'improvement': improvement,
                                'description': desc
                            })

                # Sort improvements by the amount of improvement needed (descending)
                improvements_needed.sort(key=lambda x: x['improvement'], reverse=True)

                if improvements_needed:
                    if is_french:
                        response_content += f"\n\nDomaines prioritaires d'amélioration:"
                    else:
                        response_content += f"\n\nPriority improvement areas:"

                    for imp in improvements_needed:
                        if is_french:
                            response_content += f"\n- {imp['factor']} ({imp['description']}): Passer de {imp['current']} à {imp['target']} (amélioration de +{imp['improvement']})"
                        else:
                            response_content += f"\n- {imp['factor']} ({imp['description']}): Increase from {imp['current']} to {imp['target']} (improvement of +{imp['improvement']})"

                        # Add prescriptions for the levels that need to be achieved
                        if hasattr(brain, 'csf_level_prescriptions') and imp['factor'] in brain.csf_level_prescriptions:
                            # Show prescriptions for each level between current and target
                            for level in range(int(imp['current']) + 1, int(imp['target']) + 1):
                                if level in brain.csf_level_prescriptions[imp['factor']]:
                                    prescription = brain.csf_level_prescriptions[imp['factor']][level]
                                    if not is_french:
                                        prescription = translate_text(prescription, "en", brain)
                                    if is_french:
                                        response_content += f"\n  → Action pour niveau {level}: {prescription}"
                                    else:
                                        response_content += f"\n  → Action for level {level}: {prescription}"
                        response_content += "\n"
                else:
                    if is_french:
                        response_content += f"\n\n✅ Vos scores actuels sont déjà proches des valeurs cibles!"
                    else:
                        response_content += f"\n\n✅ Your current scores are already close to the target values!"

            else:
                # Add a note if no CSF values were provided
                if is_french:
                    response_content += f"""

⚠️ Note: Aucune valeur CSF actuelle fournie. Les recommandations ci-dessous sont basées sur les organisations réussies atteignant {target_perf}%+ de performance. Pour obtenir des recommandations personnalisées, veuillez activer 'Saisie manuelle des paramètres' dans la barre latérale ou fournir vos scores CSF actuels (par exemple, 'IL1=3, IS2=4')."""
                else:
                    response_content += f"""

⚠️ Note: No current CSF values provided. The recommendations below are based on successful organizations achieving {target_perf}%+ performance. To get personalized recommendations, please enable 'Manual Parameter Input' in the sidebar or provide your current CSF scores (e.g., 'IL1=3, IS2=4')."""

            if is_french:
                response_content += "\n\nTableau d'actions cible (voir ci-dessous)."
            else:
                response_content += "\n\nTarget action table (see below)."

            action_table_rows = []
            factor_order = (
                [f"IL{i}" for i in range(1, 8)] +
                [f"IS{i}" for i in range(1, 8)] +
                [f"M{i}" for i in range(1, 8)]
            )
            no_prescription = "Aucune prescription disponible." if is_french else "No prescription available."
            for factor in factor_order:
                target_value = rec["means"].get(factor)
                if target_value is None:
                    continue
                target_value = int(min(target_value, 5)) if factor.startswith('M') else int(target_value)
                current_value = all_available_scores.get(factor)
                current_int = int(current_value) if isinstance(current_value, (int, float)) else None
                if is_french:
                    desc = csf_descriptions.get(factor) or default_csf_descriptions.get(factor) or factor
                else:
                    desc = default_csf_descriptions.get(factor) or csf_descriptions.get(factor) or factor
                prescriptions = getattr(brain, 'csf_level_prescriptions', {}).get(factor, {})

                if current_int is None:
                    levels = range(2, 6)
                elif current_int < target_value:
                    levels = range(current_int + 1, target_value + 1)
                else:
                    levels = [target_value]

                for level in levels:
                    if current_int is not None and current_int >= target_value:
                        action = "Maintenir le niveau actuel." if is_french else "Maintain current level."
                    else:
                        action = prescriptions.get(level, no_prescription)
                        if not is_french:
                            action = translate_text(action, "en", brain)
                    action_table_rows.append({
                        "factor": factor,
                        "label": desc,
                        "current": current_int,
                        "target": target_value,
                        "level": level,
                        "action": action
                    })
            # Create detailed CSF data for radar chart visualization with validation
            all_csf_values = {}
            for k, v in rec["means"].items():
                if v is not None:
                    # Cap maturity factors at 5 for consistency
                    if k.startswith('M') and v > 5:
                        all_csf_values[k] = 5
                    else:
                        all_csf_values[k] = int(v) if v is not None else 0

            # Create proper radar chart data for CSF profile
            csf_chart_data = {
                'labels': list(all_csf_values.keys()),
                'datasets': [
                    {
                        'label': f'Profil Cible pour {target_perf}% de Performance' if is_french else f'Target Profile for {target_perf}% Performance',
                        'data': list(all_csf_values.values()),
                        'backgroundColor': 'rgba(75, 192, 192, 0.2)',
                        'borderColor': 'rgba(75, 192, 192, 1)',
                        'borderWidth': 2,
                        'pointBackgroundColor': 'rgba(75, 192, 192, 1)',
                        'pointBorderColor': '#fff',
                        'pointHoverBackgroundColor': '#fff',
                        'pointHoverBorderColor': 'rgba(75, 192, 192, 1)'
                    }
                ]
            }

            # Create performance table for reverse analysis as well
            # Load performance strategy data from JSON to get the structure
            current_perf_value = float(rec['avg_perf'])
            if has_current_scores:
                try:
                    current_vector = []
                    for param in brain.feature_names:
                        if param in all_available_scores:
                            current_vector.append(all_available_scores[param])
                        elif hasattr(brain, 'df') and param in brain.df.columns:
                            current_vector.append(brain.df[param].mean())
                        else:
                            current_vector.append(3.0)
                    current_result = brain.predict_strategy_and_perf(current_vector, all_available_scores)
                    current_prediction = current_result.get('performance')
                    if isinstance(current_prediction, (int, float)):
                        current_perf_value = float(current_prediction)
                except Exception:
                    pass

            performance_table_data = {
                "high_level_summary": {
                    "recommended_strategy": rec["strategy"],
                    "current_performance": current_perf_value,
                    "target_performance": float(target_perf)
                },
                "primary_criteria": [],
                "sub_criteria": {}
            }

            # Load performance strategy data from JSON to get the structure
            perf_data = {}
            try:
                json_path = UNIFIED_JSON_PATH
                if json_path.exists():
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                        perf_data = json_data.get('performance_strategy', {})
            except:
                pass  # If file doesn't exist or can't be loaded, continue with empty perf_data

            breakdown = perf_data.get('breakdown', {})

            # Add primary criteria data based on the target performance
            for category, data in breakdown.items():
                weight = data.get('weight', 0)
                real_performance = data.get('real_performance', 0) * 100  # Convert to percentage
                target_val = data.get('target_performance', 0) * 100  # Convert to percentage
                gap = target_val - real_performance

                readable_category = category.replace('_', ' ').replace('performance', '').strip().title()

                performance_table_data["primary_criteria"].append({
                    "criterion": readable_category,
                    "weight": round(weight, 3),
                    "real_performance": round(real_performance, 1),
                    "target_performance": round(target_val, 1),
                    "gap": round(abs(gap), 1)
                })

            # Add sub-criteria data
            for category, data in breakdown.items():
                readable_category = category.replace('_', ' ').replace('performance', '').strip().title()
                sub_criteria = data.get('sub_criteria', {})

                if sub_criteria:
                    performance_table_data["sub_criteria"][readable_category] = []
                    for sub_name, sub_data in sub_criteria.items():
                        sub_readable = sub_name.replace('_', ' ').title()
                        weight = sub_data.get('weight', 0)
                        contribution = sub_data.get('contribution', 0) * 100  # Convert to percentage

                        performance_table_data["sub_criteria"][readable_category].append({
                            "subcategory": sub_readable,
                            "weight": round(weight, 3),
                            "contribution": round(contribution, 1)
                        })

            # Update response to include chart data for reverse analysis with radar chart
            response = {
                "content": response_content,
                "type": "text_with_chart",  # Changed to include chart for reverse analysis too
                "chart": csf_chart_data,  # Now using proper CSF radar chart
                "dataframe": {
                    "csf_values": all_csf_values,
                    "target_performance": float(target_perf),
                    "achieved_performance": float(rec['avg_perf']),
                    "strategy": rec["strategy"],
                    "sample_size": int(rec['sample_size'])
                },
                "performance_table": performance_table_data,
                "target_action_table": {
                    "title": "Tableau d'actions cible" if is_french else "Target Action Table",
                    "rows": action_table_rows
                }
            }

            return finalize_response(response, is_french, brain)
        except Exception as e:
            # Even on error, provide structured guidance, not LLM fallback
            if is_french:
                response_content = f"Je ne peux pas calculer les paramètres exacts pour {target_perf}%, mais voici une approche générale..."
            else:
                response_content = f"I cannot calculate exact parameters for {target_perf}%, but here's a general approach..."

            response = {
                "content": response_content,
                "type": "text",
                "chart": None,
                "dataframe": None
            }
            return finalize_response(response, is_french, brain)

    # PRIORITY 3: Check for CSF level description queries
    is_csf_query, query_factor, query_level = brain.detect_csf_level_query(message)

    if is_csf_query and query_factor:
        # Get CSF descriptions from the brain instance to ensure consistency
        csf_descriptions = getattr(brain, 'CSF_DESCRIPTIONS', {})

        factor_name = csf_descriptions.get(query_factor, query_factor)

        if query_level:
            description = brain.get_csf_level_description(query_factor, query_level)
            # The description now includes both the description and prescription
            if not is_french:
                description = translate_text(description, "en", brain)

            if is_french:
                response_content = f"{query_factor}: {factor_name}\n\nNiveau {query_level} :\n{description}"
            else:
                # For English, we still show the original French description from the JSON,
                # but with English labels
                response_content = f"{query_factor}: {factor_name}\n\nLevel {query_level}:\n{description}"
        else:
            if is_french:
                response_content = f"{query_factor}: {factor_name}\n\n"
            else:
                response_content = f"{query_factor}: {factor_name}\n\n"

            if query_factor in brain.csf_level_descriptions:
                levels = brain.csf_level_descriptions[query_factor]
                prescriptions = getattr(brain, 'csf_level_prescriptions', {}).get(query_factor, {})

                for level in sorted(levels.keys()):
                    description = levels[level].rstrip('? ').strip() + '?'
                    prescription = prescriptions.get(level, "Aucune prescription disponible." if is_french else "No prescription available.")
                    if not is_french:
                        description = translate_text(description, "en", brain)
                        prescription = translate_text(prescription, "en", brain)

                    if is_french:
                        response_content += f"Niveau {level} : {description}\nPrescription : {prescription}\n\n"
                    else:
                        # For English, translate descriptions and prescriptions
                        response_content += f"Level {level}: {description}\nPrescription: {prescription}\n\n"
            else:
                response_content += f"Aucune description disponible pour {query_factor}." if is_french else f"No descriptions available for {query_factor}."

        response = {
            "content": response_content.strip(),
            "type": "text",
            "chart": None,
            "dataframe": None
        }

        # ADD PERFORMANCE CHARTS AND TABLE FOR APPROPRIATE QUERIES
        if any(keyword in message.lower() for keyword in ['performance table', 'analytics', 'breakdown', 'overview', 'analysis', 'performance analytics']):
            try:
                perf_radar_chart = brain.generate_performance_breakdown_chart()
                perf_bar_chart = brain.generate_subcriteria_barchart()

                # Get performance analytics context text
                perf_analytics_context = brain.generate_performance_analytics_context(message)

                if perf_analytics_context:
                    # Append performance analytics to the response content
                    response["content"] += f"\n\n{perf_analytics_context}"

                if perf_radar_chart or perf_bar_chart:
                    # Add chart data to response
                    if perf_radar_chart:
                        if "performance_charts" not in response:
                            response["performance_charts"] = {}
                        response["performance_charts"]["radar_chart"] = perf_radar_chart.to_dict() if hasattr(perf_radar_chart, 'to_dict') else str(perf_radar_chart)
                    if perf_bar_chart:
                        if "performance_charts" not in response:
                            response["performance_charts"] = {}
                        response["performance_charts"]["bar_chart"] = perf_bar_chart.to_dict() if hasattr(perf_bar_chart, 'to_dict') else str(perf_bar_chart)
            except Exception as e:
                print(f"Error in performance analytics: {e}", file=sys.stderr)  # Log error for debugging
                pass  # Ignore errors in chart generation

        return finalize_response(response, is_french, brain)

    elif extracted_scores:
        # Check if this is actually a parameter explanation request rather than a prediction request
        # If the message is asking about what parameters mean/signify, use the LLM instead of forward prediction
        explanation_patterns = [
            'what.*mean', 'what.*signif', 'what.*stand', 'what.*repres', 'what.*indic',
            'define', 'definition', 'explain', 'explain.*mean', 'explain.*signif',
            'signifie', 'signifie.*quoi', 'défini', 'définition', 'explique',
            'quest.*che', 'cosa.*significa', 'meaning', 'meaning.*of'
        ]

        is_explanation_request = any(pattern in message.lower() for pattern in explanation_patterns)

        if is_explanation_request:
            # This is a parameter explanation request, send to LLM instead of forward prediction
            # General L6S-related query processing (for parameter explanations)
            if brain.llm_available:
                # Get graph-based context for more informed responses
                graph_context = brain.generate_graph_rag_context(message, extracted_scores if extracted_scores else (sidebar_scores if use_sidebar_values else {}))

                # Construct enhanced prompt with system context
                # Get CSF descriptions from the brain instance to ensure consistency
                csf_descriptions = getattr(brain, 'CSF_DESCRIPTIONS', {})

                csf_desc_text = "CSF Descriptions:\n"
                for factor in brain.feature_names:
                    desc = csf_descriptions.get(factor, f"Description for {factor}")
                    csf_desc_text += f"- {factor}: {desc}\n"

                system_context = f"""You are an expert Lean Six Sigma (L6S) advisor.
Database contains {len(brain.df)} L6S implementation cases showing that CSF scores (Critical Success Factors) are INPUT variables that PREDICT performance outcomes.
The 21 CSF scores (IL1-IL7 for Lean, IS1-IS7 for Six Sigma, M1-M7 for Maturity) are ratings that measure organizational maturity in: Leadership, Culture, Communication, Training, Tools, Employee Involvement, and Expertise.
These CSFs are used to PREDICT performance percentage (typically 30-85%) and recommend implementation strategies.
Available strategies: LM then SS (Lean then Six Sigma), SS then LM (Six Sigma then Lean), LM & SS (simultaneous implementation).
IMPORTANT: Do not make up performance calculations. CSF scores are inputs, performance % is the predicted output.

{csf_desc_text}"""

                # Check if the message is asking about specific level definitions
                message_lower = message.lower()

                # More comprehensive detection for CSF level queries - check multiple ways users might ask
                is_csf_description_query = any(word in message_lower for word in ['niveau', 'level', 'description', 'détail', 'explain', 'signification', 'significance', 'signification', 'facteur', 'factor', 'critical success factor', 'means', 'mean', 'what mean by', 'what stands for', 'what signifie', 'explain level', 'describe level', 'level means', 'level signifie']) or \
                                           any(re.search(phrase, message_lower) for phrase in [r'what.*mean.*by', r'what.*stands.*for', r'what.*signifie', r'explain.*level', r'describe.*level', r'level.*means', r'level.*signifie'])

                # Additional check: if any factor is mentioned with 'mean' or 'what', it's likely a description query
                has_factor_and_meaning = any(re.search(rf'what.*mean.*by.*{factor.lower()}', message_lower) or
                                             re.search(rf'{factor.lower()}.*mean', message_lower) or
                                             re.search(rf'what.*{factor.lower()}.*mean', message_lower) for factor in brain.feature_names)

                # Even more aggressive: check if factor and 'mean' or 'level' appear together in the query
                has_factor_and_meaning_aggressive = any(
                    factor.lower() in message_lower and any(word in message_lower for word in ['mean', 'means', 'signifie', 'signification', 'level'])
                    for factor in brain.feature_names
                )

                # Check for CSF level related queries - more comprehensive pattern matching
                has_csf = any(csf in message_lower for csf in ['il1', 'il2', 'il3', 'il4', 'il5', 'il6', 'il7',
                                                               'is1', 'is2', 'is3', 'is4', 'is5', 'is6', 'is7',
                                                               'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7'])
                has_level_indication = (is_csf_description_query or has_factor_and_meaning or has_factor_and_meaning_aggressive or
                                        any(pattern in message_lower for pattern in
                                        ['level', 'signify', 'mean', 'meaning', 'definition', 'represent', 'indicate', 'what']))

                if has_csf and has_level_indication:
                    system_context += "\n\nCSF Level Descriptions: "

                    # Only include relevant level descriptions based on the query
                    # (re module already imported at the top of the file)
                    # Extract CSF from the message with more comprehensive pattern matching
                    csf_pattern = r'\b(i[mls]\d+|m\d+)\b'
                    csf_matches = re.findall(csf_pattern, message_lower.upper())

                    # More robust level extraction considering the enhanced detection above
                    # Extract potential level numbers from context (numbers 1-5 that might be levels)
                    level_numbers = [str(i) for i in range(1, 6)]  # ['1', '2', '3', '4', '5']

                    # Look for specific level patterns more comprehensively
                    all_level_matches = []
                    for level in level_numbers:
                        # Check for patterns like "IL2 level 5", "level 5 of IL2", "IL2=5", etc.
                        patterns_to_check = [
                            rf'level\s+{level}',
                            rf'niveau\s+{level}',
                            rf'[=:\s]+{level}(?!\d)',  # Captures =5, :5, space+5 but not 50
                            rf'{level}\s*(?:level|niveau)?\s*[a-z]*\s*(?=il|is|m)',  # 5 level IL2
                            rf'level.*{level}.*il|level.*{level}.*is|level.*{level}.*m',  # level 5 il2
                        ]

                        if any(re.search(pattern, message_lower) for pattern in patterns_to_check):
                            all_level_matches.append(level)

                    # If no specific level patterns found, at least get explicit numbers from message
                    if not all_level_matches:
                        all_level_matches = re.findall(r'\b(' + '|'.join(level_numbers) + r')\b', message_lower)

                    if hasattr(brain, 'csf_level_descriptions') and csf_matches:
                        for csf_id in set(csf_matches):  # Use set to avoid duplicates
                            if csf_id in brain.csf_level_descriptions:
                                # Include only the relevant levels mentioned in the query
                                if all_level_matches:
                                    for level_str in all_level_matches:
                                        level_int = int(level_str)
                                        if level_int in brain.csf_level_descriptions[csf_id]:
                                            description = brain.csf_level_descriptions[csf_id][level_int]
                                            prescription = getattr(brain, 'csf_level_prescriptions', {}).get(csf_id, {}).get(level_int, "No prescription available.")
                                            system_context += f"- {csf_id} Level {level_int}: {description}\n  Prescription: {prescription}\n"
                                else:
                                    # If no specific level mentioned, include all levels for this CSF
                                    for level_num, description in brain.csf_level_descriptions[csf_id].items():
                                        prescription = getattr(brain, 'csf_level_prescriptions', {}).get(csf_id, {}).get(level_num, "No prescription available.")
                                        system_context += f"- {csf_id} Level {level_num}: {description}\n  Prescription: {prescription}\n"
                    else:
                        system_context += "No detailed level descriptions available in the loaded model."
                else:
                    # Only add the level descriptions for specific CSF level queries to avoid confusion
                    pass

                # Add conversation history if available
                conversation_context = ""
                if conversation_history:
                    conversation_context = "\n\nPREVIOUS CONVERSATION HISTORY:\n"
                    for i, conv_item in enumerate(conversation_history[-5:]):  # Use last 5 exchanges for context
                        role = conv_item.get('role', 'user')
                        content = conv_item.get('content', '')
                        conversation_context += f"[{role.upper()}]: {content}\n"

                # Add graph RAG context
                rag_context = f"\n\nGRAPH RAG CONTEXT from local analysis:\n{graph_context}\n\n" if graph_context else ""

                # Detect if this seems to be a follow-up question about goals/parameters
                follow_up_context = ""
                follow_up_indicators = ['parametres', 'concrets', 'minimales', 'specific', 'exact', 'precis', 'précis',
                                       'what.*parameters', 'which.*parameters', 'needed.*performance', 'need.*achieve',
                                       'quel.*parametre', 'quels.*parametres', 'valeur.*concrete', 'donne.*concrete']

                if any(re.search(phrase, message.lower()) for phrase in follow_up_indicators):
                    follow_up_context = f"""PREVIOUS CONTEXT:
User was asking about achieving a specific performance target (e.g., 80%+ performance).
Current query seems to be a follow-up asking for specific parameter values.
Instead of general advice, provide concrete CSF values (IL1-IL7, IS1-IS7, M1-M7) that research shows are associated with high performance.
For example: IL1=4-5, IS2=4-5, M3=6-7, etc. based on analysis of successful organizations."""

                # Load prompt templates
                prompt_templates = load_prompt_templates()

                # Determine response language
                lang_instruction = "Please respond in French." if is_french else "Please respond in English."

                # Check if user is requesting performance analytics
                wants_performance_table = False
                if prompt_templates:
                    triggers = prompt_templates.get("triggers", {}).get("include_performance_table", {})
                    en_triggers = triggers.get("en", [])
                    fr_triggers = triggers.get("fr", [])

                    all_triggers = en_triggers + fr_triggers
                    message_lower = message.lower()
                    wants_performance_table = any(trigger.lower() in message_lower for trigger in all_triggers)

                # Get performance analytics context if needed
                perf_analytics_context = ""
                if wants_performance_table:
                    perf_analytics_context = brain.generate_performance_analytics_context(message)

                # Use prompt templates if available
                if prompt_templates:
                    # Get the appropriate system prompt based on language
                    system_prompt_key = "fr" if is_french else "en"
                    system_prompt = prompt_templates.get("system_prompts", {}).get(system_prompt_key, {}).get("main", "")

                    # Fill in template variables
                    system_prompt = system_prompt.format(total_samples=len(brain.df) if hasattr(brain, 'df') else 0)

                    # Construct enhanced prompt with system context
                    enhanced_context = f"{system_context}\n\n{rag_context}\n\n"
                    if perf_analytics_context:
                        enhanced_context += f"PERFORMANCE ANALYTICS CONTEXT:\n{perf_analytics_context}\n\n"

                    # Construct final prompt with conversation history if available
                    prompt = f"""{system_prompt}

{enhanced_context}{conversation_context}{follow_up_context}

INSTRUCTIONS: {lang_instruction}
User asked: '{message}'
Provide a concise, accurate answer about L6S topics, referencing the conversation history and graph RAG context when relevant. Do not invent calculations or metrics. If the user is asking for specific parameters related to performance goals, provide concrete CSF values (IL1-IL7, IS1-IS7, M1-M7) - all factors use a 1-5 scale.
IMPORTANT: If CSF Level Descriptions with Prescriptions are provided in the context above, include both the description AND the prescription information in your response. The prescriptions provide specific action items that organizations should take to achieve each level of the CSF."""
                else:
                    # Fallback to original prompt construction
                    prompt = f"""{system_context}{conversation_context}{rag_context}{follow_up_context}

INSTRUCTIONS: {lang_instruction}
User asked: '{message}'
Provide a concise, accurate answer about L6S topics, referencing the conversation history and graph RAG context when relevant. Do not invent calculations or metrics. If the user is asking for specific parameters related to performance goals, provide concrete CSF values (IL1-IL7, IS1-IS7, M1-M7) - all factors use a 1-5 scale.
IMPORTANT: If CSF Level Descriptions with Prescriptions are provided in the context above, include both the description AND the prescription information in your response. The prescriptions provide specific action items that organizations should take to achieve each level of the CSF."""

                try:
                    response_content = invoke_llm(prompt)
                except Exception as llm_error:
                    # Fallback if LLM fails
                    if use_api:
                        if is_french:
                            response_content = f"Erreur modele API: {llm_error}. Verifiez votre cle API et le nom du modele."
                        else:
                            response_content = f"API model error: {llm_error}. Please verify your API key and model name."
                    elif is_french:
                        response_content = "Je peux vous aider avec les recommandations de stratégie Lean Six Sigma. Veuillez fournir des valeurs CSF (comme IL1=4, IS2=3, etc.) ou poser des questions sur les objectifs de performance."
                    else:
                        response_content = "I can help you with Lean Six Sigma strategy recommendations. Please provide CSF values (like IL1=4, IS2=3, etc.) or ask about performance targets."
            else:
                if is_french:
                    response_content = "Je peux vous aider avec les recommandations de stratégie Lean Six Sigma. Veuillez fournir des valeurs CSF (comme IL1=4, IS2=3, etc.) ou poser des questions sur les objectifs de performance."
                else:
                    response_content = "I can help you with Lean Six Sigma strategy recommendations. Please provide CSF values (like IL1=4, IS2=3, etc.) or ask about performance targets."
        else:
            # Forward prediction: predict performance with given parameters
            try:
                # Check if user is requesting comprehensive analysis or performance analytics
                message_lower = message.lower()

                # Load prompt templates to get triggers
                prompt_templates = load_prompt_templates()

                wants_performance_table = False
                if prompt_templates:
                    triggers = prompt_templates.get("triggers", {}).get("include_performance_table", {})
                    en_triggers = triggers.get("en", [])
                    fr_triggers = triggers.get("fr", [])

                    all_triggers = en_triggers + fr_triggers
                    wants_performance_table = any(trigger.lower() in message_lower for trigger in all_triggers)

                # Also check for comprehensive analysis keywords
                comprehensive_analysis_keywords = [
                    'comprehensive analysis', 'comprehensive analyse', 'detailed analysis', 'detailed analyse',
                    'full analysis', 'complete analysis', 'complete analyse', 'thorough analysis', 'thorough analyse',
                    'comprehensive review', 'detailed review', 'full review', 'complete review',
                    'gap analysis', 'gap analyse', 'performance breakdown', 'performance table',
                    'comprehensive', 'complete', 'detailed', 'thorough', 'full', 'analysis', 'analyse'
                ]

                is_comprehensive_request = any(keyword in message_lower for keyword in comprehensive_analysis_keywords)

                # ALWAYS generate performance table when CSF scores are provided (this is the key fix)
                has_csf_scores = bool(extracted_scores)
                should_generate_performance_table = has_csf_scores or wants_performance_table or is_comprehensive_request

                # Use the predict_strategy_and_perf method as it's the complete prediction function
                complete_vector = []
                for param in brain.feature_names:
                    if param in extracted_scores:
                        complete_vector.append(extracted_scores[param])
                    else:
                        # Use mean imputation for missing values
                        complete_vector.append(brain.df[param].mean())

                result = brain.predict_strategy_and_perf(complete_vector, extracted_scores)

                prediction = result.get('performance', 'N/A')
                strategy = result.get('strategy', 'N/A')
                confidence = result.get('confidence', 'N/A')
                similar_cases_list = result.get('similar_cases', [])

                # Build response content with CSF descriptions and prescriptions
                # Get CSF descriptions from the brain instance
                csf_descriptions = getattr(brain, 'CSF_DESCRIPTIONS', {})

                # Define default CSF descriptions in case they're not available in this scope
                local_default_csf_descriptions = {
                    'IL1': 'Leadership Engagement (Lean)',
                    'IL2': 'Cultural Change (Lean)',
                    'IL3': 'Communication (Lean)',
                    'IL4': 'Training (Lean)',
                    'IL5': 'Tools & Techniques (Lean)',
                    'IL6': 'Employee Involvement (Lean)',
                    'IL7': 'Expertise & Skills (Lean)',
                    'IS1': 'Leadership Engagement (Six Sigma)',
                    'IS2': 'Cultural Change (Six Sigma)',
                    'IS3': 'Communication (Six Sigma)',
                    'IS4': 'Training (Six Sigma)',
                    'IS5': 'Tools & Techniques (Six Sigma)',
                    'IS6': 'Employee Involvement (Six Sigma)',
                    'IS7': 'Expertise & Skills (Six Sigma)',
                    'M1': 'Leadership Engagement (Maturity)',
                    'M2': 'Cultural Change (Maturity)',
                    'M3': 'Communication (Maturity)',
                    'M4': 'Training (Maturity)',
                    'M5': 'Tools & Techniques (Maturity)',
                    'M6': 'Employee Involvement (Maturity)',
                    'M7': 'Expertise & Skills (Maturity)'
                }

                # Merge brain's CSF descriptions with defaults to ensure all descriptions are available
                for key, default_desc in local_default_csf_descriptions.items():
                    if key not in csf_descriptions:
                        csf_descriptions[key] = default_desc

                # Format scores with descriptions and prescriptions - each on a new line for better readability
                scores_with_descriptions = []

                for k, v in extracted_scores.items():
                    # Use description from brain if available, otherwise use default, otherwise use the key
                    desc = csf_descriptions.get(k) or local_default_csf_descriptions.get(k) or k
                    scores_with_descriptions.append(f"{k} ({desc}): {v}")

                    # Add prescription for this level if available
                    if hasattr(brain, 'csf_level_prescriptions') and k in brain.csf_level_prescriptions:
                        prescription = brain.csf_level_prescriptions[k].get(v, "No prescription available")
                        scores_with_descriptions.append(f"  → Action: {prescription}")
                    scores_with_descriptions.append("")  # Add blank line for better readability

                scores_str = "\n".join(scores_with_descriptions)

                if is_french:
                    response_content = f"""Avec les paramètres:
{scores_str}

Performance prédite: {prediction:.2f}%
Stratégie recommandée: {strategy}
Niveau de confiance: {confidence:.1f}%
"""
                else:
                    response_content = f"""With parameters:
{scores_str}

Predicted performance: {prediction:.2f}%
Recommended strategy: {strategy}
Confidence level: {confidence:.1f}%
"""

                # Add personalized performance analytics table if requested
                if wants_performance_table or is_comprehensive_request:
                    # Create personalized performance analytics based on user's specific CSF scores
                    # This will calculate performance metrics based on the user's input and weights

                    # Load the performance strategy data from the brain's JSON file
                    perf_data = {}
                    try:
                        # Use the brain's method to access the JSON data
                        json_path = UNIFIED_JSON_PATH
                        if json_path.exists():
                            with open(json_path, 'r', encoding='utf-8') as f:
                                json_data = json.load(f)
                                perf_data = json_data.get('performance_strategy', {})
                    except:
                        pass  # If file doesn't exist or can't be loaded, continue with empty perf_data

                    # Create personalized performance analytics based on user's scores
                    recommended_strategy = strategy  # Use the predicted strategy
                    current_performance = prediction  # Use the predicted performance
                    target_performance = 80.0  # Default target, could be adjusted based on user goals

                    # Calculate personalized performance criteria based on user's CSF scores
                    # For now, we'll use a simplified approach based on the breakdown structure from the JSON
                    breakdown = perf_data.get('breakdown', {})

                    # Create personalized criteria breakdown
                    criteria_breakdown = []
                    for category, data in breakdown.items():
                        # Calculate a personalized value based on user's CSF scores
                        # This is a simplified calculation - in a real implementation,
                        # this would use more sophisticated algorithms based on the user's specific scores
                        weight = data.get('weight', 0)
                        # Simplified calculation: scale the real performance based on how the user's scores compare to max possible
                        user_avg_score = sum(extracted_scores.values()) / len(extracted_scores) if extracted_scores else 3.0
                        # Scale performance based on user's average score (1-5 scale)
                        scaled_performance = (user_avg_score / 5.0) * 100 * weight
                        target_val = data.get('target_performance', 0) * 100  # Convert to percentage
                        gap = target_val - scaled_performance

                        readable_category = category.replace('_', ' ').replace('performance', '').strip().title()
                        criteria_breakdown.append(
                            f"{readable_category}: [{weight:.3f} weight] | {scaled_performance:.1f}% Real vs. {target_val:.1f}% Target (Gap: {abs(gap):.1f}%)"
                        )

                    # Create personalized sub-criteria breakdown
                    sub_criteria_breakdown = []
                    for category, data in breakdown.items():
                        readable_category = category.replace('_', ' ').replace('performance', '').strip().title()
                        sub_criteria = data.get('sub_criteria', {})

                        if sub_criteria:
                            sub_criteria_breakdown.append(f"**{readable_category} Sub-criteria:**")
                            for sub_name, sub_data in sub_criteria.items():
                                sub_readable = sub_name.replace('_', ' ').title()
                                weight = sub_data.get('weight', 0)
                                # Calculate contribution based on user's scores
                                contribution = (sum(extracted_scores.values()) / len(extracted_scores) if extracted_scores else 3.0) * weight * 10  # Simplified calculation
                                sub_criteria_breakdown.append(f"  - {sub_readable}: [{weight:.3f} weight] | {contribution:.1f}% contribution")

                    # Format the personalized analytics text
                    if is_french:
                        perf_analytics_text = f"""
## Tableau d'Analyse de Performance Personnalisé

**Stratégie Recommandée**: {recommended_strategy}
**Performance Estimée Actuelle**: {current_performance:.1f}%
**Performance Cible**: {target_performance:.1f}%

### Critères de Performance Principaux
{chr(10).join(criteria_breakdown)}

### Détail des Sous-Critères
{chr(10).join(sub_criteria_breakdown)}
                        """
                    else:
                        perf_analytics_text = f"""
## Personalized Performance Analysis Table

**Recommended Strategy**: {recommended_strategy}
**Current Estimated Performance**: {current_performance:.1f}%
**Target Performance**: {target_performance:.1f}%

### Primary Performance Criteria
{chr(10).join(criteria_breakdown)}

### Detailed Sub-Criteria Breakdown
{chr(10).join(sub_criteria_breakdown)}
                        """
                    # Don't add performance table to content since it will be sent as structured data
                    # response_content += f"\n\n{perf_analytics_text}"

                # Create proper radar chart for CSF profile with consistent scale
                all_csf_values = {}
                for feat in brain.feature_names:
                    if feat in extracted_scores:
                        # Cap maturity factors at 5 for consistency
                        if feat.startswith('M') and extracted_scores[feat] > 5:
                            all_csf_values[feat] = 5
                        else:
                            all_csf_values[feat] = extracted_scores[feat]
                    elif hasattr(brain, 'df') and feat in brain.df.columns:
                        value = brain.df[feat].mean()
                        # Cap maturity factors at 5 for consistency
                        if feat.startswith('M') and value > 5:
                            all_csf_values[feat] = 5
                        else:
                            all_csf_values[feat] = value

                # Create radar chart data for CSF profile
                csf_chart_data = {
                    'labels': list(all_csf_values.keys()),
                    'datasets': [
                        {
                            'label': 'Votre Profil CSF' if is_french else 'Your CSF Profile',
                            'data': list(all_csf_values.values()),
                            'backgroundColor': 'rgba(54, 162, 235, 0.2)',
                            'borderColor': 'rgba(54, 162, 235, 1)',
                            'borderWidth': 2,
                            'pointBackgroundColor': 'rgba(54, 162, 235, 1)',
                            'pointBorderColor': '#fff',
                            'pointHoverBackgroundColor': '#fff',
                            'pointHoverBorderColor': 'rgba(54, 162, 235, 1)'
                        }
                    ]
                }

                # If performance table was requested, add structured performance table data to the response
                if should_generate_performance_table:
                    # Create structured data for the performance table to be sent to the frontend
                    performance_table_data = {
                        "high_level_summary": {
                            "recommended_strategy": strategy,
                            "current_performance": round(prediction, 1),
                            "target_performance": 80.0  # Default target, could be adjusted based on user goals
                        },
                        "primary_criteria": [],
                        "sub_criteria": {}
                    }

                    # Load performance strategy data from JSON to get the structure
                    perf_data = {}
                    try:
                        json_path = UNIFIED_JSON_PATH
                        if json_path.exists():
                            with open(json_path, 'r', encoding='utf-8') as f:
                                json_data = json.load(f)
                                perf_data = json_data.get('performance_strategy', {})
                    except:
                        pass  # If file doesn't exist or can't be loaded, continue with empty perf_data

                    breakdown = perf_data.get('breakdown', {})

                    # Add primary criteria data based on user's scores
                    for category, data in breakdown.items():
                        weight = data.get('weight', 0)
                        # Use the user's actual predicted performance as the baseline
                        # The real performance for each category should be the user's overall performance * the weight of this category
                        # This maintains the proper relationship: weight * overall performance = contribution to this category
                        if not isinstance(prediction, (int, float)):
                            # If prediction is not a valid number, use a placeholder or skip
                            continue
                        user_overall_performance = prediction

                        # Calculate the real performance contribution for this category
                        # This should be: overall performance * weight of this category
                        scaled_performance = user_overall_performance * weight

                        target_val = data.get('target_performance', 0) * 100  # Convert to percentage
                        gap = target_val - scaled_performance

                        readable_category = category.replace('_', ' ').replace('performance', '').strip().title()

                        performance_table_data["primary_criteria"].append({
                            "criterion": readable_category,
                            "weight": round(weight, 3),
                            "real_performance": round(scaled_performance, 1),
                            "target_performance": round(target_val, 1),
                            "gap": round(abs(gap), 1)
                        })

                    # Add sub-criteria data
                    for category, data in breakdown.items():
                        readable_category = category.replace('_', ' ').replace('performance', '').strip().title()
                        sub_criteria = data.get('sub_criteria', {})

                        if sub_criteria:
                            performance_table_data["sub_criteria"][readable_category] = []
                            for sub_name, sub_data in sub_criteria.items():
                                sub_readable = sub_name.replace('_', ' ').title()
                                weight = sub_data.get('weight', 0)
                                # Calculate contribution based on user's actual performance
                                # The contribution should be proportional to the user's overall performance and the weight of this sub-criterion
                                if not isinstance(prediction, (int, float)):
                                    # If prediction is not a valid number, skip this sub-criterion to avoid faulty output
                                    continue
                                user_overall_performance = prediction

                                # Calculate contribution: overall performance * sub-criterion weight
                                # This maintains the proportional relationship
                                contribution = user_overall_performance * weight

                                performance_table_data["sub_criteria"][readable_category].append({
                                    "subcategory": sub_readable,
                                    "weight": round(weight, 3),
                                    "contribution": round(contribution, 1)
                                })

                    # Update response to include performance table data
                    # Don't add performance table to content since it will be rendered as a separate UI component
                    response = {
                        "content": response_content.strip(),  # Only the basic content without performance table text
                        "type": "text_with_chart",  # Changed to include chart
                        "chart": csf_chart_data,  # Now using proper CSF radar chart
                        "dataframe": {
                            "csf_values": all_csf_values,
                            "prediction": float(prediction) if isinstance(prediction, (int, float)) else 0,
                            "strategy": strategy,
                            "confidence": float(confidence) if isinstance(confidence, (int, float)) else 0
                        },
                        "performance_table": performance_table_data  # Add the performance table data
                    }
                else:
                    # Regular response without performance table data
                    response = {
                        "content": response_content.strip(),
                        "type": "text_with_chart",  # Changed to include chart
                        "chart": csf_chart_data,  # Now using proper CSF radar chart
                        "dataframe": {
                            "csf_values": all_csf_values,
                            "prediction": float(prediction) if isinstance(prediction, (int, float)) else 0,
                            "strategy": strategy,
                            "confidence": float(confidence) if isinstance(confidence, (int, float)) else 0
                        }
                    }

                return finalize_response(response, is_french, brain)

            except Exception as e:
                if is_french:
                    response_content = f"Erreur dans la prédiction: {str(e)}"
                else:
                    response_content = f"Error in prediction: {str(e)}"

                response = {
                    "content": response_content,
                    "type": "text",
                    "chart": None,
                    "dataframe": None
                }
                return finalize_response(response, is_french, brain)
    else:
        # General L6S-related query processing
        if brain.llm_available:
            # Get graph-based context for more informed responses
            graph_context = brain.generate_graph_rag_context(message, extracted_scores if extracted_scores else (sidebar_scores if use_sidebar_values else {}))

            # Check if this looks like a follow-up to a performance goal question, even without conversation history
            current_message_lower = message.lower()
            is_follow_up_to_goal = any(indicator in current_message_lower for indicator in
                                      ['parametres', 'concrets', 'minimales', 'specific', 'exact', 'precis', 'précis',
                                       'quelle', 'quels', 'valeur', 'donner', 'moyen de', 'avec quoi'])

            # If it looks like a follow-up to a goal question, add context to the LLM
            if is_follow_up_to_goal:
                # Even without explicit conversation history, if the query has goal-related follow-up language,
                # we can infer the user wants specific parameters
                # Get CSF descriptions from the brain instance to ensure consistency
                csf_descriptions = getattr(brain, 'CSF_DESCRIPTIONS', {})

                csf_desc_text = "CSF Descriptions:\n"
                for factor in brain.feature_names:
                    desc = csf_descriptions.get(factor, f"Description for {factor}")
                    csf_desc_text += f"- {factor}: {desc}\n"

                system_context = f"""You are an expert Lean Six Sigma (L6S) advisor.
Database contains {len(brain.df)} L6S implementation cases showing that CSF scores (Critical Success Factors) are INPUT variables that PREDICT performance outcomes.
The 21 CSF scores (IL1-IL7 for Lean, IS1-IS7 for Six Sigma, M1-M7 for Maturity) are ratings that measure organizational maturity in: Leadership, Culture, Communication, Training, Tools, Employee Involvement, and Expertise.
These CSFs are used to PREDICT performance percentage (typically 30-85%) and recommend implementation strategies.
Available strategies: LM then SS (Lean then Six Sigma), SS then LM (Six Sigma then Lean), LM & SS (simultaneous implementation).
IMPORTANT: Do not make up performance calculations. CSF scores are inputs, performance % is the predicted output.

{csf_desc_text}

CURRENT CONTEXT: User appears to be asking for specific, concrete parameter values related to performance goals.
Provide specific CSF parameter values (IL1-IL7, IS1-IS7, M1-M7) that research shows are associated with high performance.
For example: IL1=4-5, IS2=4-5, M3=4-5, etc. based on analysis of successful organizations - all factors use a 1-5 scale."""
            else:
                # Construct enhanced prompt with system context (original)
                # Get CSF descriptions from the brain instance to ensure consistency
                csf_descriptions = getattr(brain, 'CSF_DESCRIPTIONS', {})

                csf_desc_text = "CSF Descriptions:\n"
                for factor in brain.feature_names:
                    desc = csf_descriptions.get(factor, f"Description for {factor}")
                    csf_desc_text += f"- {factor}: {desc}\n"

                system_context = f"""You are an expert Lean Six Sigma (L6S) advisor.
Database contains {len(brain.df)} L6S implementation cases showing that CSF scores (Critical Success Factors) are INPUT variables that PREDICT performance outcomes.
The 21 CSF scores (IL1-IL7 for Lean, IS1-IS7 for Six Sigma, M1-M7 for Maturity) are ratings that measure organizational maturity in: Leadership, Culture, Communication, Training, Tools, Employee Involvement, and Expertise.
These CSFs are used to PREDICT performance percentage (typically 30-85%) and recommend implementation strategies.
Available strategies: LM then SS (Lean then Six Sigma), SS then LM (Six Sigma then Lean), LM & SS (simultaneous implementation).
IMPORTANT: Do not make up performance calculations. CSF scores are inputs, performance % is the predicted output.

{csf_desc_text}"""

                # Check if the message is asking about specific level definitions
                message_lower = message.lower()

                # More comprehensive detection for CSF level queries - check multiple ways users might ask
                is_csf_description_query = any(word in message_lower for word in ['niveau', 'level', 'description', 'détail', 'explain', 'signification', 'significance', 'signification', 'facteur', 'factor', 'critical success factor', 'means', 'mean', 'what mean by', 'what stands for', 'what signifie', 'explain level', 'describe level', 'level means', 'level signifie']) or \
                                           any(re.search(phrase, message_lower) for phrase in [r'what.*mean.*by', r'what.*stands.*for', r'what.*signifie', r'explain.*level', r'describe.*level', r'level.*means', r'level.*signifie'])

                # Additional check: if any factor is mentioned with 'mean' or 'what', it's likely a description query
                has_factor_and_meaning = any(re.search(rf'what.*mean.*by.*{factor.lower()}', message_lower) or
                                             re.search(rf'{factor.lower()}.*mean', message_lower) or
                                             re.search(rf'what.*{factor.lower()}.*mean', message_lower) for factor in brain.feature_names)

                # Even more aggressive: check if factor and 'mean' or 'level' appear together in the query
                has_factor_and_meaning_aggressive = any(
                    factor.lower() in message_lower and any(word in message_lower for word in ['mean', 'means', 'signifie', 'signification', 'level'])
                    for factor in brain.feature_names
                )

                # Check for CSF level related queries - more comprehensive pattern matching
                has_csf = any(csf in message_lower for csf in ['il1', 'il2', 'il3', 'il4', 'il5', 'il6', 'il7',
                                                               'is1', 'is2', 'is3', 'is4', 'is5', 'is6', 'is7',
                                                               'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7'])
                has_level_indication = (is_csf_description_query or has_factor_and_meaning or has_factor_and_meaning_aggressive or
                                        any(pattern in message_lower for pattern in
                                        ['level', 'signify', 'mean', 'meaning', 'definition', 'represent', 'indicate', 'what']))

                if has_csf and has_level_indication:
                    system_context += "\n\nCSF Level Descriptions: "

                    # Only include relevant level descriptions based on the query
                    # (re module already imported at the top of the file)
                    # Extract CSF from the message with more comprehensive pattern matching
                    csf_pattern = r'\b(i[mls]\d+|m\d+)\b'
                    csf_matches = re.findall(csf_pattern, message_lower.upper())

                    # More robust level extraction considering the enhanced detection above
                    # Extract potential level numbers from context (numbers 1-5 that might be levels)
                    level_numbers = [str(i) for i in range(1, 6)]  # ['1', '2', '3', '4', '5']

                    # Look for specific level patterns more comprehensively
                    all_level_matches = []
                    for level in level_numbers:
                        # Check for patterns like "IL2 level 5", "level 5 of IL2", "IL2=5", etc.
                        patterns_to_check = [
                            rf'level\s+{level}',
                            rf'niveau\s+{level}',
                            rf'[=:\s]+{level}(?!\d)',  # Captures =5, :5, space+5 but not 50
                            rf'{level}\s*(?:level|niveau)?\s*[a-z]*\s*(?=il|is|m)',  # 5 level IL2
                            rf'level.*{level}.*il|level.*{level}.*is|level.*{level}.*m',  # level 5 il2
                        ]

                        if any(re.search(pattern, message_lower) for pattern in patterns_to_check):
                            all_level_matches.append(level)

                    # If no specific level patterns found, at least get explicit numbers from message
                    if not all_level_matches:
                        all_level_matches = re.findall(r'\b(' + '|'.join(level_numbers) + r')\b', message_lower)

                    if hasattr(brain, 'csf_level_descriptions') and csf_matches:
                        for csf_id in set(csf_matches):  # Use set to avoid duplicates
                            if csf_id in brain.csf_level_descriptions:
                                # Include only the relevant levels mentioned in the query
                                if all_level_matches:
                                    for level_str in all_level_matches:
                                        level_int = int(level_str)
                                        if level_int in brain.csf_level_descriptions[csf_id]:
                                            description = brain.csf_level_descriptions[csf_id][level_int]
                                            prescription = getattr(brain, 'csf_level_prescriptions', {}).get(csf_id, {}).get(level_int, "No prescription available.")
                                            system_context += f"- {csf_id} Level {level_int}: {description}\n  Prescription: {prescription}\n"
                                else:
                                    # If no specific level mentioned, include all levels for this CSF
                                    for level_num, description in brain.csf_level_descriptions[csf_id].items():
                                        prescription = getattr(brain, 'csf_level_prescriptions', {}).get(csf_id, {}).get(level_num, "No prescription available.")
                                        system_context += f"- {csf_id} Level {level_num}: {description}\n  Prescription: {prescription}\n"
                    else:
                        system_context += "No detailed level descriptions available in the loaded model."
                else:
                    # Only add the level descriptions for specific CSF level queries to avoid confusion
                    pass

            # Add conversation history if available
            conversation_context = ""
            if conversation_history:
                conversation_context = "\n\nPREVIOUS CONVERSATION HISTORY:\n"
                for i, conv_item in enumerate(conversation_history[-5:]):  # Use last 5 exchanges for context
                    role = conv_item.get('role', 'user')
                    content = conv_item.get('content', '')
                    conversation_context += f"[{role.upper()}]: {content}\n"

            # Add graph RAG context
            rag_context = f"\n\nGRAPH RAG CONTEXT from local analysis:\n{graph_context}\n\n" if graph_context else ""

            # Determine response language
            lang_instruction = "Please respond in French." if is_french else "Please respond in English."

            # Detect if this seems to be a follow-up question about goals/parameters
            follow_up_context = ""
            follow_up_indicators = ['parametres', 'concrets', 'minimales', 'specific', 'exact', 'precis', 'précis',
                                   'what.*parameters', 'which.*parameters', 'needed.*performance', 'need.*achieve',
                                   'quel.*parametre', 'quels.*parametres', 'valeur.*concrete', 'donne.*concrete']

            if any(re.search(phrase, message.lower()) for phrase in follow_up_indicators):
                follow_up_context = f"""PREVIOUS CONTEXT:
User was asking about achieving a specific performance target (e.g., 80%+ performance).
Current query seems to be a follow-up asking for specific parameter values.
Instead of general advice, provide concrete CSF values (IL1-IL7, IS1-IS7, M1-M7) that research shows are associated with high performance.
For example: IL1=4-5, IS2=4-5, M3=4-5, etc. based on analysis of successful organizations - all factors use a 1-5 scale."""

            # Load prompt templates
            prompt_templates = load_prompt_templates()

            # Check if user is requesting performance analytics
            wants_performance_table = False
            if prompt_templates:
                triggers = prompt_templates.get("triggers", {}).get("include_performance_table", {})
                en_triggers = triggers.get("en", [])
                fr_triggers = triggers.get("fr", [])

                all_triggers = en_triggers + fr_triggers
                message_lower = message.lower()
                wants_performance_table = any(trigger.lower() in message_lower for trigger in all_triggers)

            # Get performance analytics context if needed
            perf_analytics_context = ""
            if wants_performance_table:
                perf_analytics_context = brain.generate_performance_analytics_context(message)

            # Use prompt templates if available
            if prompt_templates:
                # Get the appropriate system prompt based on language
                system_prompt_key = "fr" if is_french else "en"
                system_prompt = prompt_templates.get("system_prompts", {}).get(system_prompt_key, {}).get("main", "")

                # Fill in template variables
                system_prompt = system_prompt.format(total_samples=len(brain.df) if hasattr(brain, 'df') else 0)

                # Construct enhanced prompt with system context
                enhanced_context = f"{system_context}\n\n{rag_context}\n\n"
                if perf_analytics_context:
                    enhanced_context += f"PERFORMANCE ANALYTICS CONTEXT:\n{perf_analytics_context}\n\n"

                # Construct final prompt with conversation history if available
                prompt = f"""{system_prompt}

{enhanced_context}{conversation_context}{follow_up_context}

INSTRUCTIONS: {lang_instruction}
User asked: '{message}'
Provide a concise, accurate answer about L6S topics, referencing the conversation history and graph RAG context when relevant. Do not invent calculations or metrics. If the user is asking for specific parameters related to performance goals, provide concrete CSF values (IL1-IL7, IS1-IS7, M1-M7) - all factors use a 1-5 scale.
IMPORTANT: If CSF Level Descriptions with Prescriptions are provided in the context above, include both the description AND the prescription information in your response. The prescriptions provide specific action items that organizations should take to achieve each level of the CSF."""
            else:
                # Fallback to original prompt construction
                prompt = f"""{system_context}{conversation_context}{rag_context}{follow_up_context}

INSTRUCTIONS: {lang_instruction}
User asked: '{message}'
Provide a concise, accurate answer about L6S topics, referencing the conversation history and graph RAG context when relevant. Do not invent calculations or metrics. If the user is asking for specific parameters related to performance goals, provide concrete CSF values (IL1-IL7, IS1-IS7, M1-M7) - all factors use a 1-5 scale.
IMPORTANT: If CSF Level Descriptions with Prescriptions are provided in the context above, include both the description AND the prescription information in your response. The prescriptions provide specific action items that organizations should take to achieve each level of the CSF."""

            try:
                response_content = invoke_llm(prompt)
            except Exception as llm_error:
                # Fallback if LLM fails
                if use_api:
                    if is_french:
                        response_content = f"Erreur modele API: {llm_error}. Verifiez votre cle API et le nom du modele."
                    else:
                        response_content = f"API model error: {llm_error}. Please verify your API key and model name."
                elif is_french:
                    response_content = "Je peux vous aider avec les recommandations de stratégie Lean Six Sigma. Veuillez fournir des valeurs CSF (comme IL1=4, IS2=3, etc.) ou poser des questions sur les objectifs de performance."
                else:
                    response_content = "I can help you with Lean Six Sigma strategy recommendations. Please provide CSF values (like IL1=4, IS2=3, etc.) or ask about performance targets."
        else:
            if is_french:
                response_content = "Je peux vous aider avec les recommandations de stratégie Lean Six Sigma. Veuillez fournir des valeurs CSF (comme IL1=4, IS2=3, etc.) ou poser des questions sur les objectifs de performance."
            else:
                response_content = "I can help you with Lean Six Sigma strategy recommendations. Please provide CSF values (like IL1=4, IS2=3, etc.) or ask about performance targets."

        # Format and return response for general cases
        response = {
            "content": response_content.strip(),
            "type": "text",  # Default type without chart
            "chart": None,
            "dataframe": None
        }

        return finalize_response(response, is_french, brain)


def main():
    """Main entry point for the API"""
    parser = argparse.ArgumentParser(description='L6S Chat API')
    parser.add_argument('--message', type=str, required=True, help='User message to process')
    parser.add_argument('--use-sidebar-values', type=str, required=False, help='Whether to use sidebar values')
    parser.add_argument('--sidebar-scores', type=str, required=False, help='Sidebar scores as JSON string')
    parser.add_argument('--conversation-history', type=str, required=False, help='Conversation history as JSON string')
    parser.add_argument('--local-model', type=str, default='llama3:8b', help='Local model name')
    parser.add_argument('--api-model', type=str, default='gemini-pro', help='API model name')
    parser.add_argument('--temperature', type=float, default=0.3, help='Model temperature')
    parser.add_argument('--api-key', type=str, required=False, help='API key for the model')
    parser.add_argument('--use-api', type=str, default='false', help='Whether to use API model (true/false)')
    parser.add_argument('--language', type=str, required=False, help='Preferred response language (en/fr)')

    args = parser.parse_args()

    # Parse boolean value for use_api
    use_api = args.use_api.lower() in ['true', '1', 'yes', 'on']

    # Prepare model configuration
    model_config = {
        'local_model': args.local_model,
        'api_model': args.api_model,
        'temperature': args.temperature,
        'api_key': args.api_key,
        'use_api': use_api
    }

    try:
        # Process the request using cached brain instance
        response = process_request(
            args.message,
            args.use_sidebar_values,
            args.sidebar_scores,
            args.conversation_history,
            model_config,
            args.language
        )

        # Output the response as JSON
        print(json.dumps(response, default=_json_default))

    except Exception as e:
        # Error handling
        message_lower = args.message.lower()
        requested_lang = _normalize_lang(args.language) if args.language else None
        if requested_lang == "fr":
            is_french = True
        elif requested_lang == "en":
            is_french = False
        else:
            is_french = any(indicator in message_lower for indicator in ['bonjour', 'merci', 's\'il vous pla??t', 'svp', 'comment', 'quelle', 'quelle est', 'aidez', 'sujet', 'strat??gie', 'performance', 'objectif', 'je veux', 'voudrais'])
        if is_french:
            error_response = {
                "content": f"Erreur lors du traitement de la requête: {str(e)}",
                "type": "text",
                "chart": None,
                "dataframe": None
            }
        else:
            error_response = {
                "content": f"Error processing request: {str(e)}",
                "type": "text",
                "chart": None,
                "dataframe": None
            }
        print(json.dumps(error_response, default=_json_default))
        sys.exit(1)


if __name__ == "__main__":
    main()
