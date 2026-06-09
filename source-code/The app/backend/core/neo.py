"""
INTELLIGENT L6S ADVISOR - CHATBOT EDITION
Features: 
- Native Chat Interface with explicit parameter mode control
- Hybrid NLP: Extracts values from text with optional sidebar input
- Dual ML Engine: Performance Prediction (Forward) & Goal Optimization (Reverse)
- Robust Strategy Recommendation using Random Forest & k-NN
- Strict L6S domain focus with intelligent partial parameter handling
- Conversational memory and typo tolerance
"""

# Import Streamlit conditionally - only when running in Streamlit context
try:
    import streamlit as st
except ImportError:
    # When running in API mode, use a dummy object to avoid errors
    class DummyStreamlit:
        def __getattr__(self, name):
            # Return a function that can be called but does nothing
            def dummy_func(*args, **kwargs):
                pass
            return dummy_func

    st = DummyStreamlit()

import pandas as pd
import numpy as np
import re
import json
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from collections import Counter
from difflib import SequenceMatcher
import pickle
from pathlib import Path

# Import the new LLM manager
try:
    from l6s_llm_manager import LLMManager
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from l6s_llm_manager import LLMManager

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
# Only run Streamlit config if in Streamlit environment
try:
    st.set_page_config(page_title="L6S Expert Advisor", layout="wide", page_icon="📊")
except:
    # When running as API, Streamlit config is not available
    pass

# Path to store the trained graph locally
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
GRAPH_CACHE_PATH = DATA_DIR / "l6s_graph_cache_v2.pkl"

# Path to unified data JSON file
JSON_DATA_PATH = DATA_DIR / "unified_lss_data_final.json"

# CSF Descriptions
CSF_DESCRIPTIONS = {
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

STRATEGY_DESCRIPTIONS = {
    'LM then SS': 'Implement Lean Manufacturing first, then Six Sigma',
    'SS then LM': 'Implement Six Sigma first, then Lean Manufacturing',
    'LM & SS': 'Implement Lean Manufacturing and Six Sigma simultaneously',
    'Failure': 'Current configuration historically associated with poor outcomes'
}

# ==========================================
# 2. INTELLIGENT BACKEND CLASS
# ==========================================
class L6SBrain:
    def __init__(self,
                 local_model: str = "llama3:8b",
                 api_model: str = "gemini-pro",
                 temperature: float = 0.3,
                 api_key: str = None,
                 use_api: bool = False):
        self.graph = nx.Graph()  # Use undirected graph for bidirectional traversal
        self.data_loaded = False

        # Initialize LLM Manager with both local and API options
        try:
            self.llm_manager = LLMManager(
                local_model=local_model,
                api_model=api_model,
                temperature=temperature,
                api_key=api_key
            )

            # Switch to API model if requested
            if use_api:
                self.llm_manager.switch_to_api()

            self.llm_available = self.llm_manager.is_available()
            self.llm = self.llm_manager if self.llm_available else None
        except Exception as e:
            print(f"Warning: LLM Manager not available: {e}")
            self.llm_manager = None
            self.llm = None
            self.llm_available = False

        # Initialize language detection components
        self.detected_language = None
        # Force all user-facing responses to English
        self.force_english = True
        self.translation_cache = {}

        # Machine Learning Models
        self.rf_regressor = None
        self.rf_classifier = None
        self.knn = None
        self.imputer = None
        self.X_imputed = None

        # Feature definitions
        self.feature_names = [f"IL{i}" for i in range(1, 8)] + \
                             [f"IS{i}" for i in range(1, 8)] + \
                             [f"M{i}" for i in range(1, 8)]

        # Add performance optimization structures
        self.sample_index = {}  # For faster sample lookups
        self.strategy_index = {}  # For faster strategy lookups
        self.cluster_index = {}  # For faster cluster lookups

        # Load Data & Train
        self.initialize_local_graph()

        # Load CSF level descriptions
        self.load_csf_level_descriptions()

    def load_csf_level_descriptions(self):
        """Load detailed level descriptions and prescriptions for each CSF factor"""
        self.csf_level_descriptions = {}
        self.csf_level_prescriptions = {}  # New: Store prescriptions separately

        if JSON_DATA_PATH.exists():
            try:
                with open(JSON_DATA_PATH, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

                # Direct mapping - more reliable
                factor_mapping = {
                    'L_CSFs1': 'IL1', 'L_CSFs2': 'IL2', 'L_CSFs3': 'IL3',
                    'L_CSFs4': 'IL4', 'L_CSFs5': 'IL5', 'L_CSFs6': 'IL6', 'L_CSFs7': 'IL7',
                    'S_CSFs1': 'IS1', 'S_CSFs2': 'IS2', 'S_CSFs3': 'IS3',
                    'S_CSFs4': 'IS4', 'S_CSFs5': 'IS5', 'S_CSFs6': 'IS6', 'S_CSFs7': 'IS7',
                    'M_CSFs1': 'M1', 'M_CSFs2': 'M2', 'M_CSFs3': 'M3',
                    'M_CSFs4': 'M4', 'M_CSFs5': 'M5', 'M_CSFs6': 'M6', 'M_CSFs7': 'M7'
                }

                cfs_data = json_data.get('critical_success_factors', {})
                all_factors = []
                all_factors.extend(cfs_data.get('lean', []))
                all_factors.extend(cfs_data.get('six_sigma', []))
                all_factors.extend(cfs_data.get('maturity_levels', []))

                for factor_data in all_factors:
                    factor_id = factor_data.get('id')
                    levels = factor_data.get('levels', [])

                    if factor_id in factor_mapping:
                        var_name = factor_mapping[factor_id]

                        # Load descriptions
                        self.csf_level_descriptions[var_name] = {
                            level['level']: level['description']
                            for level in levels
                        }

                        # Load prescriptions
                        self.csf_level_prescriptions[var_name] = {
                            level['level']: level.get('prescription', 'No prescription available')
                            for level in levels
                        }

                        print(f"Loaded {len(levels)} levels for {var_name}")

                print(f"Total CSF factors loaded: {len(self.csf_level_descriptions)}")
            except Exception as e:
                print(f"Error loading CSF level descriptions: {e}")
                import traceback
                traceback.print_exc()

    def generate_performance_breakdown_chart(self):
        """Generate a radar chart showing the performance breakdown across criteria."""
        # Get performance strategy data from json
        json_path = JSON_DATA_PATH
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                perf_data = json_data.get('performance_strategy', {})
                if not perf_data or 'breakdown' not in perf_data:
                    return None
                
                breakdown = perf_data.get('breakdown', {})
                
                # Prepare radar chart data
                categories = []
                real_values = []
                target_values = []
                
                for category_name, data in breakdown.items():
                    # Convert snake_case to readable format
                    readable_name = category_name.replace('_', ' ').replace('performance', '').strip().title()
                    categories.append(readable_name)
                    real_values.append(data.get('real_performance', 0) * 100)  # Convert to percentage
                    target_values.append(data.get('target_performance', 0) * 100)  # Convert to percentage
                
                # Create radar chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=real_values,
                    theta=categories,
                    fill='toself',
                    name='Current Performance',
                    line=dict(color='#e74c3c', width=2),
                    marker=dict(size=8)
                ))
                
                fig.add_trace(go.Scatterpolar(
                    r=target_values,
                    theta=categories,
                    fill='toself',
                    name='Target Performance',
                    line=dict(color='#2ecc71', width=2),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, max(max(target_values), max(real_values)) * 1.1]  # Scale appropriately
                        )),
                    title="Performance Gap Analysis: Current vs Target",
                    showlegend=True,
                    height=500
                )
                
                return fig
            except Exception as e:
                print(f"Error generating performance breakdown chart: {e}")
                return None
        return None

    def generate_subcriteria_barchart(self):
        """Generate a bar chart showing detailed sub-criteria contributions."""
        # Get performance strategy data from json
        json_path = JSON_DATA_PATH
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                perf_data = json_data.get('performance_strategy', {})
                if not perf_data or 'breakdown' not in perf_data:
                    return None
                
                breakdown = perf_data.get('breakdown', {})
                
                # Collect all sub-criteria data
                sub_categories = []
                contributions = []
                weights = []
                
                for category_name, data in breakdown.items():
                    sub_criteria = data.get('sub_criteria', {})
                    for sub_name, sub_data in sub_criteria.items():
                        sub_categories.append(sub_name.replace('_', ' ').title())
                        contribution = sub_data.get('contribution', 0) * 100  # Convert to percentage
                        weight = sub_data.get('weight', 0)
                        
                        contributions.append(contribution)
                        weights.append(weight)
                
                if not sub_categories:
                    return None
                
                # Create DataFrame for plotting
                df = pd.DataFrame({
                    'Sub-Criterion': sub_categories,
                    'Contribution (%)': contributions,
                    'Weight': weights
                })
                
                # Create bar chart
                fig = px.bar(df, 
                             x='Contribution (%)', 
                             y='Sub-Criterion',
                             orientation='h',
                             title="Sub-Criteria Contribution to Overall Performance",
                             hover_data=['Weight'])
                
                fig.update_layout(height=400 + len(sub_categories) * 20,  # Adjust height based on number of items
                                 showlegend=False)
                
                return fig
            except Exception as e:
                print(f"Error generating subcriteria barchart: {e}")
                return None
        return None

    def generate_performance_analytics_context(self, user_query):
        """Generate performance analytics context based on query."""
        # Get performance strategy data from json
        json_path = JSON_DATA_PATH
        if not json_path.exists():
            return ""
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            perf_data = json_data.get('performance_strategy', {})
            
            if not perf_data:
                return ""
            
            # Check if user is asking for performance analytics
            query_lower = user_query.lower()
            
            # Keywords that indicate a request for performance analytics
            analytics_keywords = [
                'performance table', 'analytics', 'breakdown', 'overview', 
                'gap analysis', 'performance analysis', 'detailed performance',
                'performance chart', 'performance graph', 'performance data',
                'performance metrics', 'performance report', 'performance dashboard',
                'ecart de performance', 'analyse de performance', 'bilan de performance',
                'performance analytics', 'show performance', 'performance details'
            ]
            
            wants_analytics = any(keyword in query_lower for keyword in analytics_keywords)
            
            if not wants_analytics:
                # Check for more general requests that might benefit from performance analytics
                general_performance_terms = [
                    'performance', 'stratégie', 'strategy', 'results', 'results overview',
                    'situation', 'état actuel', 'current situation', 'status', 'report',
                    'dashboard', 'overview', 'résumé', 'summary', 'analytics'
                ]
                
                # Check if multiple general terms are present or if it's a general performance query
                found_terms = [term for term in general_performance_terms if term in query_lower]
                wants_general_info = len(found_terms) >= 2 or any(term in query_lower for term in ['performance', 'stratégie', 'analytics', 'bilan', 'état'])

                if not wants_general_info:
                    return ""  # Don't include analytics if not relevant
            
            # Extract the key performance info
            recommended_strategy = perf_data.get('recommended_strategy', 'Not specified')
            current_rate = perf_data.get('current_performance_rate', 0) * 100
            target_rate = perf_data.get('target_performance_rate', 0) * 100
            
            # Generate criteria breakdown
            breakdown = perf_data.get('breakdown', {})
            criteria_breakdown = []
            
            for category, data in breakdown.items():
                readable_category = category.replace('_', ' ').replace('performance', '').strip().title()
                weight = data.get('weight', 0)
                real_perf = data.get('real_performance', 0) * 100
                target_perf = data.get('target_performance', 0) * 100
                gap = data.get('performance_gap', 0) * 100
                
                criteria_breakdown.append(
                    f"{readable_category} | Weight: {weight:.3f} | Current: {real_perf:.1f}% | Target: {target_perf:.1f}% | Gap: {gap:.1f}%"
                )
            
            # Generate sub-criteria breakdown
            sub_criteria_breakdown = []
            for category, data in breakdown.items():
                readable_category = category.replace('_', ' ').replace('performance', '').strip().title()
                sub_criteria = data.get('sub_criteria', {})
                
                if sub_criteria:
                    sub_criteria_breakdown.append(f"**{readable_category} Sub-criteria:**")
                    for sub_name, sub_data in sub_criteria.items():
                        sub_readable = sub_name.replace('_', ' ').title()
                        weight = sub_data.get('weight', 0)
                        contribution = sub_data.get('contribution', 0) * 100
                        sub_criteria_breakdown.append(f"  - {sub_readable}: Weight {weight:.3f}, Contribution {contribution:.1f}%")
            
            # Format the analytics text
            analytics_text = f"""
PERFORMANCE ANALYTICS OVERVIEW:
Recommended Strategy: {recommended_strategy}
Current Performance: {current_rate:.1f}%
Target Performance: {target_rate:.1f}%

PRIMARY PERFORMANCE CRITERIA:
"""
            analytics_text += "\n".join(criteria_breakdown)
            analytics_text += "\n\nSUB-CRITERIA BREAKDOWN:\n"
            analytics_text += "\n".join(sub_criteria_breakdown)
            
            return analytics_text.strip()
            
        except Exception as e:
            print(f"Error generating performance analytics context: {e}")
            return ""


    def get_csf_level_description(self, factor, level):
        """Get the description and prescription for a specific factor and level"""
        if factor in self.csf_level_descriptions:
            if level in self.csf_level_descriptions[factor]:
                description = self.csf_level_descriptions[factor][level]

                # Get the prescription if available
                prescription = "No prescription available"
                if factor in self.csf_level_prescriptions and level in self.csf_level_prescriptions[factor]:
                    prescription = self.csf_level_prescriptions[factor][level]

                # Translate to English if requested
                description = self.translate_to_english(description, ("desc", factor, level))
                prescription = self.translate_to_english(prescription, ("presc", factor, level))

                return f"{description}\nPrescription: {prescription}"

        # If not found, provide a fallback that explains where the data should come from
        factor_map = {
            'IL1': 'L_CSFs1', 'IL2': 'L_CSFs2', 'IL3': 'L_CSFs3', 'IL4': 'L_CSFs4',
            'IL5': 'L_CSFs5', 'IL6': 'L_CSFs6', 'IL7': 'L_CSFs7',
            'IS1': 'S_CSFs1', 'IS2': 'S_CSFs2', 'IS3': 'S_CSFs3', 'IS4': 'S_CSFs4',
            'IS5': 'S_CSFs5', 'IS6': 'S_CSFs6', 'IS7': 'S_CSFs7',
            'M1': 'M_CSFs1', 'M2': 'M_CSFs2', 'M3': 'M_CSFs3', 'M4': 'M_CSFs4',
            'M5': 'M_CSFs5', 'M6': 'M_CSFs6', 'M7': 'M_CSFs7'
        }

        mapped_factor = factor_map.get(factor, factor)
        return f"Description for {factor} (mapped to {mapped_factor}) level {level} not available in the loaded model. This should be found in the 'critical_success_factors' section of unified_lss_data_final.json under the appropriate category."

    def detect_csf_level_query(self, text):
        """Detect if user is asking about CSF level descriptions"""
        text_lower = text.lower().strip()

        # Simple indicators
        level_query_indicators = [
            'signification', 'signifie', 'mean', 'means', 'definition',
            'what is', 'qu\'est ce', 'c\'est quoi', 'explain', 'explique',
            'describe', 'décrit', 'tell me about', 'parle moi de'
        ]

        is_level_query = any(indicator in text_lower for indicator in level_query_indicators)

        if not is_level_query:
            return (False, None, None)

        # Extract factor (IL1, IS2, M3, etc.)
        factor_pattern = r'\b([IiMm][LlSs]?\d+)\b'
        factor_match = re.search(factor_pattern, text, re.IGNORECASE)

        if not factor_match:
            return (False, None, None)

        factor = factor_match.group(1).upper()
        if len(factor) > 2:
            factor = factor[0] + factor[1].upper() + factor[2:]

        # Extract level
        level_pattern = r'(?:level|niveau)\s*(\d+)|(?:^|\s)(\d+)(?:\s|$)'
        level_match = re.search(level_pattern, text_lower)

        level = None
        if level_match:
            level = int(level_match.group(1) or level_match.group(2))
        else:
            digit_match = re.search(r'\b([1-7])\b', text)
            if digit_match:
                level = int(digit_match.group(1))

        return (True, factor, level)

    def get_factor_overview(self, factor):
        """Get an overview of a factor with all its levels and prescriptions"""
        if factor in self.csf_level_descriptions:
            levels = self.csf_level_descriptions[factor]
            prescriptions = self.csf_level_prescriptions.get(factor, {})

            overview = f"Overview for {factor} ({CSF_DESCRIPTIONS.get(factor, 'Unknown factor')}):\n"
            for level in sorted(levels.keys()):
                description = levels[level]
                prescription = prescriptions.get(level, "No prescription available")
                description = self.translate_to_english(description, ("desc", factor, level))
                prescription = self.translate_to_english(prescription, ("presc", factor, level))
                overview += f"- Level {level}: {description}\n  Prescription: {prescription}\n"
            return overview
        return f"Overview for {factor} not available."

    def initialize_local_graph(self):
        """Initialize the local graph and models from cached data or JSON"""
        # Try to load from cache first
        if self.load_from_cache():
            if self._cache_is_compatible():
                print("Loaded graph and models from cache")
                self.data_loaded = True
                return
            print("Cached models incompatible; rebuilding from JSON.")
            self.graph = nx.Graph()
            self.df = None
            self.rf_regressor = None
            self.rf_classifier = None
            self.knn = None
            self.imputer = None
            self.X_imputed = None
            self.sample_index = {}
            self.strategy_index = {}
            self.cluster_index = {}

        # If no cache, load from JSON and build graph
        if JSON_DATA_PATH.exists():
            try:
                with open(JSON_DATA_PATH, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

                self.build_graph_from_json(json_data)

                # Train models from graph data
                self.train_models_from_graph()

                # Save to cache
                self.save_to_cache()

                self.data_loaded = True
                print(f"Built graph from JSON with {len(self.df)} samples")
            except Exception as e:
                print(f"Error loading JSON data: {e}")
        else:
            print(f"JSON data file not found: {JSON_DATA_PATH}")

    def build_graph_from_json(self, json_data):
        """Build graph from JSON data with proper relationships"""
        # Get samples from the JSON data
        if isinstance(json_data, dict):
            if "quantitative_data" in json_data and "samples" in json_data["quantitative_data"]:
                samples = json_data["quantitative_data"]["samples"]
            elif "samples" in json_data:
                samples = json_data["samples"]
            else:
                raise ValueError("JSON structure not recognized")
        elif isinstance(json_data, list):
            samples = json_data
        else:
            raise ValueError("Invalid JSON format")

        # Build graph structure
        strategy_nodes = {}
        cluster_nodes = {}

        for idx, sample in enumerate(samples):
            sample_id = f"sample_{idx}"

            # Extract performance
            perf_str = str(sample.get('estimations', '0')).replace('%', '').replace('<', '')
            try:
                performance = float(perf_str)
            except ValueError:
                performance = 0.0

            # Extract CSF scores
            node_attrs = {
                'node_type': 'Sample',
                'performance': performance,
                'cluster_id': sample.get('cluster_id', 0),
                'strategy': sample.get('strategie', 'Unknown')
            }

            # Add all CSF scores
            for feat in self.feature_names:
                val = sample.get(feat, None)
                if val in ["NaN", "nan", "", None]:
                    node_attrs[feat] = None
                else:
                    try:
                        float_val = float(val)
                        # Cap maturity factors at 5 to maintain consistency
                        if feat.startswith('M') and float_val > 5:
                            float_val = 5.0
                        node_attrs[feat] = float_val
                    except ValueError:
                        node_attrs[feat] = None

            # Add sample node
            self.graph.add_node(sample_id, **node_attrs)

            # Add strategy node and relationship if strategy exists
            strategy_name = sample.get('strategie', 'Unknown')
            if strategy_name not in ['NaN', None, '', 'Unknown']:
                strategy_id = f"strategy_{strategy_name.replace(' ', '_').replace('-', '_')}"
                if strategy_id not in strategy_nodes:
                    self.graph.add_node(strategy_id, node_type='Strategy', name=strategy_name)
                    strategy_nodes[strategy_id] = strategy_name

                # Add relationship between sample and strategy
                self.graph.add_edge(sample_id, strategy_id, relationship='USES_STRATEGY')

            # Add cluster node and relationship
            cluster_id = sample.get('cluster_id')
            if cluster_id is not None:
                cluster_key = f"cluster_{cluster_id}"
                if cluster_key not in cluster_nodes:
                    self.graph.add_node(cluster_key, node_type='Cluster', cluster_id=cluster_id)
                    cluster_nodes[cluster_key] = cluster_id

                self.graph.add_edge(sample_id, cluster_key, relationship='IN_CLUSTER')

        # Build performance indexes
        self._build_indexes()

        print(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def train_models_from_graph(self):
        """Train ML models from graph data"""
        # Extract data from graph
        samples = []
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('node_type') == 'Sample' and node_data.get('performance', 0) > 0:
                # Get strategy from connected strategy node
                strategy = node_data.get('strategy', None)
                if not strategy or strategy in ['NaN', 'Unknown', '']:
                    # Find strategy from connected nodes in the graph
                    for neighbor in self.graph.neighbors(node_id):
                        neighbor_data = self.graph.nodes[neighbor]
                        if neighbor_data.get('node_type') == 'Strategy':
                            strategy = neighbor_data.get('name')
                            break

                if strategy and strategy not in ['NaN', 'Unknown', 'Failure']:
                    sample = {'sample_id': node_id}
                    for feat in self.feature_names:
                        sample[feat] = node_data.get(feat)

                    sample['performance'] = node_data['performance']
                    sample['strategy'] = strategy
                    samples.append(sample)

        self.df = pd.DataFrame(samples)

        if len(self.df) == 0:
            raise ValueError("No valid samples found in graph")

        # Cap maturity factor values at 5 to maintain consistency
        for feat in self.feature_names:
            if feat.startswith('M') and feat in self.df.columns:
                self.df[feat] = self.df[feat].apply(lambda x: min(x, 5.0) if pd.notna(x) else x)

        # Prepare feature matrix
        X = self.df[self.feature_names].copy()

        # Identify rows with at least one valid CSF value
        valid_rows = X.dropna(how='all').index
        self.df = self.df.loc[valid_rows]
        X = X.loc[valid_rows]

        # Remove any remaining all-NaN rows
        X = X.fillna(X.mean())  # Impute remaining NaNs with column mean temporarily for training

        self.imputer = SimpleImputer(strategy='mean')
        X_imputed = self.imputer.fit_transform(X)
        self.X_imputed = X_imputed

        y_perf = self.df['performance'].values
        y_strat = self.df['strategy'].values

        # Train models
        self.rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_regressor.fit(X_imputed, y_perf)

        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_classifier.fit(X_imputed, y_strat)

        self.knn = NearestNeighbors(n_neighbors=min(5, len(self.df)), metric='euclidean')
        self.knn.fit(X_imputed)

        print(f"Models trained on {len(self.df)} samples")

    def save_to_cache(self):
        """Save trained graph and models locally"""
        cache_data = {
            'graph': self.graph,
            'df': self.df,
            'rf_regressor': self.rf_regressor,
            'rf_classifier': self.rf_classifier,
            'knn': self.knn,
            'imputer': self.imputer,
            'X_imputed': self.X_imputed,
            'feature_names': self.feature_names,
            'sample_index': self.sample_index,
            'strategy_index': self.strategy_index,
            'cluster_index': self.cluster_index,
            'csf_level_descriptions': getattr(self, 'csf_level_descriptions', {}),
            'csf_level_prescriptions': getattr(self, 'csf_level_prescriptions', {})  # Add prescriptions to cache
        }

        with open(GRAPH_CACHE_PATH, 'wb') as f:
            pickle.dump(cache_data, f)

        print(f"Graph and models cached at {GRAPH_CACHE_PATH}")

    def load_from_cache(self):
        """Load pre-trained graph from local cache"""
        if not GRAPH_CACHE_PATH.exists():
            return False

        try:
            with open(GRAPH_CACHE_PATH, 'rb') as f:
                cache_data = pickle.load(f)

            self.graph = cache_data['graph']
            self.df = cache_data['df']

            # Apply validation to cap maturity factors at 5 when loading from cache
            for feat in self.feature_names:
                if feat.startswith('M') and feat in self.df.columns:
                    self.df[feat] = self.df[feat].apply(lambda x: min(x, 5.0) if pd.notna(x) else x)

            self.rf_regressor = cache_data['rf_regressor']
            self.rf_classifier = cache_data['rf_classifier']
            self.knn = cache_data['knn']
            self.imputer = cache_data['imputer']
            self.X_imputed = cache_data['X_imputed']
            self.feature_names = cache_data['feature_names']

            # Load indexes if available
            if 'sample_index' in cache_data:
                self.sample_index = cache_data['sample_index']
            if 'strategy_index' in cache_data:
                self.strategy_index = cache_data['strategy_index']
            if 'cluster_index' in cache_data:
                self.cluster_index = cache_data['cluster_index']

            # Load CSF level descriptions if available in cache
            if 'csf_level_descriptions' in cache_data:
                self.csf_level_descriptions = cache_data['csf_level_descriptions']
            else:
                # If not in cache, load them from JSON
                self.load_csf_level_descriptions()

            # Load CSF level prescriptions if available in cache
            if 'csf_level_prescriptions' in cache_data:
                self.csf_level_prescriptions = cache_data['csf_level_prescriptions']
            else:
                # If not in cache, load them from JSON (they will be loaded with descriptions)
                self.load_csf_level_descriptions()

            print(f"Loaded from cache: {len(self.df)} samples, {self.graph.number_of_nodes()} nodes")
            return True
        except Exception as e:
            print(f"Cache load failed: {e}")
            return False

    def _cache_is_compatible(self):
        """Validate cached models against current sklearn version."""
        if self.imputer is None or self.df is None:
            return False
        try:
            sample = self.df[self.feature_names].head(1)
            self.imputer.transform(sample)
        except Exception as e:
            print(f"Cached models incompatible: {e}")
            return False
        return True

    def _build_indexes(self):
        """Build performance indexes for faster lookups"""
        self.sample_index = {}
        self.strategy_index = {}
        self.cluster_index = {}

        for node_id, node_data in self.graph.nodes(data=True):
            node_type = node_data.get('node_type')

            if node_type == 'Sample':
                self.sample_index[node_id] = node_data
            elif node_type == 'Strategy':
                strategy_name = node_data.get('name')
                if strategy_name:
                    if strategy_name not in self.strategy_index:
                        self.strategy_index[strategy_name] = []
                    self.strategy_index[strategy_name].append(node_id)
            elif node_type == 'Cluster':
                cluster_id = node_data.get('cluster_id')
                if cluster_id is not None:
                    if cluster_id not in self.cluster_index:
                        self.cluster_index[cluster_id] = []
                    self.cluster_index[cluster_id].append(node_id)

    def find_similar_by_graph_traversal(self, csf_scores, k=5):
        """
        True graph RAG: Use graph structure to find similar samples via traversal
        This finds related samples by navigating through the graph relationships
        """
        # Convert input CSF scores to a vector for KNN search
        input_vector = []
        for feat in self.feature_names:
            if feat in csf_scores and csf_scores[feat] is not None:
                input_vector.append(csf_scores[feat])
            else:
                # Use mean imputation for missing values
                input_vector.append(self.df[feat].mean())

        # Find k nearest neighbors using the trained KNN
        distances, indices = self.knn.kneighbors([input_vector], n_neighbors=min(k, len(self.df)))

        similar_samples = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.df):
                sample_row = self.df.iloc[idx].copy()

                # Get strategy from graph if not in dataframe
                sample_id = sample_row.get('sample_id', f"sample_{idx}")
                node_data = self.graph.nodes.get(sample_id, {})

                # Find strategy in connected nodes if not in sample_row
                strategy = sample_row.get('strategy', 'Unknown')
                if strategy in ['Unknown', 'NaN', None, '']:
                    for neighbor in self.graph.neighbors(sample_id):
                        neighbor_data = self.graph.nodes[neighbor]
                        if neighbor_data.get('node_type') == 'Strategy':
                            strategy = neighbor_data.get('name', 'Unknown')
                            break

                similar_sample = {
                    'sample_id': sample_id,
                    'distance': distances[0][i],
                    'performance': sample_row['performance'],
                    'strategy': strategy,
                    'similarity_score': 100 / (1 + distances[0][i]),
                }

                # Add CSF values
                for feat in self.feature_names:
                    if feat in sample_row:
                        similar_sample[feat] = sample_row[feat]

                similar_samples.append(similar_sample)

        return similar_samples

    def find_related_via_graph_paths(self, csf_scores, k=5):
        """
        Advanced graph traversal: Find paths through the graph to discover relationships
        """
        # Find the most similar samples first
        similar_samples = self.find_similar_by_graph_traversal(csf_scores, k=k*2)

        # Enhance with graph relationships
        enhanced_samples = []
        for sample in similar_samples[:k]:
            sample_id = sample.get('sample_id', None)

            if sample_id and self.graph.has_node(sample_id):
                # Look for relationships through the graph
                related_strategies = []
                related_clusters = []

                # Traverse to find connected strategies
                for neighbor in self.graph.neighbors(sample_id):
                    neighbor_data = self.graph.nodes[neighbor]
                    rel_type = self.graph.edges[sample_id, neighbor].get('relationship', 'UNKNOWN')

                    if neighbor_data.get('node_type') == 'Strategy':
                        related_strategies.append(neighbor_data.get('name'))
                    elif neighbor_data.get('node_type') == 'Cluster':
                        related_clusters.append(neighbor_data.get('cluster_id'))

                # Add relationship info to sample
                sample['related_strategies'] = related_strategies
                sample['related_clusters'] = related_clusters

            enhanced_samples.append(sample)

        return enhanced_samples

    def detect_language(self, text):
        """Simple language detection based on common words in French vs English queries"""
        # Honor forced English responses
        if getattr(self, "force_english", False):
            return 'en'

        text_lower = text.lower()

        # French language indicators
        french_indicators = [
            'le', 'la', 'les', 'des', 'un', 'une', 'du', 'au', 'aux',
            'et', 'ou', 'dans', 'sur', 'pour', 'avec', 'sans', 'par',
            'quelle', 'quel', 'quels', 'quelles', 'est', 'sont', 'dans',
            'comment', 'pourquoi', 'quelle', 'quelle est', 'quelles sont',
            'svp', 's\'il vous plaît', 'merci', 'bonjour', 'bonsoir',
            'soir', 'matin', 'soirée', 'jour', 'nous', 'vous', 'ils',
            'elles', 'son', 'sa', 'ses', 'mon', 'ma', 'mes', 'ton',
            'ta', 'tes', 'notre', 'nos', 'votre', 'vos', 'leur', 'leurs'
        ]

        # English language indicators
        english_indicators = [
            'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for',
            'with', 'without', 'by', 'what', 'which', 'how', 'why', 'is',
            'are', 'please', 'thank', 'thanks', 'hello', 'hi', 'goodbye',
            'good', 'morning', 'evening', 'night', 'day', 'we', 'you', 'they',
            'his', 'her', 'its', 'my', 'mine', 'your', 'yours', 'our',
            'ours', 'their', 'theirs'
        ]

        french_count = sum(1 for word in text_lower.split() if word in french_indicators)
        english_count = sum(1 for word in text_lower.split() if word in english_indicators)

        # If we can't determine from common words, use the first few words as a hint
        if french_count == 0 and english_count == 0:
            # Check for French-specific characters or phrases
            if any(char in text_lower for char in ['ç', 'à', 'é', 'è', 'ù', 'ê', 'â', 'î', 'ô', 'û']):
                return 'fr'
            if any(phrase in text_lower for phrase in ['svp', 's\'il', 'nous', 'vous', 'dans', 'pour', 'cette', 'ceci', 'cela']):
                return 'fr'
            if any(phrase in text_lower for phrase in ['please', 'thank', 'hello', 'the', 'how', 'what', 'this', 'that', 'these', 'those']):
                return 'en'

        # For mixed content, consider the first 2-3 words to determine the main language
        first_words = text_lower.split()[:3]
        first_french = sum(1 for word in first_words if word in french_indicators)
        first_english = sum(1 for word in first_words if word in english_indicators)

        if french_count > english_count or first_french > first_english:
            return 'fr'
        elif english_count > french_count or first_english > first_french:
            return 'en'
        else:
            # If counts are equal, return the previously detected language, or default to English
            return self.detected_language or 'en'

    def translate_to_english(self, text, cache_key=None):
        """Translate text to English using the LLM when available, with simple caching."""
        if not text or not getattr(self, "force_english", False):
            return text

        if cache_key is not None and cache_key in self.translation_cache:
            return self.translation_cache[cache_key]

        if not self.llm_available or self.llm is None:
            fallback = "English translation unavailable. Configure an LLM to translate this content."
            if cache_key is not None:
                self.translation_cache[cache_key] = fallback
            return fallback

        def build_prompt(strict=False):
            base = (
                "You are a professional translator. Translate the following text to English. "
                "Keep the meaning, formatting, and any labels exactly as they appear. "
                "Return only the translation. If the text is already English, return it unchanged."
            )
            if strict:
                base += " Do NOT include any French words or phrases."
            return f"{base}\n\n{text}"

        def invoke_translation(prompt):
            try:
                # Prefer API model if available for translation quality
                if hasattr(self.llm, "get_available_models"):
                    availability = self.llm.get_available_models()
                    if availability.get("api", {}).get("available", False):
                        return self.llm.invoke_api_only(prompt).strip()
                return self.llm.invoke(prompt).strip()
            except Exception:
                return ""

        translated = invoke_translation(build_prompt(strict=False))
        if translated and not self.is_likely_french(translated):
            if cache_key is not None:
                self.translation_cache[cache_key] = translated
            return translated

        translated = invoke_translation(build_prompt(strict=True))
        if translated and not self.is_likely_french(translated):
            if cache_key is not None:
                self.translation_cache[cache_key] = translated
            return translated

        # Last resort: return English-only notice
        fallback = "English translation failed. Please try again or check LLM configuration."
        if cache_key is not None:
            self.translation_cache[cache_key] = fallback
        return fallback

    def is_likely_french(self, text):
        """Heuristic check to detect French in text without forcing language."""
        if not text:
            return False

        text_lower = text.lower()
        french_indicators = {
            'le', 'la', 'les', 'des', 'un', 'une', 'du', 'au', 'aux',
            'et', 'ou', 'dans', 'sur', 'pour', 'avec', 'sans', 'par',
            'quelle', 'quel', 'quels', 'quelles', 'est', 'sont',
            'comment', 'pourquoi', 'niveau', 'signifie', 'signification',
            'svp', 's\'il', 'merci', 'bonjour', 'bonsoir'
        }
        english_indicators = {
            'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for',
            'with', 'without', 'by', 'what', 'which', 'how', 'why', 'is',
            'are', 'please', 'thank', 'thanks', 'hello', 'hi', 'good'
        }

        words = re.findall(r"[a-zA-Z']+", text_lower)
        french_count = sum(1 for word in words if word in french_indicators)
        english_count = sum(1 for word in words if word in english_indicators)

        if any(ch in text_lower for ch in "àâäçéèêëîïôöùûüÿœ"):
            return True

        if french_count >= 2:
            return True
        if french_count == 1 and english_count == 0:
            return True
        return False

    def ensure_english(self, text, cache_key=None):
        """Guarantee English output, falling back to an English notice if needed."""
        if not text or not getattr(self, "force_english", False):
            return text
        # Always translate to guarantee English output
        return self.translate_to_english(text, cache_key)

    def fuzzy_match_keyword(self, word, keywords, threshold=0.75):
        """Fuzzy match a word against a list of keywords for typo tolerance"""
        best_match = None
        best_ratio = 0

        for keyword in keywords:
            ratio = SequenceMatcher(None, word.lower(), keyword.lower()).ratio()
            if ratio > best_ratio and ratio >= threshold:
                best_ratio = ratio
                best_match = keyword

        return best_match

    def is_l6s_related(self, text):
        """Check if query is related to L6S domain with typo tolerance"""
        l6s_keywords = [
            'lean', 'six sigma', 'sigma', 'l6s', 'strategy', 'performance', 'csf',
            'il1', 'il2', 'il3', 'il4', 'il5', 'il6', 'il7',
            'is1', 'is2', 'is3', 'is4', 'is5', 'is6', 'is7',
            'leadership', 'culture', 'training', 'communication', 'tools',
            'employee', 'expertise', 'implementation', 'kaizen', 'dmaic',
            'achieve', 'reach', 'target', 'goal', 'improve', 'optimization',
            'recommend', 'suggest', 'advice', 'help'
        ]
        text_lower = text.lower().strip()
        
        # Check for greetings/thanks - always allow these
        greetings = ['hello', 'hi', 'hey', 'thanks', 'thank you', 'bye', 'goodbye']
        if any(greeting in text_lower for greeting in greetings):
            return True
        
        # Check for numbers that might be follow-up goals/scores
        if re.match(r'^\d+%?$', text_lower) or re.match(r'^(how about|what about)\s*\d+', text_lower):
            return True
        
        # Direct keyword match
        if any(keyword in text_lower for keyword in l6s_keywords):
            return True
        
        # Fuzzy match for typos
        words = text_lower.split()
        for word in words:
            if len(word) > 3:  # Only fuzzy match words longer than 3 chars
                if self.fuzzy_match_keyword(word, l6s_keywords, threshold=0.75):
                    return True
        
        return False

    def extract_scores_from_text(self, text):
        """Robust NLP to extract 'IL1 is 4', 'IS2=5', 'M1=3', etc. with typo tolerance"""
        scores = {}

        # First try standard patterns for all factor types
        matches = re.findall(r'([IM][LS]\s*\d+|M\s*\d+)\s*(?:is|=|:|equals?)\s*(\d+)', text, re.IGNORECASE)

        for key_raw, val in matches:
            key = key_raw.upper().replace(" ", "")
            if key in self.feature_names:
                val_int = int(val)
                # All factors (IL/IS/M) are 1-5 scale
                if 1 <= val_int <= 5:
                    scores[key] = val_int

        # Try to handle typos like "IL1" misspelled as "ILI" or "1L1", "M1" as "MI1" etc.
        typo_patterns = re.findall(r'([I1][LS]\s*\d+|I\s*[LS]\s*\d+|[M1]\s*\d+|M\s*\d+)\s*(?:is|=|:|equals?)\s*(\d+)', text, re.IGNORECASE)
        for key_raw, val in typo_patterns:
            # Normalize: replace 1 with I/M, remove spaces
            key = key_raw.upper().replace(" ", "")
            key = key.replace("1L", "IL").replace("1S", "IS").replace("1M", "M")  # Handle "1" typos
            if key in self.feature_names and key not in scores:
                val_int = int(val)
                # All factors (IL/IS/M) are 1-5 scale
                if 1 <= val_int <= 5:
                    scores[key] = val_int

        return scores

    def analyze_goal(self, text):
        """Detects if user is asking for a target performance"""
        # Enhanced patterns to catch more natural language
        patterns = [
            r'(?:achieve|target|get|reach|attain|hit)\s*(\d+)(?:%)?',
            r'(\d+)(?:%)?(?:\s+performance)?',
            r'(?:goal|aim)(?:\s+is|\s+of)?\s*(\d+)(?:%)?',
            r'(?:how\s+about|what\s+about)\s*(\d+)(?:%)?'
        ]
        
        text_lower = text.lower()
        # Typo-tolerant goal contexts
        goal_contexts = ['achieve', 'reach', 'target', 'goal', 'get', 'attain', 'hit', 'how to', 'aim', 'how about', 'what about']
        
        # Check for fuzzy matches
        words = text_lower.split()
        has_goal_context = any(ctx in text_lower for ctx in goal_contexts)
        
        if not has_goal_context:
            # Try fuzzy matching
            for word in words:
                if self.fuzzy_match_keyword(word, goal_contexts, threshold=0.8):
                    has_goal_context = True
                    break
        
        if has_goal_context:
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    perf = float(match.group(1))
                    if 30 <= perf <= 100:
                        return perf
        return None
    
    def detect_follow_up_goal(self, text, last_goal):
        """Detect if user is asking about a different goal in follow-up"""
        text_lower = text.lower().strip()
        
        # Match patterns like: "how about 70", "what about 75", "and 65"
        follow_up_patterns = [
            r'^(?:how\s+about|what\s+about|and|or)\s*(\d+)(?:%)?',
            r'^(\d+)(?:%)?$'
        ]
        
        for pattern in follow_up_patterns:
            match = re.match(pattern, text_lower)
            if match:
                perf = float(match.group(1))
                if 30 <= perf <= 100 and perf != last_goal:
                    return perf
        return None

    def get_reverse_recommendation(self, target_perf):
        """Reverse logic: What parameters needed to achieve target performance?"""
        success_df = self.df[self.df['performance'] >= target_perf]

        if len(success_df) < 3:
            success_df = self.df.nlargest(int(len(self.df)*0.1), 'performance')
            msg = f"Note: Only {len(success_df)} cases found achieving ≥{target_perf}%. Analyzing top {len(success_df)} performers instead."
            sample_warning = True
        else:
            msg = f"Analysis based on {len(success_df)} successful cases achieving ≥{target_perf}%."
            sample_warning = False

        means = {k: round(v) for k, v in success_df[self.feature_names].mean().to_dict().items()}

        # Cap maturity factor recommendations at 5 to maintain consistency
        for key, value in means.items():
            if key.startswith('M') and value > 5:
                means[key] = 5

        top_strat = success_df['strategy'].mode()[0] if len(success_df) > 0 else "Unknown"

        return {
            "means": means,
            "strategy": top_strat,
            "message": msg,
            "avg_perf": success_df['performance'].mean(),
            "sample_size": len(success_df),
            "sample_warning": sample_warning
        }

    def predict_with_partial_params(self, provided_scores):
        """Make intelligent predictions with partial CSF parameters using graph traversal"""
        # Use graph traversal to find similar cases
        graph_similar_cases = self.find_related_via_graph_paths(provided_scores, k=10)

        # Build complete vector
        complete_vector = []
        imputation_details = {}

        for param in self.feature_names:
            if param in provided_scores:
                complete_vector.append(provided_scores[param])
                imputation_details[param] = f"{provided_scores[param]} (provided)"
            else:
                # Use graph traversal to estimate missing values
                if graph_similar_cases:
                    # Extract values for this parameter from similar cases
                    vals = [case.get(param) for case in graph_similar_cases if case.get(param) is not None and pd.notna(case.get(param))]
                    if vals:
                        avg_val = round(np.mean(vals))
                        # Cap maturity factors at 5
                        if param.startswith('M') and avg_val > 5:
                            avg_val = 5
                        complete_vector.append(avg_val)
                        imputation_details[param] = f"{avg_val} (from {len(vals)} graph neighbors)"
                    else:
                        avg_val = round(self.df[param].mean())
                        # Cap maturity factors at 5
                        if param.startswith('M') and avg_val > 5:
                            avg_val = 5
                        complete_vector.append(avg_val)
                        imputation_details[param] = f"{avg_val} (database average)"
                else:
                    avg_val = round(self.df[param].mean())
                    # Cap maturity factors at 5
                    if param.startswith('M') and avg_val > 5:
                        avg_val = 5
                    complete_vector.append(avg_val)
                    imputation_details[param] = f"{avg_val} (database average)"

        # Convert graph similar cases to dataframe format for compatibility
        if graph_similar_cases:
            similar_cases = pd.DataFrame(graph_similar_cases)
        else:
            similar_cases = pd.DataFrame()

        return complete_vector, imputation_details, similar_cases

    def predict_strategy_and_perf(self, input_vector, provided_params, imputation_details=None):
        """Forward logic: Predict strategy and performance from parameters"""
        vector_clean = self.imputer.transform([input_vector])

        pred_perf = self.rf_regressor.predict(vector_clean)[0]
        pred_strat = self.rf_classifier.predict(vector_clean)[0]
        pred_proba = self.rf_classifier.predict_proba(vector_clean)[0]
        confidence = max(pred_proba) * 100

        dists, indices = self.knn.kneighbors(vector_clean)
        similar_cases = self.df.iloc[indices[0]].copy()
        similar_cases['distance'] = dists[0]
        similar_cases['similarity_score'] = 100 / (1 + dists[0])

        feature_importance = dict(zip(self.feature_names, self.rf_regressor.feature_importances_))

        # Enhance with graph-based relationships
        enhanced_similar_cases = []
        for idx, case in similar_cases.iterrows():
            sample_id = case.get('sample_id', None)
            if sample_id and self.graph.has_node(sample_id):
                # Get additional graph relationships
                related_strategies = []
                related_clusters = []

                for neighbor in self.graph.neighbors(sample_id):
                    neighbor_data = self.graph.nodes[neighbor]
                    if neighbor_data.get('node_type') == 'Strategy':
                        related_strategies.append(neighbor_data.get('name'))
                    elif neighbor_data.get('node_type') == 'Cluster':
                        related_clusters.append(neighbor_data.get('cluster_id'))

                case = case.copy()  # Make a copy to avoid SettingWithCopyWarning
                case['related_strategies'] = related_strategies
                case['related_clusters'] = related_clusters

            enhanced_similar_cases.append(case)

        if enhanced_similar_cases:
            similar_cases = pd.DataFrame(enhanced_similar_cases)

        # Handle "Failure" case - warn the user instead of recommending failure
        if pred_strat == "Failure":
            # Return a special result indicating failure risk
            return {
                "strategy": pred_strat,
                "performance": pred_perf,
                "confidence": confidence,
                "similar_cases": similar_cases,
                "feature_importance": feature_importance,
                "provided_params": provided_params,
                "imputation_details": imputation_details,
                "failure_warning": True
            }

        return {
            "strategy": pred_strat,
            "performance": pred_perf,
            "confidence": confidence,
            "similar_cases": similar_cases,
            "feature_importance": feature_importance,
            "provided_params": provided_params,
            "imputation_details": imputation_details,
            "failure_warning": False
        }

    def find_paths_between_csfs_and_outcomes(self, start_csfs, target_performance=None, k=3):
        """
        Advanced graph traversal: Find paths between CSF configurations and performance outcomes
        This allows for more sophisticated reasoning about causality and relationships
        """
        # Find samples with similar CSF configurations
        similar_samples = self.find_related_via_graph_paths(start_csfs, k*2)

        # Analyze paths in the graph to understand relationships
        path_analysis = []

        for sample in similar_samples[:k]:
            sample_id = sample.get('sample_id')
            if sample_id and self.graph.has_node(sample_id):
                # Look for paths from this sample to related outcomes
                sample_node = sample_id

                # Analyze the immediate neighborhood
                neighbors = list(self.graph.neighbors(sample_node))

                # Group neighbors by type
                strategy_neighbors = []
                cluster_neighbors = []

                for neighbor in neighbors:
                    neighbor_data = self.graph.nodes[neighbor]
                    if neighbor_data.get('node_type') == 'Strategy':
                        strategy_neighbors.append({
                            'strategy': neighbor_data.get('name'),
                            'relationship': self.graph.edges[sample_node, neighbor].get('relationship')
                        })
                    elif neighbor_data.get('node_type') == 'Cluster':
                        cluster_neighbors.append({
                            'cluster_id': neighbor_data.get('cluster_id'),
                            'relationship': self.graph.edges[sample_node, neighbor].get('relationship')
                        })

                path_info = {
                    'sample_id': sample_id,
                    'performance': sample.get('performance'),
                    'strategies': strategy_neighbors,
                    'clusters': cluster_neighbors,
                    'similarity_score': sample.get('similarity_score')
                }

                path_analysis.append(path_info)

        return path_analysis

    def analyze_causal_paths(self, csf_input, depth=2):
        """
        Analyze potential causal paths in the graph from CSF inputs to outcomes
        This simulates more sophisticated graph reasoning
        """
        # This would implement more complex path analysis
        # For now, we'll use a simplified version based on our graph structure
        results = []

        # Find samples matching the input CSFs
        matching_samples = self.find_related_via_graph_paths(csf_input, k=10)

        for sample in matching_samples:
            sample_id = sample.get('sample_id')
            if sample_id and self.graph.has_node(sample_id):
                # Extract the full context through graph relationships
                context = {
                    'sample_id': sample_id,
                    'csf_values': {k: v for k, v in sample.items() if k in self.feature_names},
                    'performance': sample.get('performance'),
                    'strategy': sample.get('strategy'),
                    'related_strategies': sample.get('related_strategies', []),
                    'related_clusters': sample.get('related_clusters', [])
                }
                results.append(context)

        return results

    def generate_graph_rag_context(self, user_query, csf_scores=None):
        """
        Generate context for the LLM from graph traversal results
        This is true Graph RAG - using graph relationships to inform responses
        """
        context_parts = []

        # Add basic info
        context_parts.append(f"L6S domain with {len(self.df)} implementation cases in local graph.")
        context_parts.append(f"Graph contains {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} relationships.")

        # If CSF scores provided, add related graph insights
        if csf_scores:
            # Find similar cases through graph traversal
            similar_cases = self.find_related_via_graph_paths(csf_scores, k=5)

            if similar_cases:
                context_parts.append(f"Found {len(similar_cases)} similar cases in graph:")
                for i, case in enumerate(similar_cases[:3]):  # Top 3 cases
                    perf = case.get('performance', 'Unknown')
                    strat = case.get('strategy', 'Unknown')
                    context_parts.append(f"- Case {i+1}: Performance {perf}%, Strategy '{strat}'")

                    # Add relationship context
                    if 'related_strategies' in case and case['related_strategies']:
                        context_parts.append(f"  Related strategies: {', '.join(case['related_strategies'][:2])}")
                    if 'related_clusters' in case and case['related_clusters']:
                        context_parts.append(f"  Related clusters: {', '.join(map(str, case['related_clusters'][:2]))}")

        # Add path analysis if relevant
        if csf_scores:
            path_analysis = self.find_paths_between_csfs_and_outcomes(csf_scores, k=2)
            if path_analysis:
                context_parts.append("Relationship patterns found:")
                for i, path in enumerate(path_analysis[:2]):
                    perf = path.get('performance', 'Unknown')
                    context_parts.append(f"- Path {i+1}: Performance {perf}%")
                    if path.get('strategies'):
                        strategies = [s['strategy'] for s in path['strategies'][:2]]
                        context_parts.append(f"  Connected strategies: {', '.join(strategies)}")

        # ADD PERFORMANCE ANALYTICS CONTEXT IF REQUESTED
        perf_analytics = self.generate_performance_analytics_context(user_query)
        if perf_analytics:
            context_parts.append(perf_analytics)

        return "\n".join(context_parts)

    def get_graph_insights_for_query(self, query, csf_scores=None):
        """
        Extract specific graph-based insights relevant to the user's query
        This allows for query-specific graph traversal and analysis
        """
        insights = []

        # Analyze query intent to determine what graph insights are most relevant
        query_lower = query.lower()

        if any(word in query_lower for word in ['strategy', 'recommend', 'suggest', 'approach']):
            # Find strategy-related insights from the graph
            if csf_scores:
                similar_cases = self.find_related_via_graph_paths(csf_scores, k=5)
                strategy_counts = {}

                for case in similar_cases:
                    strategy = case.get('strategy')
                    if strategy and strategy != 'Unknown':
                        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

                if strategy_counts:
                    top_strategies = sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                    insights.append("Most successful strategies for similar organizations:")
                    for strategy, count in top_strategies:
                        insights.append(f"- {strategy} (used by {count} similar organizations)")

        if any(word in query_lower for word in ['performance', 'achieve', 'reach', 'target', 'goal']):
            # Find performance-related insights from the graph
            if csf_scores:
                similar_cases = self.find_related_via_graph_paths(csf_scores, k=5)
                performances = [case.get('performance', 0) for case in similar_cases if case.get('performance', 0) > 0]

                if performances:
                    avg_perf = np.mean(performances)
                    max_perf = max(performances)
                    min_perf = min(performances)

                    insights.append(f"Similar organizations achieved:")
                    insights.append(f"- Average performance: {avg_perf:.1f}%")
                    insights.append(f"- Range: {min_perf:.1f}% - {max_perf:.1f}%")

        if any(word in query_lower for word in ['improve', 'increase', 'enhance', 'better', 'higher']):
            # Find improvement pathways in the graph
            if csf_scores:
                # Analyze which CSFs tend to be higher in better-performing similar cases
                similar_cases = self.find_related_via_graph_paths(csf_scores, k=10)
                top_performers = sorted(similar_cases, key=lambda x: x.get('performance', 0), reverse=True)[:5]

                if top_performers:
                    avg_top_csfs = {}
                    for feat in self.feature_names:
                        values = [case.get(feat, 0) for case in top_performers if case.get(feat) is not None]
                        if values:
                            avg_top_csfs[feat] = np.mean(values)

                    # Compare with user's scores to suggest improvements
                    if csf_scores:
                        improvements = []
                        for feat, avg_val in avg_top_csfs.items():
                            user_val = csf_scores.get(feat, 0)
                            if avg_val > user_val + 0.5:  # Only suggest meaningful improvements
                                improvements.append((feat, avg_val - user_val))

                        if improvements:
                            improvements.sort(key=lambda x: x[1], reverse=True)
                            insights.append("Areas for improvement based on top performers:")
                            for i, (feat, diff) in enumerate(improvements[:3]):
                                insights.append(f"- {feat}: Increase by {diff:.1f} points ({CSF_DESCRIPTIONS[feat]})")

        return "\n".join(insights)

    def switch_to_local_model(self):
        """Switch to local model if available"""
        if self.llm_manager:
            return self.llm_manager.switch_to_local()
        return False

    def switch_to_api_model(self):
        """Switch to API model if available"""
        if self.llm_manager:
            return self.llm_manager.switch_to_api()
        return False

    def get_available_models(self):
        """Get information about available models"""
        if self.llm_manager:
            return self.llm_manager.get_available_models()
        return {
            "local": {"available": False, "model": None},
            "api": {"available": False, "model": None},
            "active": "none"
        }

    def get_active_model_info(self):
        """Get information about the currently active model"""
        if self.llm_manager:
            return self.llm_manager.get_active_model_info()
        return "No model available"

# ==========================================
# 3. STREAMLIT UI LOGIC
# ==========================================

def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Welcome to the Lean Six Sigma Strategy Advisor. I analyze L6S implementation data from 156 organizations to provide data-driven recommendations.\n\n**How it works**: You provide your Critical Success Factor (CSF) scores (1-5 ratings for IL/IS factors, 1-7 for M factors; 21 total factors), and I predict your expected performance percentage and recommend the best implementation strategy.\n\n**CSF Factors**:\n- IL1-IL7: Lean factors (1-5 ratings, Leadership, Culture, Communication, Training, Tools, Employee Involvement, Expertise)\n- IS1-IS7: Six Sigma factors (1-5 ratings, same categories)\n- M1-M7: Maturity factors (1-7 ratings, same categories)\n\nYou can:\n- Get strategy recommendations (e.g., 'What strategy if IL1=4, IS1=3, M1=5?')\n- Request goal analysis (e.g., 'How do I achieve 80% performance?')\n- Ask about your predicted performance with current scores\n- Ask about L6S concepts and best practices\n\nHow can I assist you today?"}
        ]
    if "bot" not in st.session_state:
        try:
            st.session_state.bot = L6SBrain()
        except Exception as e:
            st.error(f"Failed to initialize local graph: {e}")
    if "use_sidebar_values" not in st.session_state:
        st.session_state.use_sidebar_values = False
    if "conversation_context" not in st.session_state:
        st.session_state.conversation_context = {
            "last_topic": None,
            "last_goal": None,
            "last_strategy": None,
            "last_scores": None
        }

def main():
    init_session()
    bot = st.session_state.bot

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("Configuration")
        
        use_sidebar = st.checkbox(
            "Enable Manual Parameter Input", 
            value=st.session_state.use_sidebar_values,
            help="When enabled, sidebar values will be used if parameters are not specified in your message"
        )
        st.session_state.use_sidebar_values = use_sidebar
        
        if use_sidebar:
            st.info("Manual input mode active. Values below will be used unless overridden in chat.")
            
            sidebar_scores = {}
            
            with st.expander("Lean Parameters (IL1-IL7)", expanded=True):
                for i in range(1, 8):
                    k = f"IL{i}"
                    sidebar_scores[k] = st.slider(
                        CSF_DESCRIPTIONS[k], 
                        1, 5, 3, 
                        key=f"s_{k}",
                        help=f"Critical Success Factor: {CSF_DESCRIPTIONS[k]}"
                    )
                    
            with st.expander("Six Sigma Parameters (IS1-IS7)", expanded=True):
                for i in range(1, 8):
                    k = f"IS{i}"
                    sidebar_scores[k] = st.slider(
                        CSF_DESCRIPTIONS[k],
                        1, 5, 3,
                        key=f"s_{k}",
                        help=f"Critical Success Factor: {CSF_DESCRIPTIONS[k]}"
                    )

            with st.expander("Maturity Parameters (M1-M7)", expanded=False):
                for i in range(1, 8):
                    k = f"M{i}"
                    sidebar_scores[k] = st.slider(
                        CSF_DESCRIPTIONS[k],
                        1, 5, 3,  # Changed range to 1-5 for maturity factors to match IL/IS, default 3
                        key=f"s_{k}",
                        help=f"Critical Success Factor: {CSF_DESCRIPTIONS[k]}"
                    )

            st.session_state.sidebar_scores = sidebar_scores
        else:
            st.info("Manual input disabled. Provide CSF scores directly in your messages (e.g., 'IL1=4, IS1=3').")
            st.session_state.sidebar_scores = {}

        st.divider()
        st.caption(f"Local Graph: {len(bot.df) if bot.data_loaded else 0} samples | {bot.graph.number_of_nodes() if bot.data_loaded else 0} nodes")

        with st.expander("CSF Reference Guide"):
            st.markdown("**Lean Critical Success Factors (IL1-IL7)** (Rating: 1-5)")
            for i in range(1, 8):
                st.caption(f"IL{i}: {CSF_DESCRIPTIONS[f'IL{i}']}")
            st.markdown("**Six Sigma Critical Success Factors (IS1-IS7)** (Rating: 1-5)")
            for i in range(1, 8):
                st.caption(f"IS{i}: {CSF_DESCRIPTIONS[f'IS{i}']}")
            st.markdown("**Maturity Critical Success Factors (M1-M7)** (Rating: 1-5)")
            for i in range(1, 8):
                st.caption(f"M{i}: {CSF_DESCRIPTIONS[f'M{i}']}")

    # --- MAIN CHAT INTERFACE ---
    st.title("Lean Six Sigma Strategy Advisor")
    st.caption("Data-driven recommendations from 156 L6S implementation cases")

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if "content" in msg:
                st.markdown(msg["content"])
            if "chart" in msg:
                st.plotly_chart(msg["chart"], use_container_width=True)
            if "dataframe" in msg:
                st.dataframe(msg["dataframe"], use_container_width=True)

    # Handle User Input
    if prompt := st.chat_input("Enter your question or provide CSF scores..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            def set_status(message):
                status_placeholder.markdown(f"⏳ {message}")

            set_status("Analyzing your question...")
            with st.spinner("Analyzing database..."):
                try:

                    # PRIORITY: Check for CSF level description queries FIRST
                    is_csf_query, query_factor, query_level = bot.detect_csf_level_query(prompt)

                    if is_csf_query and query_factor:
                        response_text = ""

                        # Force English UI labels unless explicitly disabled
                        is_english = True
                        if not getattr(bot, "force_english", False):
                            # Detect if the query is in English by checking for common English words
                            query_lower = prompt.lower()
                            is_english_query = any(eng_word in query_lower for eng_word in ['what', 'level', 'mean', 'explain', 'how', 'does', 'the', 'is', 'are', 'and', 'or', 'for'])
                            is_french_query = any(fra_word in query_lower for fra_word in ['qu\'est', 'c\'est', 'quand', 'pourquoi', 'comment', 'quelle', 'quelle est', 'niveau', 'signifie', 's\'il', 'svp', 'est', 'pas', 'dans', 'pour'])

                            # If it looks more like French, or if both are present but French indicators are stronger, use French
                            is_english = is_english_query and not is_french_query

                        if query_level:
                            description = bot.get_csf_level_description(query_factor, query_level)
                            # Clean up the description to remove any trailing question marks that might be part of the text
                            description = description.rstrip('? ').strip() + '?'
                            factor_name = CSF_DESCRIPTIONS.get(query_factor, query_factor)

                            response_text = f"## {query_factor}: {factor_name}\n\n"
                            level_label = "Niveau" if not is_english else "Level"
                            response_text += f"**{level_label} {query_level}**:\n{description}\n\n"

                            if query_factor in bot.csf_level_descriptions:
                                all_levels = sorted(bot.csf_level_descriptions[query_factor].keys())
                                response_text += f"\n*This factor has {len(all_levels)} levels. "
                                response_text += f"Ask about other levels for more details.*"
                        else:
                            factor_name = CSF_DESCRIPTIONS.get(query_factor, query_factor)
                            response_text = f"## {query_factor}: {factor_name}\n\n"

                            if query_factor in bot.csf_level_descriptions:
                                levels = bot.csf_level_descriptions[query_factor]
                                level_label = "Niveau" if not is_english else "Level"
                                for level in sorted(levels.keys()):
                                    description = bot.translate_to_english(levels[level], ("desc", query_factor, level))
                                    description = description.rstrip('? ').strip() + '?'
                                    response_text += f"**{level_label} {level}**: {description}\n"
                            else:
                                response_text += f"No descriptions available for {query_factor}."

                        response_text = bot.ensure_english(response_text, ("response", "csf", prompt))
                        st.markdown(response_text)
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                        return  # Exit early

                    # Check if query is L6S-related
                    if not bot.is_l6s_related(prompt):
                        response_text = "I specialize in Lean Six Sigma (L6S) strategy recommendations and analysis. Your question doesn't appear to be related to L6S implementation, Critical Success Factors (CSFs), or performance optimization.\n\n"
                        response_text += "I can help you with:\n"
                        response_text += "- Strategy recommendations based on CSF scores (IL1-IL7 for Lean, IS1-IS7 for Six Sigma)\n"
                        response_text += "- Performance prediction and goal analysis\n"
                        response_text += "- L6S implementation best practices\n"
                        response_text += "- Comparison of implementation strategies (LM then SS, SS then LM, LM & SS)\n\n"
                        response_text += "Please ask a question related to Lean Six Sigma implementation."
                        
                        st.markdown(response_text)
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                        return
                
                    # Handle greetings and thanks
                    prompt_lower = prompt.lower().strip()
                    if prompt_lower in ['hello', 'hi', 'hey', 'hi there', 'hello there']:
                        response_text = "Hello! I'm your Lean Six Sigma Strategy Advisor. I can help you:\n\n"
                        response_text += "- Predict performance based on your CSF scores\n"
                        response_text += "- Recommend the best implementation strategy\n"
                        response_text += "- Analyze what it takes to reach specific performance goals\n\n"
                        response_text += "What would you like to know about your L6S implementation?"
                        st.markdown(response_text)
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                        return
                
                    if any(word in prompt_lower for word in ['thanks', 'thank you', 'thx', 'appreciate']):
                        response_text = "You're welcome! Feel free to ask if you have any other questions about Lean Six Sigma implementation or strategy recommendations."
                        st.markdown(response_text)
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                        return
                    
                    if prompt_lower in ['bye', 'goodbye', 'see you', 'exit']:
                        response_text = "Goodbye! Best of luck with your Lean Six Sigma implementation. Come back anytime you need strategy recommendations!"
                        st.markdown(response_text)
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                        return
                    
                    set_status("Processing your request...")
                    extracted_scores = bot.extract_scores_from_text(prompt)
                    target_goal = bot.analyze_goal(prompt)
                    
                    # Check for follow-up goal questions
                    if target_goal is None and st.session_state.conversation_context["last_topic"] == "goal_analysis":
                        follow_up_goal = bot.detect_follow_up_goal(prompt, st.session_state.conversation_context["last_goal"])
                        if follow_up_goal:
                            target_goal = follow_up_goal
                    
                    response_text = ""
                    extra_data = {}
                    
                    # --- SCENARIO 1: GOAL SEEKING ---
                    if target_goal:
                        previous_goal = st.session_state.conversation_context["last_goal"]
                        was_follow_up = (st.session_state.conversation_context["last_topic"] == "goal_analysis" 
                                        and previous_goal is not None
                                        and target_goal != previous_goal)
                        
                        st.session_state.conversation_context["last_topic"] = "goal_analysis"
                        st.session_state.conversation_context["last_goal"] = target_goal
                        
                        set_status("Building a performance roadmap...")
                        res = bot.get_reverse_recommendation(target_goal)
                        
                        if was_follow_up:
                            diff = target_goal - previous_goal
                            comparison = "higher" if diff > 0 else "lower"
                            response_text += f"*Comparing {target_goal}% (current) vs {previous_goal}% (previous): {abs(diff):.0f} percentage points {comparison}*\n\n"
                        
                        response_text += f"### Roadmap to {target_goal}% Performance\n\n"
                        response_text += f"{res['message']}\n\n"
                        
                        if res['sample_warning']:
                            response_text += "⚠️ **Note**: Limited historical data at this performance level. Recommendations based on top performers in the database.\n\n"
                        
                        response_text += f"**Recommended Strategy**: {res['strategy']}\n"
                        if res['strategy'] in STRATEGY_DESCRIPTIONS:
                            response_text += f"*{STRATEGY_DESCRIPTIONS[res['strategy']]}*\n\n"
                        response_text += f"**Organizations using this approach achieved**: {res['avg_perf']:.1f}% average performance\n"
                        response_text += f"**Based on**: {res['sample_size']} similar organizations\n\n"
                        
                        # Create visual comparison
                        categories = list(res['means'].keys())
                        target_values = [res['means'][k] for k in categories]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatterpolar(
                            r=target_values, 
                            theta=categories, 
                            fill='toself', 
                            name=f'Target Profile ({res["avg_perf"]:.1f}%)',
                            line=dict(color='#2ecc71', width=2)
                        ))
                        
                        if st.session_state.use_sidebar_values:
                            current_values = [st.session_state.sidebar_scores.get(k, 3) for k in categories]
                            fig.add_trace(go.Scatterpolar(
                                r=current_values, 
                                theta=categories, 
                                fill='toself', 
                                name='Your Current Profile',
                                line=dict(color='#e74c3c', width=2),
                                opacity=0.6
                            ))
                        
                        # Determine appropriate range for the radar chart based on max values
                        max_value = max(max(target_values), max(current_values)) if st.session_state.use_sidebar_values else max(target_values)
                        max_range = min(5, int(max_value * 1.1))  # Add slight buffer but cap at 5 for consistency
    
                        fig.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, max_range])),
                            title=f"CSF Profile for {target_goal}% Performance",
                            showlegend=True,
                            height=500
                        )
                        extra_data["chart"] = fig
                        
                        # Check if user has provided scores in the current message
                        has_extracted_scores = bool(extracted_scores)
                        has_sidebar_scores = st.session_state.use_sidebar_values
    
                        # Combine scores for comparison
                        current_scores = {}
                        if has_sidebar_scores:
                            current_scores.update(st.session_state.sidebar_scores)
                        if has_extracted_scores:
                            current_scores.update(extracted_scores)
    
                        # Add guidance when no current values are available
                        if not has_sidebar_scores and not has_extracted_scores:
                            response_text += "⚠️ **Note:** No current CSF values provided. The recommendations below are based on statistical models for achieving 80%+ performance. To get personalized recommendations, please enable 'Manual Parameter Input' in the sidebar or provide your current CSF scores (e.g., 'IL1=3, IS2=4').\n\n"
                        else:
                            # Show comparison with current scores
                            response_text += "#### Comparison with Your Current Scores\n\n"
    
                            # Calculate and display improvements needed
                            improvements_needed = []
                            for factor in categories:
                                current_value = current_scores.get(factor, 3)  # Default to 3 if not provided
                                target_value = res['means'][factor]
                                improvement = target_value - current_value
                                if improvement > 0:  # Only show factors that need improvement
                                    improvements_needed.append({
                                        'factor': factor,
                                        'current': current_value,
                                        'target': target_value,
                                        'improvement': improvement
                                    })
    
                            # Sort improvements by the amount of improvement needed (descending)
                            improvements_needed.sort(key=lambda x: x['improvement'], reverse=True)
    
                            if improvements_needed:
                                response_text += "#### 🎯 Priority Improvement Areas\n\n"
                                for i, imp in enumerate(improvements_needed[:5], 1):  # Show top 5
                                    direction = "Increase"
                                    response_text += f"{i}. **{imp['factor']}** ({CSF_DESCRIPTIONS[imp['factor']]}) from {imp['current']} to {imp['target']} (improvement of +{imp['improvement']})\n"
    
                                    # Add prescriptions for the levels that need to be achieved
                                    if hasattr(bot, 'csf_level_prescriptions') and imp['factor'] in bot.csf_level_prescriptions:
                                        # Show prescriptions for each level between current and target
                                        for level in range(int(imp['current']) + 1, int(imp['target']) + 1):
                                            if level in bot.csf_level_prescriptions[imp['factor']]:
                                                prescription = bot.csf_level_prescriptions[imp['factor']][level]
                                                response_text += f"   → *Action for level {level}*: {prescription}\n"
                                    response_text += "\n"
                            else:
                                response_text += "\n✅ Your current CSF profile is already close to the target! Focus on maintaining these levels.\n"
    
                        response_text += "#### Target CSF Profile\n\n"
    
                        response_text += "**Lean Critical Success Factors (IL)**:\n"
                        for k in [f'IL{i}' for i in range(1, 8)]:
                            response_text += f"- {k} ({CSF_DESCRIPTIONS[k]}): **{res['means'][k]}**"
                            if k in current_scores:
                                current = current_scores[k]
                                gap = res['means'][k] - current
                                if abs(gap) > 0.3:
                                    arrow = "↑" if gap > 0 else "↓"
                                    response_text += f" {arrow} (currently {current}, gap: {abs(gap):.1f})"
                            response_text += "\n"
    
                        response_text += "\n**Six Sigma Critical Success Factors (IS)**:\n"
                        for k in [f'IS{i}' for i in range(1, 8)]:
                            response_text += f"- {k} ({CSF_DESCRIPTIONS[k]}): **{res['means'][k]}**"
                            if k in current_scores:
                                current = current_scores[k]
                                gap = res['means'][k] - current
                                if abs(gap) > 0.3:
                                    arrow = "↑" if gap > 0 else "↓"
                                    response_text += f" {arrow} (currently {current}, gap: {abs(gap):.1f})"
                            response_text += "\n"
    
                        response_text += "\n**Maturity Critical Success Factors (M)**:\n"
                        for k in [f'M{i}' for i in range(1, 8)]:
                            response_text += f"- {k} ({CSF_DESCRIPTIONS[k]}): **{res['means'][k]}**"
                            if k in current_scores:
                                current = current_scores[k]
                                gap = res['means'][k] - current
                                if abs(gap) > 0.3:
                                    arrow = "↑" if gap > 0 else "↓"
                                    response_text += f" {arrow} (currently {current}, gap: {abs(gap):.1f})"
                            response_text += "\n"
    
                        # Calculate gaps based on combined scores
                        if has_sidebar_scores or has_extracted_scores:
                            gaps = []
                            for k in categories:
                                if k in current_scores:
                                    gap = res['means'][k] - current_scores[k]
                                    gaps.append((k, gap))
    
                            gaps.sort(key=lambda x: abs(x[1]), reverse=True)
    
                            priority_gaps = [g for g in gaps if abs(g[1]) > 0.5]
                            if priority_gaps:
                                response_text += f"\n#### 🎯 Top Priorities (gaps > 0.5 points)\n\n"
                                for i, (k, diff) in enumerate(priority_gaps[:5], 1):
                                    direction = "Increase" if diff > 0 else "Decrease"
                                    response_text += f"{i}. **{k}** ({CSF_DESCRIPTIONS[k]}): {direction} by {abs(diff):.1f} points\n"
                            else:
                                response_text += "\n✅ Your current CSF profile is already close to the target! Focus on maintaining these levels.\n"
    
                    # --- SCENARIO 2: STRATEGY RECOMMENDATION ---
                    elif extracted_scores and not any(phrase in prompt.lower() for phrase in ['niveau', 'level', 'description', 'détail', 'explain', 'signification', 'significance', 'what does', 'what mean', 'what stands for', 'signification', 'facteur', 'factor', 'critical success factor', 'means', 'mean']):
                        # Only go to strategy prediction if scores are provided AND it's not a question about level descriptions
    
                        if st.session_state.use_sidebar_values:
                            final_scores = st.session_state.sidebar_scores.copy()
                            final_scores.update(extracted_scores)
                        else:
                            final_scores = extracted_scores
    
                        if len(final_scores) >= 4:
                            input_vector, imputation_details, similar_cases_for_imputation = bot.predict_with_partial_params(final_scores)
                            res = bot.predict_strategy_and_perf(input_vector, final_scores, imputation_details)

                            response_text += "### Strategy Recommendation\n\n"

                            if extracted_scores:
                                # Format scores with descriptions and prescriptions
                                scores_with_details = []
                                for k, v in extracted_scores.items():
                                    desc = CSF_DESCRIPTIONS.get(k, k)
                                    scores_with_details.append(f"**{k}** ({desc}): **{v}**")

                                    # Add prescription for this level if available
                                    if hasattr(bot, 'csf_level_prescriptions') and k in bot.csf_level_prescriptions:
                                        prescription = bot.csf_level_prescriptions[k].get(v, "No prescription available")
                                        scores_with_details.append(f"  -> *Action*: {prescription}")
                                    scores_with_details.append("")  # Add blank line for better readability

                                response_text += f"**Parameters Provided**:\n" + "\n".join(scores_with_details) + "\n"

                            if len(final_scores) < 14:
                                response_text += f"**Note**: Prediction based on {len(final_scores)}/14 CSF parameters. Missing values estimated using statistical models.\n\n"

                            st.session_state.conversation_context["last_topic"] = "strategy_rec"
                            st.session_state.conversation_context["last_strategy"] = res['strategy']
                            st.session_state.conversation_context["last_scores"] = final_scores

                            # Handle failure warning
                            if res.get('failure_warning', False):
                                response_text += f"⚠️ **WARNING**: Based on your parameters, the most likely outcome is FAILURE. You need to significantly improve your CSFs to make L6S viable.\n\n"
                                response_text += f"**Predicted Strategy Outcome**: {res['strategy']} (⚠️ HIGH RISK)\n"
                                response_text += f"**Predicted Performance**: {res['performance']:.1f}%\n"
                                response_text += f"**Confidence Level**: {res['confidence']:.1f}%\n\n"
                            else:
                                response_text += f"**Recommended Strategy**: {res['strategy']}\n"
                                if res['strategy'] in STRATEGY_DESCRIPTIONS:
                                    response_text += f"*{STRATEGY_DESCRIPTIONS[res['strategy']]}*\n\n"
                                response_text += f"**Predicted Performance**: {res['performance']:.1f}%\n"
                                response_text += f"**Confidence Level**: {res['confidence']:.1f}%\n\n"

                            if res['confidence'] < 60:
                                response_text += "**Low Confidence Warning**: The model has lower confidence in this prediction. Providing more CSF parameters would improve accuracy.\n\n"

                            if len(final_scores) < 14 and imputation_details:
                                with st.expander("View Parameter Estimation Details"):
                                    st.markdown("**How missing parameters were estimated:**")
                                    for param, detail in imputation_details.items():
                                        if param not in final_scores:
                                            st.caption(f"{param} ({CSF_DESCRIPTIONS[param]}): {detail}")

                            # Removed similar historical cases section to protect data confidentiality
                            response_text += f"\n**Statistical Summary**:\n"
                            response_text += f"- Mean Performance: {res['similar_cases']['performance'].mean():.1f}%\n"
                            response_text += f"- Performance Range: {res['similar_cases']['performance'].min():.1f}% - {res['similar_cases']['performance'].max():.1f}%\n"
                        else:
                            response_text += "### Insufficient Parameter Information\n\n"
                            response_text += f"I detected {len(final_scores)} CSF parameter(s) from your message"
                            if extracted_scores:
                                # Format scores with descriptions and prescriptions
                                scores_with_details = []
                                for k, v in extracted_scores.items():
                                    desc = CSF_DESCRIPTIONS.get(k, k)
                                    scores_with_details.append(f"**{k}** ({desc}): **{v}**")

                                    # Add prescription for this level if available
                                    if hasattr(bot, 'csf_level_prescriptions') and k in bot.csf_level_prescriptions:
                                        prescription = bot.csf_level_prescriptions[k].get(v, "No prescription available")
                                        scores_with_details.append(f"  -> *Action*: {prescription}")
                                    scores_with_details.append("")  # Add blank line for better readability

                                response_text += f":\n" + "\n".join(scores_with_details)
                            response_text += "\n\n"
                            response_text += "For reliable strategy recommendations, I need at least 4-5 CSF parameters. The more parameters you provide, the more accurate the prediction.\n\n"
                            response_text += "**Options**:\n"
                            response_text += "1. Provide more parameters in your message (e.g., 'IL1=4, IL2=3, IL3=4, IS1=5, IS2=3')\n"
                            response_text += "2. Enable 'Manual Parameter Input' in the sidebar to set values\n\n"
                            response_text += "**Most impactful parameters to provide**:\n"
                            importance = dict(zip(bot.feature_names, bot.rf_regressor.feature_importances_))
                            top_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
                            for param, imp in top_params:
                                response_text += f"- {param} ({CSF_DESCRIPTIONS[param]})\n"
    
                    # --- SCENARIO 3: GENERAL L6S CONVERSATION ---
                    else:
                        # Check if it's a question about level descriptions first, to override other logic
                        is_description_question = any(word in prompt.lower() for word in ['niveau', 'level', 'description', 'détail', 'explain', 'signification', 'significance', 'signification', 'facteur', 'factor', 'critical success factor', 'means', 'mean', 'what does']) and any(word in ['level', 'niveau'] for word in prompt.lower().split())
    
                        if is_description_question and bot.llm_available:
                            # Force this to LLM path which will include CSF descriptions
                            # This will be handled in the LLM section below
                            pass
                        elif any(word in prompt.lower() for word in ['my performance', 'performance score', 'what performance', 'how well']):
                            if st.session_state.use_sidebar_values and all(k in st.session_state.sidebar_scores for k in bot.feature_names):
                                final_scores = st.session_state.sidebar_scores.copy()
                                input_vector = [final_scores[f] for f in bot.feature_names]

                                set_status("Predicting performance and strategy...")
                                res = bot.predict_strategy_and_perf(input_vector, final_scores)

                                response_text = "### Your Predicted Performance\n\n"
                                response_text += f"Based on your current CSF scores from the sidebar:\n\n"

                                # Handle failure warning for performance queries too
                                if res.get('failure_warning', False):
                                    response_text += f"⚠️ **WARNING**: Based on your parameters, the most likely outcome is FAILURE. You need to significantly improve your CSFs to make L6S viable.\n\n"
                                    response_text += f"**Predicted Performance**: {res['performance']:.1f}%\n"
                                    response_text += f"**Predicted Strategy Outcome**: {res['strategy']} (⚠️ HIGH RISK)\n"
                                    response_text += f"**Confidence Level**: {res['confidence']:.1f}%\n\n"
                                else:
                                    response_text += f"**Predicted Performance**: {res['performance']:.1f}%\n"
                                    response_text += f"**Recommended Strategy**: {res['strategy']}\n"
                                    if res['strategy'] in STRATEGY_DESCRIPTIONS:
                                        response_text += f"*{STRATEGY_DESCRIPTIONS[res['strategy']]}*\n\n"
                                    response_text += f"**Confidence Level**: {res['confidence']:.1f}%\n\n"

                                # Format scores with descriptions and prescriptions
                                scores_with_details = []
                                for k, v in final_scores.items():
                                    desc = CSF_DESCRIPTIONS.get(k, k)
                                    scores_with_details.append(f"- **{k}** ({desc}): **{v}**")

                                    # Add prescription for this level if available
                                    if hasattr(bot, 'csf_level_prescriptions') and k in bot.csf_level_prescriptions:
                                        prescription = bot.csf_level_prescriptions[k].get(v, "No prescription available")
                                        scores_with_details.append(f"  -> *Action*: {prescription}")
                                    scores_with_details.append("")  # Add blank line for better readability

                                response_text += "This prediction is based on your Critical Success Factor scores:\n"
                                response_text += "\n".join(scores_with_details) + "\n\n"
                                response_text += "To improve your performance, consider providing specific CSF values or asking for goal-based recommendations."
                            else:
                                response_text = "To calculate your performance prediction, I need your Critical Success Factor (CSF) scores.\n\n"
                                response_text += "**Option 1**: Enable 'Manual Parameter Input' in the sidebar and set your CSF values (IL1-IL7 for Lean, IS1-IS7 for Six Sigma)\n\n"
                                response_text += "**Option 2**: Provide your CSF scores in the chat (e.g., 'IL1=4, IL2=5, IL3=3, IS1=4, IS2=3...')\n\n"
                                response_text += "The CSF scores represent your organization's maturity level (1-5) in areas like Leadership Engagement, Cultural Change, Communication, Training, Tools & Techniques, Employee Involvement, and Expertise."
    
                        elif bot.llm_available:
                            # Detect language and maintain consistency
                            detected_lang = bot.detect_language(prompt)
                            bot.detected_language = detected_lang  # Update the bot's detected language
    
                            # Generate comprehensive graph RAG context
                            graph_context = bot.generate_graph_rag_context(prompt, extracted_scores if extracted_scores else (st.session_state.sidebar_scores if st.session_state.use_sidebar_values else None))
    
                            # Get query-specific graph insights
                            query_insights = bot.get_graph_insights_for_query(prompt, extracted_scores if extracted_scores else (st.session_state.sidebar_scores if st.session_state.use_sidebar_values else None))
    
                            context_str = "You are a Lean Six Sigma (L6S) implementation expert. Only answer questions related to L6S, Critical Success Factors, implementation strategies, performance optimization, and related topics. Always respond in English, even if the user writes in French. "
                            context_str += f"Database contains {len(bot.df)} L6S implementation cases showing that CSF scores (Critical Success Factors) are INPUT variables that PREDICT performance outcomes. "
                            context_str += "The 14 CSF scores (IL1-IL7 for Lean, IS1-IS7 for Six Sigma) are ratings from 1-5 that measure organizational maturity in: Leadership, Culture, Communication, Training, Tools, Employee Involvement, and Expertise. "
                            context_str += "These CSFs are used to PREDICT performance percentage (typically 30-85%) and recommend implementation strategies. "
                            context_str += "Available strategies: LM then SS (Lean then Six Sigma), SS then LM (Six Sigma then Lean), LM & SS (simultaneous implementation). "
                            context_str += "IMPORTANT: Do not make up performance calculations. CSF scores are inputs, performance % is the predicted output. "
    
                            # Check if the user is asking for specific CSF level descriptions
                            prompt_lower = prompt.lower()
                            csf_descriptions = []
    
                            # More comprehensive detection for CSF level queries - check multiple ways users might ask
                            is_csf_description_query = any(word in prompt_lower for word in ['niveau', 'level', 'description', 'détail', 'explain', 'signification', 'significance', 'signification', 'facteur', 'factor', 'critical success factor', 'means', 'mean', 'what mean by', 'what stands for', 'what signifie', 'explain level', 'describe level', 'level means', 'level signifie']) or \
                                                       any(re.search(phrase, prompt_lower) for phrase in [r'what.*mean.*by', r'what.*stands.*for', r'what.*signifie', r'explain.*level', r'describe.*level', r'level.*means', r'level.*signifie'])
    
                            # Additional check: if any factor is mentioned with 'mean' or 'what', it's likely a description query
                            has_factor_and_meaning = any(re.search(rf'what.*mean.*by.*{factor.lower()}', prompt_lower) or
                                                         re.search(rf'{factor.lower()}.*mean', prompt_lower) or
                                                         re.search(rf'what.*{factor.lower()}.*mean', prompt_lower) for factor in bot.feature_names)
    
                            # Even more aggressive: check if factor and 'mean' or 'level' appear together in the query
                            has_factor_and_meaning_aggressive = any(
                                factor.lower() in prompt_lower and any(word in prompt_lower for word in ['mean', 'means', 'signifie', 'signification', 'level', 'prescription', 'action', 'recommendation', 'should do', 'do to', 'what to do', 'how to'])
                                for factor in bot.feature_names
                            )
    
                            if is_csf_description_query or has_factor_and_meaning or has_factor_and_meaning_aggressive:
                                # Look for factor patterns with possible variations
                                for factor in bot.feature_names:
                                    # Check for various patterns where factor and level might be mentioned
                                    factor_mentioned = factor.lower() in prompt_lower or factor in prompt
    
                                    if factor_mentioned:
                                        # Check if a specific level is mentioned near this factor
                                        level_found = False
                                        for level in range(1, 6):  # Levels 1-5
                                            # Check for all possible patterns like "IL2 level 5", "level 5 il2", "il2 = 5", etc.
                                            patterns_to_check = [
                                                f"{factor.lower()} level {level}",
                                                f"{factor.lower()} niveau {level}",
                                                f"level {level}.*{factor.lower()}",
                                                f"niveau {level}.*{factor.lower()}",
                                                f"{factor.lower()}[\\s]*=[\\s]*{level}",  # Handles "il2=5", "IL2 = 5", etc.
                                                f"{factor.lower()}[\\s]+{level}",    # Handles "il2 5"
                                                f"{factor.lower()}.*mean.*{level}",
                                                f"{factor.lower()}.*what.*{level}",
                                                f"what.*{factor.lower()}.*{level}",
                                                f"{level}.*{factor.lower()}",      # Handles "5 il2"
                                                f"{factor.lower()}.*prescription",  # Check for prescription-related queries
                                                f"{factor.lower()}.*action",       # Check for action-related queries
                                                f"{factor.lower()}.*recommendation", # Check for recommendation-related queries
                                                f"{factor.lower()}.*should do",    # Check for "should do" queries
                                                f"{factor.lower()}.*do to",        # Check for "do to" queries
                                                f"{factor.lower()}.*what to do",   # Check for "what to do" queries
                                                f"{factor.lower()}.*how to"        # Check for "how to" queries
                                            ]
    
                                            for pattern in patterns_to_check:
                                                if re.search(pattern.lower(), prompt_lower):
                                                    description = bot.get_csf_level_description(factor, level)
                                                    if "not available" not in description.lower():
                                                        csf_descriptions.append(f"{factor} Level {level}: {description}")
                                                    level_found = True
                                                    break  # Found the level, no need to check other patterns
    
                                        # Additional check: if factor and level and 'mean'/'what' are present, try to get description anyway
                                        if not level_found and ('mean' in prompt_lower or 'what' in prompt_lower or 'prescription' in prompt_lower or 'should' in prompt_lower or 'action' in prompt_lower or 'how to' in prompt_lower or 'what to do' in prompt_lower):
                                            for level in range(1, 6):
                                                # Look for this specific level in the prompt more broadly
                                                if str(level) in re.findall(r'\d+', prompt_lower):
                                                    description = bot.get_csf_level_description(factor, level)
                                                    if "not available" not in description.lower():
                                                        csf_descriptions.append(f"{factor} Level {level}: {description}")
                                                        level_found = True
                                                        break
    
                                            if level_found:
                                                break  # Found the level for this factor, move to next factor
    
                                        # If no specific level found but factor was mentioned, get all levels
                                        if not level_found:
                                            for level in range(1, 6):
                                                description = bot.get_csf_level_description(factor, level)
                                                if "not available" not in description.lower():
                                                    csf_descriptions.append(f"{factor} Level {level}: {description}")
    
                            if csf_descriptions:
                                context_str += f"\n\nCSF LEVEL DESCRIPTIONS FROM DATABASE:\n" + "\n".join(csf_descriptions) + "\n\n"
    
                            # Add comprehensive graph RAG context
                            context_str += f"\n\nGRAPH RAG CONTEXT from local analysis:\n{graph_context}\n\n"
    
                            # Add query-specific insights
                            if query_insights:
                                context_str += f"QUERY-SPECIFIC GRAPH INSIGHTS:\n{query_insights}\n\n"
    
                            if st.session_state.use_sidebar_values:
                                context_str += f"User's current sidebar CSF scores: {json.dumps(st.session_state.sidebar_scores)}\n"
    
                            full_prompt = f"{context_str}\n\nUser Question: {prompt}\n\nProvide a concise, accurate answer about L6S topics in English only. Do not include French. IMPORTANT: If CSF LEVEL DESCRIPTIONS FROM DATABASE are provided, use those exact descriptions verbatim in your response. Do not rephrase, summarize, or create new descriptions. Simply include the exact text from the CSF LEVEL DESCRIPTIONS section. The descriptions include both the level meaning and specific action items (prescriptions) that organizations should take to achieve each level. Reference the graph RAG context and query-specific insights when relevant. Do not invent calculations or metrics. Only include predictions about performance/strategy if specifically asked for those (not when asking about level descriptions)."
                            set_status("Generating response...")
                            response_text = bot.llm.invoke(full_prompt)
                            response_text = bot.ensure_english(response_text, ("llm", prompt))
                        else:
                            response_text = "I can help you with:\n\n"
                            response_text += "**Strategy Recommendations**: Provide CSF scores (e.g., 'What strategy if IL1=4, IS1=3?')\n\n"
                            response_text += "**Goal Analysis**: Ask about achieving specific performance targets (e.g., 'How do I achieve 80%?')\n\n"
                            response_text += "**L6S Implementation**: Ask about Lean and Six Sigma methodologies, best practices, and implementation approaches\n\n"
                            response_text += "**CSF Analysis**: Learn about the 14 Critical Success Factors (IL1-IL7 for Lean, IS1-IS7 for Six Sigma)\n\n"
                            response_text += "All recommendations are based on analysis of 156 real L6S implementation cases."
    
                    response_text = bot.ensure_english(response_text, ("response", prompt))
                    # Render Response
                    st.markdown(response_text)
                    if "chart" in extra_data:
                        st.plotly_chart(extra_data["chart"], use_container_width=True)
                    if "dataframe" in extra_data:
                        st.dataframe(extra_data["dataframe"], use_container_width=True)
                    
                    # Save to history
                    msg_payload = {"role": "assistant", "content": response_text}
                    if "chart" in extra_data: 
                        msg_payload["chart"] = extra_data["chart"]
                    if "dataframe" in extra_data: 
                        msg_payload["dataframe"] = extra_data["dataframe"]
                    st.session_state.messages.append(msg_payload)
                finally:
                    status_placeholder.empty()

if __name__ == "__main__":
    main()
