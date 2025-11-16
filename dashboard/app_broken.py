import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import socket libraries for server communication
import socket
import pickle
import json
import threading
import time
from collections import defaultdict
import logging
import traceback

# Global exception handler to capture unhandled exceptions and write them to a log
import sys
def _dashboard_excepthook(exc_type, exc_value, exc_tb):
    tb = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
    try:
        log_path = os.path.join(os.path.dirname(__file__), 'dashboard_error.log')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write('\n' + ('='*80) + '\n')
            f.write('Unhandled exception in dashboard/app.py:\n')
            f.write(tb)
            f.write('\n' + ('='*80) + '\n')
    except Exception:
        pass
    try:
        # Try to show a user-friendly error in Streamlit
        st.error('The dashboard encountered an unexpected error. See dashboard_error.log for details.')
        st.text(tb)
    except Exception:
        print('Unhandled exception in dashboard/app.py:')
        print(tb)

sys.excepthook = _dashboard_excepthook

# Page config
st.set_page_config(
    page_title="FairFinance Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import sys
import os
import socket
import pickle
import json
import threading
import time
from collections import defaultdict

# Page config
st.set_page_config(
    page_title="FairFinance Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Metric cards */
    .stMetricCard {
        background-color: #1E2027 !important;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        padding: 1rem;
        border: 1px solid #2E2E2E;
    }
    
    /* Navigation menu */
    .stSelectbox label,
    .stMultiSelect label {
        color: #FAFAFA !important;
    }
    
    /* Tables */
    .dataframe {
        background-color: #1E2027 !important;
        color: #FAFAFA !important;
    }
    
    /* Text elements */
    h1, h2, h3, h4, h5, h6, p {
        color: #FAFAFA !important;
    }
    
    /* Cards and containers */
    div[data-testid="stVerticalBlock"] > div {
        background-color: #1E2027;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: #1E2027 !important;
        color: #FAFAFA !important;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background-color: #262931 !important;
    }
    
    /* Adjust main content padding */
    .main {
        padding: 1rem 2rem;
    }
    
    /* Charts background */
    .js-plotly-plot {
        background-color: #1E2027 !important;
    }
    
    /* Grid styling */
    .grid {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# Navigation
selected = option_menu(
    menu_title=None,
    options=["Dashboard", "Model Analysis", "Fairness Metrics"],
    icons=["house", "graph-up", "shield-check"],
    orientation="horizontal",
)

# Create a connection to track metrics
import socket
import pickle
import json
import threading
import time
from collections import defaultdict

class MetricsTracker:
    def __init__(self):
        self.reset_metrics()
        self.lock = threading.Lock()

    def reset_metrics(self):
        self.metrics = {
            'Round': [],
            'Accuracy': [],
            'FairFinance_Score': [],
            'Fairness': [],
            'Communication_Efficiency': [],
            'Robustness': []
        }
        self.current_round = 0
        self.active_clients = 0

    def update_metrics(self, response):
        with self.lock:
            round_num = response.get('round', 0)
            
            # Handle metrics arrays from server
            metrics_updated = False
            
            # Get the number of new data points
            if 'accuracy' in response:
                # Server returns the full history as lists. Append any tail
                # entries we don't have yet. Assign sensible round numbers
                # per new data point instead of repeating the server 'round'
                try:
                    new_points = len(response['accuracy'])
                except Exception:
                    new_points = 0

                if new_points > len(self.metrics['Round']):
                    start_idx = len(self.metrics['Round'])
                    end_idx = new_points

                    # Add new rounds. Use incremental round ids based on
                    # how many points we already have so the x-axis
                    # increases monotonically.
                    for i in range(start_idx, end_idx):
                        round_label = len(self.metrics['Round']) + 1
                        self.metrics['Round'].append(round_label)
                        self.metrics['Accuracy'].append(float(response['accuracy'][i]))
                        self.metrics['FairFinance_Score'].append(float(response['fairness_score'][i]))
                        self.metrics['Fairness'].append(float(response['fairness'][i]))
                        self.metrics['Communication_Efficiency'].append(float(response['communication'][i]))
                        self.metrics['Robustness'].append(float(response['robustness'][i]))
                        metrics_updated = True
                else:
                    # Fallback: if server returned a single/latest scalar
                    # wrapped in the response (older clients), append that
                    # single point if it's newer than what we have.
                    try:
                        # scalar fields may be present as floats
                        if isinstance(response.get('accuracy'), (int, float)):
                            round_label = len(self.metrics['Round']) + 1
                            self.metrics['Round'].append(round_label)
                            self.metrics['Accuracy'].append(float(response.get('accuracy', 0.0)))
                            self.metrics['FairFinance_Score'].append(float(response.get('fairness_score', 0.0)))
                            self.metrics['Fairness'].append(float(response.get('fairness', 0.0)))
                            self.metrics['Communication_Efficiency'].append(float(response.get('communication', 0.95)))
                            self.metrics['Robustness'].append(float(response.get('robustness', 0.9)))
                            metrics_updated = True
                    except Exception:
                        pass
            
            if metrics_updated:
                print(f"[METRICS] Updated metrics to round {round_num}")
                self.current_round = round_num
                self.active_clients = response.get('active_clients', 0)
            else:
                print(f"[METRICS] No new metrics for round {round_num}")

    def get_metrics_df(self):
        with self.lock:
            return pd.DataFrame(self.metrics)

# Initialize a module-level MetricsTracker singleton. Using a module-level
# singleton (instead of storing the tracker in `st.session_state`) ensures the
# background polling thread and any Streamlit session share the same object
# even after hot-reloads. We keep a small global marker in `globals()` so the
# object persists across reloads.
if '_METRICS_TRACKER_SINGLETON' not in globals():
    _METRICS_TRACKER_SINGLETON = MetricsTracker()
    globals()['_METRICS_TRACKER_SINGLETON'] = _METRICS_TRACKER_SINGLETON
METRICS_TRACKER = globals()['_METRICS_TRACKER_SINGLETON']

# Replace direct session_state access in the background thread with a module-level
# threading.Event and a module-level reference to the MetricsTracker instance.
running_event = threading.Event()

def update_metrics_thread():
    """Background thread that queries the server for metrics and stores them in
    the module-level METRICS_TRACKER instance. This avoids accessing
    `st.session_state` from a background thread which causes Streamlit
    'missing ScriptRunContext' warnings and can lead to the app stopping.
    """
    running_event.set()
    error_count = 0

    def recv_msg(sock):
        size_data = sock.recv(4)
        if not size_data:
            return None
        size = int.from_bytes(size_data, 'big')
        data = bytearray()
        while len(data) < size:
            chunk = sock.recv(min(size - len(data), 8192))
            if not chunk:
                return None
            data.extend(chunk)
        return data

    def send_msg(sock, data):
        size = len(data)
        sock.send(size.to_bytes(4, 'big'))
        sock.sendall(data)

    while running_event.is_set():
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect(('localhost', 8082))
            sock.settimeout(None)
            print("[DASHBOARD] Connected to server")
            error_count = 0  # Reset error count on successful connection

            while running_event.is_set():
                try:
                    message = {'type': 'get_metrics'}
                    send_msg(sock, pickle.dumps(message))
                    data = recv_msg(sock)

                    if data:
                        response = pickle.loads(data)
                        if response.get('type') == 'metrics':
                            # Update the module-level metrics tracker (thread-safe)
                            METRICS_TRACKER.update_metrics(response)
                    time.sleep(1)
                except Exception as e:
                    print(f"Error in metrics loop: {e}")
                    break

            try:
                sock.close()
            except:
                pass
        except Exception as e:
            print(f"Could not connect to server: {e}")
            error_count += 1
            if error_count > 5:  # After 5 failed attempts
                print("[DASHBOARD] Too many connection failures, resetting metrics")
                METRICS_TRACKER.reset_metrics()
                error_count = 0
            time.sleep(5)

# Start metrics update thread once per python process. We guard with a global
# flag so hot-reloads don't spawn duplicate threads.
if '_DASH_METRICS_THREAD_STARTED' not in globals():
    _DASH_METRICS_THREAD_STARTED = True
    globals()['_DASH_METRICS_THREAD_STARTED'] = True
    thread = threading.Thread(target=update_metrics_thread, daemon=True)
    thread.start()

def get_metrics_df():
    return METRICS_TRACKER.get_metrics_df()

if selected == "Dashboard":
    st.title("🏦 FairFinance Dashboard")
    st.markdown("### Federated Learning Evaluation Framework")    # Metric Cards
    col1, col2, col3 = st.columns(3)
    # Get latest metrics
    df = get_metrics_df()
    if not df.empty and len(df) > 0:
        with col1:
            st.metric(
                "FairFinance Score", 
                f"{df['FairFinance_Score'].iloc[-1]*100:.2f}%",
                None if len(df) == 1 else f"{(df['FairFinance_Score'].iloc[-1] - df['FairFinance_Score'].iloc[-2])*100:+.1f}%"
            )
        with col2:
            st.metric(
                "Model Accuracy", 
                f"{df['Accuracy'].iloc[-1]*100:.2f}%",
                None if len(df) == 1 else f"{(df['Accuracy'].iloc[-1] - df['Accuracy'].iloc[-2])*100:+.1f}%"
            )
        with col3:
            st.metric(
                "Fairness Score", 
                f"{df['Fairness'].iloc[-1]*100:.2f}%",
                None if len(df) == 1 else f"{(df['Fairness'].iloc[-1] - df['Fairness'].iloc[-2])*100:+.1f}%"
            )
    else:
        with col1:
            st.metric("FairFinance Score", "Waiting...", "")
        with col2:
            st.metric("Model Accuracy", "Waiting...", "")
        with col3:
            st.metric("Fairness Score", "Waiting...", "")

    # Charts
    st.markdown("### Training Progress")
    tab1, tab2 = st.tabs(["Performance Metrics", "Component Scores"])
    
    with tab1:
        df = get_metrics_df()
        if not df.empty:
            # Create figure with secondary y-axis
            from plotly.subplots import make_subplots
            # Make a copy and convert fraction metrics to percentages for plotting
            df_plot = df.copy()
            if 'Accuracy' in df_plot.columns:
                df_plot['Accuracy'] = df_plot['Accuracy'] * 100
            if 'FairFinance_Score' in df_plot.columns:
                df_plot['FairFinance_Score'] = df_plot['FairFinance_Score'] * 100

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add traces
            fig.add_trace(
                go.Scatter(x=df_plot["Round"], y=df_plot["Accuracy"], name="Accuracy",
                          line=dict(color="#636efa", width=2)),
                secondary_y=False,
            )

            fig.add_trace(
                go.Scatter(x=df_plot["Round"], y=df_plot["FairFinance_Score"], name="FairFinance Score",
                          line=dict(color="#ef553b", width=2)),
                secondary_y=True,
            )
            
            # Add figure title
            fig.update_layout(
                title="Model Performance Over Time",
                template="plotly_dark",
                plot_bgcolor="#1E2027",
                paper_bgcolor="#1E2027",
                font_color="#FAFAFA",
                height=400,
                hovermode="x unified",
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(30,32,39,0.8)"
                )
            )
            
            # Set y-axes titles
            fig.update_yaxes(
                title_text="Accuracy (%)",
                range=[0, 100],
                gridcolor="#2E2E2E",
                secondary_y=False
            )
            fig.update_yaxes(
                title_text="FairFinance Score (%)",
                range=[0, 100],
                gridcolor="#2E2E2E",
                secondary_y=True
            )
            
            # Update x-axis
            fig.update_xaxes(
                title_text="Training Round",
                gridcolor="#2E2E2E",
                showgrid=True
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Waiting for training data...")
    
    with tab2:
        df = get_metrics_df()
        if not df.empty:
            # Convert metrics to percentages for display
            df_display = df.copy()
            for col in ['Fairness', 'Communication_Efficiency', 'Robustness']:
                df_display[col] = df_display[col] * 100
                
            fig = go.Figure()
            
            # Add traces for each component
            fig.add_trace(
                go.Scatter(x=df_display["Round"], y=df_display["Fairness"], 
                          name="Fairness", line=dict(color="#00cc96", width=2))
            )
            fig.add_trace(
                go.Scatter(x=df_display["Round"], y=df_display["Communication_Efficiency"],
                          name="Communication", line=dict(color="#ab63fa", width=2))
            )
            fig.add_trace(
                go.Scatter(x=df_display["Round"], y=df_display["Robustness"],
                          name="Robustness", line=dict(color="#ffa15a", width=2))
            )
            
            # Update layout
            fig.update_layout(
                title="Component Scores Over Time",
                template="plotly_dark",
                plot_bgcolor="#1E2027",
                paper_bgcolor="#1E2027",
                font_color="#FAFAFA",
                height=400,
                hovermode="x unified",
                yaxis=dict(
                    title="Score (%)",
                    range=[0, 100],
                    gridcolor="#2E2E2E"
                ),
                xaxis=dict(
                    title="Training Round",
                    gridcolor="#2E2E2E"
                ),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(30,32,39,0.8)"
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Waiting for training data...")

elif selected == "Model Analysis":
    st.title("Model Analysis")
    
    # Model Performance Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("##### Training Time\n⏱️ 15.3 minutes")
    with col2:
        st.success("##### Model Size\n💾 2.4 MB")
    with col3:
        st.warning("##### Active Clients\n👥 3/5 Connected")

    # Model Architecture
    st.markdown("### Model Architecture")
    st.code("""
    FraudDetector(
        (fc1): Linear(in=30, out=64)
        (fc2): Linear(64, 32)
        (fc3): Linear(32, 1)
        (relu): ReLU()
        (sigmoid): Sigmoid()
    )
    """)

elif selected == "Fairness Metrics":
    st.title("Fairness Analysis")
    
    # Fairness Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Group Performance")
        group_data = pd.DataFrame({
            "Group": ["High Income", "Medium Income", "Low Income"],
            "Accuracy": [0.89, 0.87, 0.85],
            "False Positive Rate": [0.11, 0.13, 0.15]
        })
        st.dataframe(group_data, use_container_width=True)
        
    with col2:
        st.markdown("### Fairness Components")
        # Create radar chart using go.Figure instead of px.radar
        import plotly.graph_objects as go

        # Create default radar chart with initial values
        fig = go.Figure(data=go.Scatterpolar(
            r=[80, 95, 90, 75],  # Initial values
            theta=['Fairness', 'Communication', 'Robustness', 'Accuracy'],
            fill='toself',
            line_color='#00cc96'
        ))

        df = get_metrics_df()
        if not df.empty:
            df_last = df.iloc[-1:]
            df_last_display = df_last.copy()
            for col in ['Accuracy', 'FairFinance_Score', 'Fairness', 'Communication_Efficiency', 'Robustness']:
                if col in df_last_display.columns:
                    df_last_display[col] = df_last_display[col] * 100
            
            # Update radar chart with actual values if available
            if all(col in df_last_display.columns for col in ['Fairness', 'Communication_Efficiency', 'Robustness', 'Accuracy']):
                fig.update_traces(
                    r=[df_last_display['Fairness'].iloc[0],
                       df_last_display['Communication_Efficiency'].iloc[0],
                       df_last_display['Robustness'].iloc[0],
                       df_last_display['Accuracy'].iloc[0]]
                )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    gridcolor="#2E2E2E"
                ),
                bgcolor="#1E2027"
            ),
            showlegend=False,
            title="FairFinance Score Components",
            template="plotly_dark",
            paper_bgcolor="#1E2027"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Fairness Insights")
    st.info("""
    ℹ️ The current model shows a bias variance of 4% across income groups.
    Recommendations:
    - Implement data balancing techniques
    - Adjust model weights for underrepresented groups
    - Monitor group-wise performance regularly
    """)

import warnings
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
