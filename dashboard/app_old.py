import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import socket, pickle, threading, time, sys, os, logging

st.set_page_config(page_title="FairFinance Dashboard", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    .stApp { 
        background: linear-gradient(135deg, #0E1117 0%, #1A1F2E 100%);
        color: #FAFAFA;
    }
    
    .stMetricCard { 
        background: linear-gradient(135deg, #1E2847 0%, #2D3748 100%) !important;
        border-radius: 15px !important;
        border: 1px solid #3D4A5C !important;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3) !important;
        padding: 20px !important;
    }
    
    .stMetricLabel {
        color: #9CA3AF !important;
        font-size: 14px !important;
    }
    
    .stMetricValue {
        color: #60A5FA !important;
        font-size: 32px !important;
        font-weight: bold !important;
    }
    
    .dataframe { 
        background-color: #1E2847 !important;
        color: #FAFAFA !important;
        border-color: #3D4A5C !important;
    }
    
    h1 { 
        color: #FAFAFA !important;
        font-size: 42px !important;
        font-weight: 800 !important;
        margin-bottom: 10px !important;
    }
    
    h2 { 
        color: #E0E7FF !important;
        font-size: 28px !important;
        font-weight: 700 !important;
        margin-bottom: 15px !important;
    }
    
    h3 { 
        color: #C7D2FE !important;
        font-size: 20px !important;
        font-weight: 600 !important;
    }
    
    p { 
        color: #D1D5DB !important;
        font-size: 16px !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: transparent;
        border-bottom: 2px solid #3D4A5C;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        background-color: #1E2847;
        color: #9CA3AF;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: #FAFAFA !important;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1E2847 0%, #2D3748 100%);
        border: 2px solid #3B82F6;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.15);
    }
    
    hr {
        border: none;
        border-top: 2px solid #3D4A5C;
        margin: 30px 0;
    }
    
    .section-title {
        color: #FAFAFA;
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 3px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

class MetricsTracker:
    def __init__(self):
        self.reset_metrics()
        self.lock = threading.Lock()
    
    def reset_metrics(self):
        self.metrics = {'Round': [], 'Accuracy': [], 'FairFinance_Score': [], 'Fairness': [], 'Communication_Efficiency': [], 'Robustness': []}
        self.current_round = 0
        self.active_clients = 0
    
    def update_metrics(self, response):
        with self.lock:
            round_num = response.get('round', 0)
            if 'accuracy' in response:
                new_points = len(response.get('accuracy', []))
                if new_points > len(self.metrics['Round']):
                    for i in range(len(self.metrics['Round']), new_points):
                        self.metrics['Round'].append(i+1)
                        self.metrics['Accuracy'].append(float(response['accuracy'][i]))
                        self.metrics['FairFinance_Score'].append(float(response['fairness_score'][i]))
                        self.metrics['Fairness'].append(float(response['fairness'][i]))
                        self.metrics['Communication_Efficiency'].append(float(response['communication'][i]))
                        self.metrics['Robustness'].append(float(response['robustness'][i]))
                    print(f"[METRICS] Updated metrics to round {round_num}")
    
    def get_metrics_df(self):
        with self.lock:
            return pd.DataFrame(self.metrics).copy()

if '_METRICS_TRACKER_SINGLETON' not in globals():
    _METRICS_TRACKER_SINGLETON = MetricsTracker()
    globals()['_METRICS_TRACKER_SINGLETON'] = _METRICS_TRACKER_SINGLETON
METRICS_TRACKER = globals()['_METRICS_TRACKER_SINGLETON']

def fetch_metrics_once():
    """Do one immediate fetch to populate data on startup"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        sock.connect(('localhost', 8082))
        msg = {'type': 'get_metrics'}
        data = pickle.dumps(msg)
        sock.send(len(data).to_bytes(4, 'big'))
        sock.sendall(data)
        size_data = sock.recv(4)
        if size_data:
            size = int.from_bytes(size_data, 'big')
            response_data = b''
            while len(response_data) < size:
                chunk = sock.recv(min(size - len(response_data), 8192))
                if not chunk:
                    break
                response_data += chunk
            if response_data:
                response = pickle.loads(response_data)
                if response.get('type') == 'metrics':
                    METRICS_TRACKER.update_metrics(response)
                    print("[DASHBOARD] Initial data loaded")
        sock.close()
    except Exception as e:
        print(f"[DASHBOARD] Initial fetch error: {e}")

def update_metrics_thread():
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect(('localhost', 8082))
            print("[DASHBOARD] Connected to server")
            while True:
                try:
                    msg = {'type': 'get_metrics'}
                    data = pickle.dumps(msg)
                    sock.send(len(data).to_bytes(4, 'big'))
                    sock.sendall(data)
                    size_data = sock.recv(4)
                    if not size_data:
                        break
                    size = int.from_bytes(size_data, 'big')
                    response_data = b''
                    while len(response_data) < size:
                        chunk = sock.recv(min(size - len(response_data), 8192))
                        if not chunk:
                            break
                        response_data += chunk
                    if response_data:
                        response = pickle.loads(response_data)
                        if response.get('type') == 'metrics':
                            METRICS_TRACKER.update_metrics(response)
                    time.sleep(1)
                except Exception as e:
                    print(f"[DASHBOARD] Error: {e}")
                    break
            sock.close()
        except Exception as e:
            print(f"[DASHBOARD] Connection error: {e}")
            time.sleep(5)

if '_DASH_METRICS_THREAD_STARTED' not in globals():
    _DASH_METRICS_THREAD_STARTED = True
    globals()['_DASH_METRICS_THREAD_STARTED'] = True
    # Do immediate fetch first
    fetch_metrics_once()
    # Then start background polling thread
    thread = threading.Thread(target=update_metrics_thread, daemon=True)
    thread.start()

selected = option_menu(menu_title=None, options=["Dashboard", "Model Analysis", "Fairness Metrics"],
                       icons=["house", "graph-up", "shield-check"], orientation="horizontal")

if selected == "Dashboard":
    st.title("🤖 FairFinance Dashboard")
    st.markdown("### Federated Learning Evaluation Framework")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3, gap="large")
    df = METRICS_TRACKER.get_metrics_df()
    
    if not df.empty and len(df) > 0:
        with col1:
            st.metric(
                "📊 FairFinance Score",
                f"{df['FairFinance_Score'].iloc[-1]*100:.2f}%",
                delta=f"{(df['FairFinance_Score'].iloc[-1] - df['FairFinance_Score'].iloc[-2])*100 if len(df) > 1 else 0:.2f}%" if len(df) > 1 else None
            )
        with col2:
            st.metric(
                "🎯 Model Accuracy",
                f"{df['Accuracy'].iloc[-1]*100:.2f}%",
                delta=f"{(df['Accuracy'].iloc[-1] - df['Accuracy'].iloc[-2])*100 if len(df) > 1 else 0:.2f}%" if len(df) > 1 else None
            )
        with col3:
            st.metric(
                "⚖️ Fairness Score",
                f"{df['Fairness'].iloc[-1]*100:.2f}%",
                delta=f"{(df['Fairness'].iloc[-1] - df['Fairness'].iloc[-2])*100 if len(df) > 1 else 0:.2f}%" if len(df) > 1 else None
            )
    else:
        with col1:
            st.metric("📊 FairFinance Score", "Waiting...")
        with col2:
            st.metric("🎯 Model Accuracy", "Waiting...")
        with col3:
            st.metric("⚖️ Fairness Score", "Waiting...")

    st.markdown("### 📈 Training Progress")
    tab1, tab2 = st.tabs(["Performance Metrics", "Component Scores"])
    
    with tab1:
        df = METRICS_TRACKER.get_metrics_df()
        if not df.empty:
            df_plot = df.copy()
            df_plot['Accuracy'] = df_plot['Accuracy'] * 100
            df_plot['FairFinance_Score'] = df_plot['FairFinance_Score'] * 100
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_plot["Round"], 
                y=df_plot["Accuracy"], 
                name="Accuracy", 
                mode='lines+markers',
                line=dict(color="#3B82F6", width=4),
                marker=dict(size=8, color="#3B82F6", symbol="circle")
            ))
            fig.add_trace(go.Scatter(
                x=df_plot["Round"], 
                y=df_plot["FairFinance_Score"], 
                name="FairFinance Score", 
                mode='lines+markers',
                line=dict(color="#EF4444", width=4),
                marker=dict(size=8, color="#EF4444", symbol="diamond"),
                yaxis="y2"
            ))
            fig.update_layout(
                title="<b>Model Performance Over Time</b>",
                template="plotly_dark",
                plot_bgcolor="#1E2847",
                paper_bgcolor="#0E1117",
                font=dict(color="#FAFAFA", size=12),
                height=550,
                hovermode='x unified',
                yaxis=dict(
                    title="<b>Accuracy (%)</b>",
                    side="left",
                    range=[0, 100],
                    gridcolor="#3D4A5C"
                ),
                yaxis2=dict(
                    title="<b>FairFinance Score (%)</b>",
                    side="right",
                    overlaying="y",
                    range=[0, 100],
                    gridcolor="#3D4A5C"
                ),
                xaxis=dict(
                    title="<b>Training Round</b>",
                    gridcolor="#3D4A5C"
                ),
                legend=dict(
                    x=0.02,
                    y=0.98,
                    bgcolor="rgba(30, 40, 71, 0.8)",
                    bordercolor="#3B82F6",
                    borderwidth=2
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⏳ Waiting for training data...")
    
    with tab2:
        df = METRICS_TRACKER.get_metrics_df()
        if not df.empty:
            df_display = df.copy()
            for col in ['Fairness', 'Communication_Efficiency', 'Robustness']:
                df_display[col] = df_display[col] * 100
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_display["Round"],
                y=df_display["Fairness"],
                name="Fairness",
                mode='lines+markers',
                line=dict(color="#10B981", width=4),
                marker=dict(size=8, color="#10B981")
            ))
            fig.add_trace(go.Scatter(
                x=df_display["Round"],
                y=df_display["Communication_Efficiency"],
                name="Communication",
                mode='lines+markers',
                line=dict(color="#A78BFA", width=4),
                marker=dict(size=8, color="#A78BFA")
            ))
            fig.add_trace(go.Scatter(
                x=df_display["Round"],
                y=df_display["Robustness"],
                name="Robustness",
                mode='lines+markers',
                line=dict(color="#F59E0B", width=4),
                marker=dict(size=8, color="#F59E0B")
            ))
            fig.update_layout(
                title="<b>Component Scores Over Time</b>",
                template="plotly_dark",
                plot_bgcolor="#1E2847",
                paper_bgcolor="#0E1117",
                font=dict(color="#FAFAFA", size=12),
                height=550,
                hovermode='x unified',
                xaxis=dict(
                    title="<b>Training Round</b>",
                    gridcolor="#3D4A5C"
                ),
                yaxis=dict(
                    title="<b>Score (%)</b>",
                    range=[0, 100],
                    gridcolor="#3D4A5C"
                ),
                legend=dict(
                    x=0.02,
                    y=0.98,
                    bgcolor="rgba(30, 40, 71, 0.8)",
                    bordercolor="#3B82F6",
                    borderwidth=2
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⏳ Waiting for training data...")

elif selected == "Model Analysis":
    st.title("Model Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background-color: #1E3A8A; padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="color: #60A5FA; margin: 0; font-size: 14px;">Training Time</h3>
            <p style="color: #FAFAFA; margin: 10px 0 0 0; font-size: 24px; font-weight: bold;">15.3 minutes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #064E3B; padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="color: #6EE7B7; margin: 0; font-size: 14px;">Model Size</h3>
            <p style="color: #FAFAFA; margin: 10px 0 0 0; font-size: 24px; font-weight: bold;">2.4 MB</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background-color: #713F12; padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="color: #FCD34D; margin: 0; font-size: 14px;">Active Clients</h3>
            <p style="color: #FAFAFA; margin: 10px 0 0 0; font-size: 24px; font-weight: bold;">3/5 Connected</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Model Architecture")
    st.markdown("""
    ```
    FraudDetector(
        (fc1): Linear(in_features=30, out_features=64)
        (fc2): Linear(in_features=64, out_features=32)
        (fc3): Linear(in_features=32, out_features=1)
        (relu): ReLU()
        (sigmoid): Sigmoid()
    )
    ```
    """)

elif selected == "Fairness Metrics":
    st.title("Fairness Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Group Performance")
        group_data = pd.DataFrame({
            "Group": ["High Income", "Medium Income", "Low Income"], 
            "Accuracy": [0.89, 0.87, 0.85], 
            "False Positive Rate": [0.11, 0.13, 0.15]
        })
        st.dataframe(group_data, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### Fairness Components")
        df = METRICS_TRACKER.get_metrics_df()
        if not df.empty and len(df) > 0:
            # Display current scores as a table
            components_scores = {
                "Component": ["Fairness", "Communication", "Robustness"],
                "Score (%)": [
                    f"{df['Fairness'].iloc[-1]*100:.2f}",
                    f"{df['Communication_Efficiency'].iloc[-1]*100:.2f}",
                    f"{df['Robustness'].iloc[-1]*100:.2f}"
                ]
            }
            components_df = pd.DataFrame(components_scores)
            st.dataframe(components_df, use_container_width=True, hide_index=True)
        else:
            st.info("Waiting for fairness data...")
    
    st.markdown("---")
    st.markdown("### Fairness Score Breakdown")
    
    col_radar1, col_radar2 = st.columns([2, 1])
    
    with col_radar1:
        df = METRICS_TRACKER.get_metrics_df()
        if not df.empty and len(df) > 0:
            # Radar chart for fairness components
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=[
                    df['Accuracy'].iloc[-1]*100,
                    df['Fairness'].iloc[-1]*100,
                    df['Communication_Efficiency'].iloc[-1]*100,
                    df['Robustness'].iloc[-1]*100
                ],
                theta=['Accuracy', 'Fairness', 'Communication', 'Robustness'],
                fill='toself',
                fillcolor='rgba(99, 110, 250, 0.3)',
                line=dict(color='#636efa'),
                name='Current Score'
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)),
                    bgcolor="rgba(30, 32, 39, 0.5)",
                    angularaxis=dict(rotation=90)
                ),
                template="plotly_dark",
                paper_bgcolor="#0E1117",
                font=dict(color="#FAFAFA", size=12),
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info("Waiting for fairness data...")
    
    with col_radar2:
        st.markdown("#### Latest Insights")
        df = METRICS_TRACKER.get_metrics_df()
        if not df.empty and len(df) > 0:
            st.metric("Accuracy", f"{df['Accuracy'].iloc[-1]*100:.1f}%")
            st.metric("Current Fairness", f"{df['Fairness'].iloc[-1]*100:.1f}%")
            st.metric("Communication", f"{df['Communication_Efficiency'].iloc[-1]*100:.1f}%")
            st.metric("Robustness", f"{df['Robustness'].iloc[-1]*100:.1f}%")
