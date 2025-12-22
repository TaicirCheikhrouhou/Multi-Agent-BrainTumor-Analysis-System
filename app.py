import os
import streamlit as st
from datetime import datetime
from PIL import Image
import time
from dotenv import load_dotenv

os.environ["CREWAI_TELEMETRY"] = "false"
# Load environment variables
load_dotenv()

# Disable cloud services
os.environ["GEMINI_API_KEY"] = "------"

# CrewAI imports
from crew.agents import (
    classifier_agent,
    clinical_analyst_agent,
    recommendations_agent,
    report_agent
)
from crew.tasks import (
    create_classification_task,
    create_clinical_analysis_task,
    create_recommendations_task,
    create_report_task
)

# Page configuration
st.set_page_config(
    page_title="BrainTumorAISystem",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced medical professional styling
st.markdown("""
<style>
    .main {
        background-color: #f0f8f5;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .header {
        background: linear-gradient(135deg, #00695c 0%, #00897b 50%, #26a69a 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,105,92,0.3);
    }
    .agent-card {
        background: white;
        padding: 1.8rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #e0e0e0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .agent-running {
        border-left-color: #00897b;
        transform: translateX(5px);
        box-shadow: 0 8px 24px rgba(0,137,123,0.2);
        background: linear-gradient(to right, #ffffff 0%, #e8f5f0 100%);
    }
    .agent-completed {
        border-left-color: #2e7d32;
        background-color: #f9fcf9;
    }
    .agent-pending {
        opacity: 0.6;
    }
    .result-box {
        background: linear-gradient(135deg, #e8f5f0 0%, #f0f9f5 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #00897b;
        margin-top: 1rem;
        font-family: 'Courier New', monospace;
        white-space: pre-wrap;
        line-height: 1.6;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
    }
    .status-badge {
        padding: 0.5rem 1.2rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
        display: inline-block;
    }
    .running {
        background: linear-gradient(135deg, #e8f5f0, #b2dfdb);
        color: #00695c;
        animation: pulse 2s ease-in-out infinite;
    }
    .completed { 
        background: linear-gradient(135deg, #e8f5e8, #c8e6c9); 
        color: #2e7d32; 
    }
    .pending { 
        background: #f5f5f5; 
        color: #999; 
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    .progress-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .disclaimer {
        background: linear-gradient(135deg, #fff8e1 0%, #fff9e6 100%);
        border: 2px solid #ffb74d;
        padding: 1.5rem; 
        border-radius: 12px; 
        margin-top: 3rem;
        font-size: 0.95rem;
        box-shadow: 0 2px 8px rgba(255,183,77,0.2);
    }
    .image-container {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    .section-title {
        color: #003366;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #0055A4;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header">
    <h1 style="margin:0; font-size:4rem;">🧠 BrainScanAI 🧠</h1>
    <h2 style="margin:0.8rem 0; font-weight:300;">Multi-Agent Brain Tumor Analysis System</h2>
    <p style="margin:0; font-size:1.1rem; opacity:0.9;">Sequential AI Pipeline • VGG19 • Clinical Reasoning</p>
</div>
""", unsafe_allow_html=True)

# Session state initialization
def init_session_state():
    defaults = {
        'step': 'upload',
        'current_agent': 0,
        'results': {},
        'final_report': None,
        'image_path': None,
        'gradcam_path': None,
        'start_time': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()
import streamlit as st
from neo4j_connector import driver, get_treatments_for_tumor
from neo4j_visualizer import render_neo4j_graph
st.sidebar.title("🧠 Knowledge Graph")

if st.sidebar.button("Test Neo4j Connection"):
    try:
        with driver.session() as session:
            msg = session.run("RETURN 'Neo4j connected!' AS msg")
            st.sidebar.success(msg.single()["msg"])
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

if st.sidebar.button("Display Neo4j Graph"):
    try:
        graph_file = render_neo4j_graph()
        st.success("Graph generated!")

        # Display in Streamlit
        with open(graph_file, "r", encoding="utf-8") as f:
            html_code = f.read()
        st.components.v1.html(html_code, height=600)

    except Exception as e:
        st.error(f"Error: {e}")

# ===================== Agents =====================
AGENTS = [
    {
        "name": "Image Classification Agent",
        "task": "Tumor detection via VGG19",
        "id": "classification",
        "icon": "🔍",
        "agent_obj": classifier_agent,
        "task_creator": lambda: create_classification_task(st.session_state.image_path)
    },
    {
        "name": "Clinical Knowledge Agent",
        "task": "Medical data synthesis",
        "id": "clinique",
        "icon": "📊",
        "agent_obj": clinical_analyst_agent,
        "task_creator": lambda: create_clinical_analysis_task(
            st.session_state.results.get("classification"))
    },
    {
        "name": "Recommendations Agent",
        "task": "Therapeutic proposal and follow-up",
        "id": "recommandations",
        "icon": "💊",
        "agent_obj": recommendations_agent,
        "task_creator": lambda: create_recommendations_task(
            st.session_state.results.get("clinique", "")
        )
    },
    {
        "name": "Report Writing Agent",
        "task": "Generation of complete medical report",
        "id": "rapport",
        "icon": "📋",
        "agent_obj": report_agent,
        "task_creator": lambda: create_report_task(
            classification_result=st.session_state.results.get("classification", ""),
            clinical_result=st.session_state.results.get("clinique", ""),
            recommendations_result=st.session_state.results.get("recommandations", "")
        )
    }
]

def clean_result_text(result_str, agent_id):
    """Clean and format agent result"""
    if agent_id == "classification":
        keywords = ["Diagnosis", "Confidence", "Type"]
    elif agent_id == "clinique":
        keywords = ["Type", "Grade", "Prognosis", "Characteristics"]
    elif agent_id == "recommandations":
        keywords = ["Urgency", "Treatment", "Follow-up", "Next"]
    else:
        return result_str
    
    lines = [l for l in result_str.split("\n") if any(k in l for k in keywords)]
    return "\n".join(lines) if lines else result_str

def execute_agent(agent_config):
    """Execute an agent and return the result"""
    try:
        task = agent_config['task_creator']()
        result = agent_config['agent_obj'].execute_task(task=task)
        result_str = str(result)
        
        print(f"\n{'='*60}")
        print(f"AGENT: {agent_config['name']}")
        print(f"RAW RESULT:\n{result_str}")
        print(f"{'='*60}\n")
        
        # Clean result
        clean_text = clean_result_text(result_str, agent_config['id'])
        return clean_text,None
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        return None, error_detail
# === MAIN INTERFACE ===
col1, col2 = st.columns([1, 2.5])

with col1:
    st.markdown('<p class="section-title">📤 MRI Upload</p>', unsafe_allow_html=True)
    
    fichier = st.file_uploader(
        "Select a brain MRI (PNG/JPG/JPEG)",
        type=['png', 'jpg', 'jpeg'],
        disabled=st.session_state.step == "running",
        help="Accepted formats: PNG, JPG, JPEG"
    )

    if fichier:
        # Create temporary folder
        os.makedirs("temp", exist_ok=True)
        chemin = f"temp/{fichier.name}"
        
        # Check if new image
        if st.session_state.image_path != chemin:
            Image.open(fichier).save(chemin)
            st.session_state.image_path = chemin
            # Complete reset
            st.session_state.step = "upload"
            st.session_state.current_agent = 0
            st.session_state.results = {}
            st.session_state.final_report = None
            st.session_state.gradcam_path = None

        # Display image
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(st.session_state.image_path, caption="Loaded Brain MRI", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        
        # Launch button
        if st.session_state.step == "upload":
            st.markdown("---")
            if not os.path.exists("models/best_model_VGG19.keras"):
                st.error("⚠️ VGG19 model missing in /models/")
            else:
                if st.button("🚀 Start Multi-Agent Analysis", type="primary", use_container_width=True):
                    st.session_state.step = "running"
                    st.session_state.start_time = time.time()
                    st.rerun()

with col2:
    st.markdown('<p class="section-title">⚙️ Analysis Progress</p>', unsafe_allow_html=True)

    if st.session_state.step == "running":
        # Progress bar
        progress = st.session_state.current_agent / len(AGENTS)
        st.markdown('<div class="progress-container">', unsafe_allow_html=True)
        st.progress(progress)
        
        col_prog1, col_prog2 = st.columns(2)
        with col_prog1:
            st.metric("Step", f"{st.session_state.current_agent + 1} / {len(AGENTS)}")
        with col_prog2:
            if st.session_state.start_time:
                elapsed = int(time.time() - st.session_state.start_time)
                st.metric("Elapsed Time", f"{elapsed}s")
        st.markdown('</div>', unsafe_allow_html=True)

        current_idx = st.session_state.current_agent

        # Display all agents
        for i, agent in enumerate(AGENTS):
            # Determine status
            if i < current_idx:
                statut = "completed"
                badge_text = "✓ COMPLETED"
                badge_class = "completed"
            elif i == current_idx:
                statut = "running"
                badge_text = "⚡ RUNNING"
                badge_class = "running"
            else:
                statut = "pending"
                badge_text = "⏳ PENDING"
                badge_class = "pending"

            # Carte de l'agent
            st.markdown(f"""
            <div class="agent-card agent-{statut}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h4 style="margin:0; color:#003366;">
                            {agent['icon']} {agent['name']}
                        </h4>
                        <p style="margin:5px 0; color:#666; font-size:0.9rem;">{agent['task']}</p>
                    </div>
                    <span class="status-badge {badge_class}">
                        {badge_text}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Display result if agent has completed
            if i < current_idx and agent['id'] in st.session_state.results:
                res = st.session_state.results[agent['id']]
                st.markdown(
                    f"<div class='result-box'><strong>📄 Result:</strong><br><br>{res}</div>",
                    unsafe_allow_html=True
                )

            # Execute current agent
            if i == current_idx:
                with st.spinner(f"⚙️ {agent['name']} executing..."):
                    result_text, error = execute_agent(agent)
                    
                    if error:
                        st.error(f"❌ Error with {agent['name']}: {error}")
                        if st.button("🔄 Reset", key=f"reset_{i}"):
                            st.session_state.step = "upload"
                            st.session_state.current_agent = 0
                            st.rerun()
                        st.stop()
                    else:
                        # Save result
                        st.session_state.results[agent['id']] = result_text
                        from neo4j_connector import driver, get_treatments_for_tumor

                        # If the finished agent is the classification agent
                        if agent['id'] == "classification":
                            tumor_type = result_text  # or extract tumor type from result_text
                            if tumor_type:
                                with driver.session() as session:
                                    traitements = session.execute_read(get_treatments_for_tumor, tumor_type)
                                st.session_state.results["neo4j_treatments"] = "\n".join(traitements)

                        # If it's the last agent, save the final report
                        if i == len(AGENTS) - 1:
                            st.session_state.final_report = result_text
                            st.session_state.step = "completed"
                        
                        # Move to next agent
                        st.session_state.current_agent += 1
                        time.sleep(0.5)  # Small pause for visual effect
                        st.rerun()

    elif st.session_state.step == "completed":
        st.success("✅ Analysis completed successfully!")
        st.balloons()
        
        # Total time
        if st.session_state.start_time:
            total_time = int(time.time() - st.session_state.start_time)
            st.info(f"⏱️ Total analysis time: {total_time} seconds")

        # Display results by agent
        st.markdown("---")
        st.markdown('<p class="section-title">📊 Detailed Results by Agent</p>', unsafe_allow_html=True)
        
        for agent in AGENTS:
            if agent['id'] in st.session_state.results:
                with st.expander(f"{agent['icon']} {agent['name']}", expanded=False):
                    st.markdown(
                        f"<div class='result-box'>{st.session_state.results[agent['id']]}</div>",
                        unsafe_allow_html=True
                    )
        if "neo4j_treatments" in st.session_state.results:
            st.markdown(
                f"<div class='result-box'><strong>Recommended Treatments (Neo4j):</strong><br>{st.session_state.results['neo4j_treatments']}</div>",
                unsafe_allow_html=True
            )

        # Final report
        st.markdown("---")
        st.markdown('<p class="section-title">📋 Complete Medical Report</p>', unsafe_allow_html=True)
        st.markdown(
            f"<div class='result-box' style='font-size:1.05rem;'>{st.session_state.final_report}</div>",
            unsafe_allow_html=True
        )

        # Action buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.download_button(
                label="📥 Download Complete Report",
                data=st.session_state.final_report,
                file_name=f"brain_tumor_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                type="primary",
                use_container_width=True
            )
        with col_btn2:
            if st.button("🔄 New Analysis", use_container_width=True):
                st.session_state.clear()
                st.rerun()

    else:
        st.info("👈 Upload a brain MRI in the left column to start the analysis.")
