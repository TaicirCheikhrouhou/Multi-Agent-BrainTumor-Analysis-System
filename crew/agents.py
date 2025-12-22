from crewai import Agent
from config import llm

# AGENTS

classifier_agent = Agent(
    role="AI Radiology Specialist",
    goal="Classify brain MRI images to detect tumors",
    backstory="Expert in medical imaging with deep learning specialization",
    llm=llm,
    verbose=True,
    allow_delegation=False
)

clinical_analyst_agent = Agent(
    role="Clinical Analyst",
    goal="Analyze results with medical context and determine tumor characteristics",
    backstory="Experienced clinician with oncology expertise",
    llm=llm,
    verbose=True,
    allow_delegation=False
)

recommendations_agent = Agent(
    role="Treatment Recommendations Specialist",
    goal="Provide evidence-based clinical recommendations for patient care",
    backstory="Board-certified oncologist with treatment protocol expertise",
    llm=llm,
    verbose=True,
    allow_delegation=False
)

report_agent = Agent(
    role="Medical Report Writer",
    goal="Generate structured and comprehensive medical reports",
    backstory="Medical documentation expert with clinical writing specialization",
    llm=llm,
    verbose=True,
    allow_delegation=False
)
