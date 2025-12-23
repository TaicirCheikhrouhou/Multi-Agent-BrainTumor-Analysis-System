from crewai import Crew, Process
from crew.tasks import (
    create_classification_task,
    create_clinical_analysis_task,
    create_recommendations_task,
    create_report_task
)
from crew.agents import (
    classifier_agent,
    clinical_analyst_agent,
    recommendations_agent,
    report_agent
)


class BrainTumorAnalysisCrew:
    """CrewAI-based system for comprehensive brain tumor MRI analysis."""

    def __init__(self):
        """Initialize the analysis crew with empty results storage."""
        self.results = {}

    def analyze(self, image_path: str):
        """
        Execute the complete analysis workflow for brain tumor detection and analysis.

        Args:
            image_path (str): Path to the MRI image file to analyze.

        Returns:
            str: Complete analysis report including classification, clinical analysis,
                 recommendations, and final medical report.
        """

        # 1️⃣ Classification task
        classification_task = create_classification_task(image_path)
        classification_crew = Crew(
            agents=[classifier_agent],
            tasks=[classification_task],
            process=Process.sequential,
            verbose=True
        )
        classification_result = classification_crew.kickoff()

        # 2️⃣ Create subsequent tasks (all using classification result)
        clinical_task = create_clinical_analysis_task(str(classification_result))
        recommendations_task = create_recommendations_task(str(classification_result))
        report_task = create_report_task(
            str(classification_result),
            str(classification_result),  # Placeholder - could be clinical result later
            str(classification_result)   # Placeholder - could be recommendations later
        )

        # 3️⃣ Full crew for remaining agents
        full_crew = Crew(
            agents=[clinical_analyst_agent, recommendations_agent, report_agent],
            tasks=[clinical_task, recommendations_task, report_task],
            process=Process.sequential,
            verbose=True
        )
        result = full_crew.kickoff()

        return result
