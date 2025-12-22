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
        """Execute the complete analysis workflow for brain tumor detection and analysis.

        Args:
            image_path (str): Path to the MRI image file to analyze.

        Returns:
            str: Complete analysis report including classification, clinical analysis,
                 recommendations, and final medical report.
        """

        # Create tasks with proper parameter passing
        classification_task = create_classification_task(image_path)

        # Execute classification first to get results
        crew_temp = Crew(
            agents=[classifier_agent],
            tasks=[classification_task],
            process=Process.sequential,
            verbose=True
        )
        classification_result = crew_temp.kickoff()

        # Create subsequent tasks with results
        clinical_task = create_clinical_analysis_task(str(classification_result))
        recommendations_task = create_recommendations_task(str(classification_result))
        report_task = create_report_task(
            str(classification_result),
            str(classification_result),  # Placeholder - should be actual clinical result
            str(classification_result)   # Placeholder - should be actual recommendations
        )

        # Create the full crew
        crew = Crew(
            agents=[
                clinical_analyst_agent,
                recommendations_agent,
                report_agent
            ],
            tasks=[clinical_task, recommendations_task, report_task],
            process=Process.sequential,
            verbose=True
        )

        # Execute the analysis
        result = crew.kickoff()

        return result