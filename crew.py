import os
import time
import yaml
from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from PDFReaderTool import PDFReaderTool
from custom_llm import PortkeyLLM

# Load Portkey configuration
portkey_config_path = 'config/portkey.yaml'
with open(portkey_config_path, 'r') as f:
    portkey_config = yaml.safe_load(f)

# Initialize LLM with Portkey
llm = PortkeyLLM(
    model=portkey_config['portkey']['model']['name'],
    config=portkey_config,
    config_path=portkey_config_path
)


class PDFReaderToolWrapper:
    """
    Wraps the PDFReaderTool so CrewAI sees it as a valid tool object
    with name, description, and func attributes.
    """
    def __init__(self, pdf_path: str):
        self.name = "PDF Reader Tool"
        self.description = "A tool to read and extract text from PDF files."
        # Assign the PDFReaderTool's .run method to the func attribute
        self.func = PDFReaderTool(pdf_path).run


@CrewBase
class RiskAssesCrew:
    """RiskAsses crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def risk_agent(self) -> Agent:
        pdf_reader_tool = PDFReaderToolWrapper("./creditReport.pdf")
        return Agent(
            config=self.agents_config['risk_agent'],
            tools=[pdf_reader_tool],  # Provide the wrapper instance
            llm=llm,
            verbose=True
        )

    @agent
    def loan_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['loan_agent'],
            llm=llm,
            verbose=True
        )

    @task
    def research_credit_report_task(self) -> Task:
        """
        This task uses the risk_agent to read the PDF and analyze credit report.
        The output of the agent is stored into a timestamped text file.
        """
        def store_output_callback(result):
            # Store the result in a text file with a timestamped name
            timestamp = time.strftime('%Y%m%d%H%M%S')
            filename = f"creditreport_crewai_{timestamp}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(str(result))
            return result

        return Task(
            config=self.tasks_config['research_credit_report_task'],
            agent=self.risk_agent(),
            callback=store_output_callback
        )

    @task
    def loan_assesment_task(self) -> Task:
        return Task(
            config=self.tasks_config['loan_assesment_task'],
            agent=self.loan_agent()
        )

    @crew
    def crew(self) -> Crew:
        """Creates the RiskAssesCrew"""
        return Crew(
            agents=[self.risk_agent(), self.loan_agent()],
            tasks=[self.research_credit_report_task(), self.loan_assesment_task()],
            process=Process.sequential,
            verbose=True,
        )