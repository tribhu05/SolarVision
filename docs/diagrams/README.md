# SolarVision Architecture & Workflow Diagrams

This directory contains the formal architectural and engineering diagrams for the **SolarVision** autonomous solar computer vision and space weather demonstration system. All diagrams are authored in GitHub-compliant Mermaid and fully documented with mathematical foundations and structural specifications.

## Diagram Index
1. [**System Architecture Diagram**](1_system_architecture.md): Full end-to-end multi-layer architecture from external NASA/NOAA feeds down to the Streamlit UI.
2. [**Data Processing Workflow Diagram**](2_data_processing_workflow.md): Detailed sequential pipeline flow matching:
   $$\text{Data Source} \to \text{Ingestion} \to \text{Preprocessing} \to \text{Segmentation} \to \text{Feature Extraction} \to \text{Classification} \to \text{Tracking} \to \text{Database} \to \text{Dashboard}$$
3. [**Module Dependency Diagram**](3_module_dependencies.md): Low-coupling structural breakdown of all Python modules under `src/`, configuration, and application layers.
4. [**Database Entity-Relationship (ER) Diagram**](4_database_er.md): Complete relational schema of `solarvision.db` detailing the 5 tables, foreign keys, cascades, and constraints.
5. [**Deployment Architecture Diagram**](5_deployment_architecture.md): Topology for local workstation execution and zero-secret containerized Streamlit Community Cloud hosting.
