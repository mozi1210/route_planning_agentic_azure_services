# route_planning_agentic_azure_services

This project focuses to help schedule freight trains route with the focus of lowering the pendency of containers which needs to be moved from location a to b.
This has been developed as a demo only and only with certain constraints such
1. Every train must go offline for 5 days after every 30 days of operation
2. Each origin-destination pair supports either Double Stack or Single Stack container loading — not both
3. Approximately 12 hours per station for unloading + loading of the containers
4. 20ft and 40ft containers, which affect stacking and capacity calculations

The solution developed is 2 agents solution (Agentic) where Google ortool's CP-SAT is the primary optimizer. 
