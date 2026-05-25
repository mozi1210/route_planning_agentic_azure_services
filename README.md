# route_planning_agentic_azure_services

This project focuses to help schedule freight trains route with the focus of lowering the pendency of containers which needs to be moved from location a to b.
This has been developed as a demo only and only with certain constraints such
1. Every train must go offline for 5 days after every 30 days of operation
2. Each origin-destination pair supports either Double Stack or Single Stack container loading — not both
3. Approximately 12 hours per station for unloading + loading of the containers
4. 20ft and 40ft containers, which affect stacking and capacity calculations

The solution developed is 2 agents solution (Agentic) where Google ortool's CP-SAT is the primary optimizer. 
Along with the python files, you can create environment.env where you can store all your keys and endpoint in following format

AZURE_OPENAI_CHAT_API_KEY=your_secret_key_here
AZURE_OPENAI_CHAT_ENDPOINT=https://your-private-endpoint.com
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDING_API_KEY=your_secret_key_here
AZURE_OPENAI_EMBEDDING_ENDPOINT=https://your-private-endpoint.com
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

AZURE_SEARCH_ENDPOINT=https://your-private-endpoint.com
AZURE_SEARCH_KEY=your_secret_key_here
AZURE_SEARCH_INDEX=your-search-index-name-here

# Azure Blob Storage settings
AZURE_BLOB_CONNECTION_STRING=https://your-private-connectionstring-here.com
AZURE_BLOB_CONTAINER=your-container-name-here

BLOB_CONT_PEND_FILE=Container_Pendency_Queue.xlsx
BLOB_MAINT_LOG_FILE=Maintenance_Logs.xlsx
BLOB_ROUTES_FILES=Routes.xlsx
BLOB_STN_LDUNLD_TIME_FILE=Stations_Load_Unload_Time.xlsx
BLOB_TRN_SCH_TRIPS_FILE=Train_Schedule_Trips.xlsx
BLOB_TRN_DATA_FILE=Trains_Data.xlsx
