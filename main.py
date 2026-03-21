# Azure Blob Storage
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=studentcentralstore;AccountKey=...;EndpointSuffix=core.windows.net
AZURE_STORAGE_CONTAINER_NAME=course-pdfs

# Azure Cosmos DB
AZURE_COSMOS_ENDPOINT=https://student-central-db.documents.azure.com:443/
AZURE_COSMOS_KEY=your-primary-key-here
AZURE_COSMOS_DATABASE=student-central
AZURE_COSMOS_CONTAINER=courses

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://student-central-aoai.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-08-01-preview

# CORS — comma-separated list of allowed frontend origins
ALLOWED_ORIGINS=https://student-central.vercel.app,http://localhost:3000
