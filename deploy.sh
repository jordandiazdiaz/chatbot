#!/bin/bash

# Script for deploying the chatbot to Google Cloud Run
# Make sure to have gcloud CLI installed and authenticated

# Configuration
PROJECT_ID="billing-chatbot-konecta"
SERVICE_NAME="billing-support-chatbot"
REGION="us-central1"
IMAGE="gcr.io/$PROJECT_ID/$SERVICE_NAME"

# Colors for pretty output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting deployment of $SERVICE_NAME to Google Cloud Run...${NC}\n"

# Build the Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build -t $IMAGE .

# Push to Google Container Registry
echo -e "\n${YELLOW}Pushing image to Google Container Registry...${NC}"
docker push $IMAGE

# Deploy to Cloud Run
echo -e "\n${YELLOW}Deploying to Cloud Run...${NC}"
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 10 \
  --set-env-vars="DEBUG=False"

# Get the URL of the deployed service
URL=$(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format='value(status.url)')

echo -e "\n${GREEN}Deployment completed successfully!${NC}"
echo -e "Your chatbot is now available at: ${GREEN}$URL${NC}"