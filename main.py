#!/usr/bin/env python3
"""
Main script for Suspicious Account Activity and Billing Support Chatbot.

This script provides a CLI for running the different components of the chatbot system,
including data ingestion, evaluation, and launching the web interface.
"""

import os
import sys
import logging
import argparse
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def ingest_data(args):
    """Ingest PDF documents and create vector store."""
    from src.ingest import DocumentIngester
    
    logger.info("Starting document ingestion")
    
    # Initialize document ingester
    ingester = DocumentIngester(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Process PDFs
    ingester.process_pdf_directory(
        pdf_directory=args.pdf_dir,
        images_dir=args.images_dir,
        vector_store_dir=args.vector_store_dir
    )
    
    logger.info("Document ingestion completed")

def run_evaluation(args):
    """Run evaluation on the chatbot."""
    from src.chatbot import SupportChatbot
    from src.evaluate import ChatbotEvaluator
    
    logger.info("Starting chatbot evaluation")
    
    # Initialize chatbot
    chatbot = SupportChatbot(
        vector_store_path=args.vector_store_dir
    )
    
    # Initialize evaluator
    evaluator = ChatbotEvaluator(
        chatbot=chatbot,
        test_cases_path=args.test_cases,
        results_dir=args.results_dir
    )
    
    # Run evaluation
    report = evaluator.run_evaluation()
    
    # Print summary
    if report["success"]:
        metrics = report["metrics"]
        print("\nEvaluation Results Summary:")
        print(f"Accuracy: {metrics['accuracy']:.2f}")
        print(f"Precision: {metrics['precision']:.2f}")
        print(f"Recall: {metrics['recall']:.2f}")
        print(f"F1 Score: {metrics['f1_score']:.2f}")
        print(f"Average Response Time: {metrics['avg_response_time']:.2f} seconds")
        print(f"Results saved to {evaluator.results_dir}/")
    else:
        print(f"Evaluation failed: {report.get('error', 'Unknown error')}")

def run_web_app(args):
    """Run the Flask web application."""
    import subprocess
    
    logger.info("Starting web application")
    
    # Run Flask app
    cmd = [
        sys.executable, 
        "app/app.py"
    ]
    
    # Add arguments
    if args.host:
        cmd.extend(["--host", args.host])
    if args.port:
        cmd.extend(["--port", str(args.port)])
    if args.debug:
        cmd.append("--debug")
    
    # Execute
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        logger.info("Web application stopped")

def main():
    """Parse command line arguments and execute commands."""
    parser = argparse.ArgumentParser(
        description="Suspicious Account Activity and Billing Support Chatbot CLI"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest PDF documents")
    ingest_parser.add_argument("--pdf-dir", default="data", help="Directory containing PDF files")
    ingest_parser.add_argument("--images-dir", default="static/images", help="Directory to save extracted images")
    ingest_parser.add_argument("--vector-store-dir", default="models/vector_store", help="Directory to save vector store")
    ingest_parser.add_argument("--chunk-size", type=int, default=500, help="Text chunk size")
    ingest_parser.add_argument("--chunk-overlap", type=int, default=50, help="Text chunk overlap")
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate chatbot performance")
    evaluate_parser.add_argument("--vector-store-dir", default="models/vector_store", help="Directory containing vector store")
    evaluate_parser.add_argument("--test-cases", default="tests/test_cases.json", help="Path to test cases JSON file")
    evaluate_parser.add_argument("--results-dir", default="evaluation_results", help="Directory to save evaluation results")
    
    # Web app command
    webapp_parser = subparsers.add_parser("webapp", help="Run web application")
    webapp_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    webapp_parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    webapp_parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == "ingest":
        ingest_data(args)
    elif args.command == "evaluate":
        run_evaluation(args)
    elif args.command == "webapp":
        run_web_app(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()