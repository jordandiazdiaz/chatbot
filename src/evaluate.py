"""
Evaluation Module for Suspicious Account Activity and Billing Chatbot.

This module evaluates the chatbot's performance on various metrics,
including accuracy, relevance, and response time.
"""

import os
import sys
import logging
import json
import time
import csv
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.chatbot import SupportChatbot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ChatbotEvaluator:
    """
    Evaluates chatbot performance using predefined test cases and metrics.
    
    This class measures the chatbot's accuracy, relevance, response time,
    and other key performance indicators.
    """
    
    def __init__(
        self,
        chatbot: SupportChatbot,
        test_cases_path: str = "tests/test_cases.json",
        results_dir: str = "evaluation_results"
    ):
        """
        Initialize the ChatbotEvaluator.
        
        Args:
            chatbot: Initialized SupportChatbot instance
            test_cases_path: Path to JSON file with test cases
            results_dir: Directory to save evaluation results
        """
        self.chatbot = chatbot
        self.test_cases_path = test_cases_path
        self.results_dir = results_dir
        
        # Create results directory if it doesn't exist
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        # Load test cases
        self.test_cases = self._load_test_cases()
        
    def _load_test_cases(self) -> List[Dict[str, Any]]:
        """
        Load test cases from JSON file.
        
        Returns:
            List[Dict[str, Any]]: List of test case dictionaries
        """
        if not os.path.exists(self.test_cases_path):
            logger.warning(f"Test cases file not found at {self.test_cases_path}")
            return []
            
        try:
            with open(self.test_cases_path, 'r') as f:
                test_cases = json.load(f)
                logger.info(f"Loaded {len(test_cases)} test cases")
                return test_cases
        except Exception as e:
            logger.error(f"Error loading test cases: {str(e)}", exc_info=True)
            return []
            
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run evaluation on all test cases.
        
        Returns:
            Dict[str, Any]: Evaluation results and metrics
        """
        if not self.test_cases:
            logger.error("No test cases available for evaluation")
            return {
                "success": False,
                "error": "No test cases available"
            }
            
        logger.info("Starting evaluation")
        
        results = []
        response_times = []
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        # Process each test case
        for idx, test_case in enumerate(self.test_cases):
            logger.info(f"Processing test case {idx+1}/{len(self.test_cases)}")
            
            # Extract test case data
            query = test_case.get("query", "")
            expected_in_scope = test_case.get("in_scope", True)
            expected_keywords = test_case.get("expected_keywords", [])
            
            # Reset conversation before each test
            self.chatbot.reset_conversation()
            
            # Measure response time
            start_time = time.time()
            response = self.chatbot.get_answer(query)
            end_time = time.time()
            response_time = end_time - start_time
            response_times.append(response_time)
            
            # Check if the response is in scope
            actual_in_scope = not response.get("is_fallback", False)
            
            # Check for expected keywords in response
            answer_text = response.get("answer", "").lower()
            keywords_found = [
                keyword for keyword in expected_keywords 
                if keyword.lower() in answer_text
            ]
            keywords_match_ratio = len(keywords_found) / len(expected_keywords) if expected_keywords else 1.0
            
            # Determine if response is correct (true positive, true negative, etc.)
            if expected_in_scope and actual_in_scope:
                # True positive: Should be in scope and is in scope
                true_positives += 1
                correct = True
            elif not expected_in_scope and not actual_in_scope:
                # True negative: Should be out of scope and is out of scope
                true_negatives += 1
                correct = True
            elif expected_in_scope and not actual_in_scope:
                # False negative: Should be in scope but is out of scope
                false_negatives += 1
                correct = False
            else:  # not expected_in_scope and actual_in_scope
                # False positive: Should be out of scope but is in scope
                false_positives += 1
                correct = False
            
            # Save result
            result = {
                "query": query,
                "expected_in_scope": expected_in_scope,
                "actual_in_scope": actual_in_scope,
                "expected_keywords": expected_keywords,
                "keywords_found": keywords_found,
                "keywords_match_ratio": keywords_match_ratio,
                "response_time": response_time,
                "correct": correct,
                "answer": response.get("answer", ""),
                "sources": response.get("sources", []),
                "token_usage": response.get("token_usage", {})
            }
            results.append(result)
            
        # Calculate metrics
        accuracy = (true_positives + true_negatives) / len(self.test_cases)
        if true_positives + false_positives > 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0
            
        if true_positives + false_negatives > 0:
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = 0
            
        if precision + recall > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0
            
        avg_response_time = sum(response_times) / len(response_times)
        avg_keywords_match = sum(r["keywords_match_ratio"] for r in results) / len(results)
        
        # Create confusion matrix
        y_true = [1 if case.get("in_scope", True) else 0 for case in self.test_cases]
        y_pred = [1 if not result["actual_in_scope"] else 0 for result in results]
        cm = confusion_matrix(y_true, y_pred)
        
        # Save detailed results to CSV
        self._save_results_to_csv(results)
        
        # Generate evaluation report
        report = {
            "success": True,
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "avg_response_time": avg_response_time,
                "avg_keywords_match_ratio": avg_keywords_match,
                "confusion_matrix": cm.tolist() if hasattr(cm, "tolist") else cm,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "true_negatives": true_negatives,
                "false_negatives": false_negatives
            },
            "results": results
        }
        
        # Save report as JSON
        self._save_report(report)
        
        # Generate plots
        self._generate_plots(report)
        
        logger.info(f"Evaluation completed with accuracy: {accuracy:.2f}, F1-score: {f1_score:.2f}")
        
        return report
    
    def _save_results_to_csv(self, results: List[Dict[str, Any]]):
        """
        Save detailed test results to CSV.
        
        Args:
            results: List of test result dictionaries
        """
        csv_path = os.path.join(self.results_dir, "detailed_results.csv")
        
        try:
            with open(csv_path, 'w', newline='') as f:
                fieldnames = [
                    "query", "expected_in_scope", "actual_in_scope", 
                    "keywords_match_ratio", "response_time", "correct"
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                writer.writeheader()
                for result in results:
                    # Extract only the fields we want to save
                    row = {field: result[field] for field in fieldnames}
                    writer.writerow(row)
                    
            logger.info(f"Detailed results saved to {csv_path}")
        except Exception as e:
            logger.error(f"Error saving results to CSV: {str(e)}", exc_info=True)
    
    def _save_report(self, report: Dict[str, Any]):
        """
        Save evaluation report as JSON.
        
        Args:
            report: Evaluation report dictionary
        """
        report_path = os.path.join(self.results_dir, "evaluation_report.json")
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info(f"Evaluation report saved to {report_path}")
        except Exception as e:
            logger.error(f"Error saving evaluation report: {str(e)}", exc_info=True)
    
    def _generate_plots(self, report: Dict[str, Any]):
        """
        Generate evaluation plots and save them.
        
        Args:
            report: Evaluation report dictionary
        """
        metrics = report["metrics"]
        results = report["results"]
        
        try:
            # Plot 1: Confusion Matrix
            plt.figure(figsize=(8, 6))
            cm = metrics["confusion_matrix"]
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            
            classes = ['Out of Scope', 'In Scope']
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)
            
            # Add text annotations
            thresh = cm[0][0] + cm[1][1] / 2
            for i in range(len(classes)):
                for j in range(len(classes)):
                    plt.text(j, i, format(cm[i][j], 'd'),
                            horizontalalignment="center",
                            color="white" if cm[i][j] > thresh else "black")
            
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            # Save plot
            confusion_matrix_path = os.path.join(self.results_dir, "confusion_matrix.png")
            plt.savefig(confusion_matrix_path)
            plt.close()
            
            # Plot 2: Response Times
            plt.figure(figsize=(10, 6))
            response_times = [result["response_time"] for result in results]
            plt.hist(response_times, bins=10, alpha=0.7)
            plt.axvline(metrics["avg_response_time"], color='r', linestyle='dashed', linewidth=2)
            plt.title('Response Time Distribution')
            plt.xlabel('Response Time (seconds)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.annotate(f'Average: {metrics["avg_response_time"]:.2f}s', 
                        xy=(metrics["avg_response_time"], 0), 
                        xytext=(metrics["avg_response_time"] + 0.5, 5),
                        arrowprops=dict(facecolor='black', shrink=0.05))
            
            # Save plot
            response_time_path = os.path.join(self.results_dir, "response_times.png")
            plt.savefig(response_time_path)
            plt.close()
            
            # Plot 3: Metrics Comparison
            plt.figure(figsize=(8, 6))
            metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            metrics_values = [
                metrics["accuracy"], 
                metrics["precision"], 
                metrics["recall"], 
                metrics["f1_score"]
            ]
            
            # Create bar chart
            bars = plt.bar(metrics_names, metrics_values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
            plt.title('Performance Metrics')
            plt.ylim(0, 1.0)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.2f}', ha='center', va='bottom')
            
            # Save plot
            metrics_path = os.path.join(self.results_dir, "performance_metrics.png")
            plt.savefig(metrics_path)
            plt.close()
            
            logger.info("Evaluation plots generated successfully")
        except Exception as e:
            logger.error(f"Error generating evaluation plots: {str(e)}", exc_info=True)


# Example test cases in JSON format
EXAMPLE_TEST_CASES = [
    {
        "query": "What should I do if I see suspicious activity on my account?",
        "in_scope": True,
        "expected_keywords": ["suspicious", "activity", "report", "contact"]
    },
    {
        "query": "How do I report an unauthorized charge?",
        "in_scope": True,
        "expected_keywords": ["unauthorized", "charge", "billing", "report"]
    },
    {
        "query": "What happens if someone logs into my account from a different country?",
        "in_scope": True,
        "expected_keywords": ["login", "different", "country", "location"]
    },
    {
        "query": "Tell me about the latest movies on your platform",
        "in_scope": False,
        "expected_keywords": []
    },
    {
        "query": "How do I set up a new device?",
        "in_scope": False,
        "expected_keywords": []
    }
]


if __name__ == "__main__":
    # Create test cases directory if it doesn't exist
    if not os.path.exists("tests"):
        os.makedirs("tests")
        
    # Create sample test cases if they don't exist
    test_cases_path = "tests/test_cases.json"
    if not os.path.exists(test_cases_path):
        with open(test_cases_path, 'w') as f:
            json.dump(EXAMPLE_TEST_CASES, f, indent=2)
        logger.info(f"Created sample test cases at {test_cases_path}")
    
    # Initialize chatbot
    chatbot = SupportChatbot()
    
    # Initialize evaluator
    evaluator = ChatbotEvaluator(chatbot, test_cases_path)
    
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