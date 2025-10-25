from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
import google.generativeai as genai
from dotenv import load_dotenv
import os
import subprocess
import tempfile
import sys
import json
import ast
import yaml
import logging
from datetime import datetime
import argparse

load_dotenv()

with open("prompts.yaml", "r") as f:
    PROMPTS = yaml.safe_load(f)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel(
    PROMPTS['model_config']['default_model'],
    generation_config=genai.GenerationConfig(
        max_output_tokens=PROMPTS['model_config']['max_output_tokens'],
        temperature=PROMPTS['model_config']['default_temperature'],
    )
)

# Global verbosity setting
VERBOSITY = 1


def setup_logging(verbosity):
    """Configure logging based on verbosity level.
    
    Args:
        verbosity (int): 0=silent, 1=low, 2=medium, 3=full
    """
    global VERBOSITY
    VERBOSITY = verbosity
    
    level_map = {
        0: logging.CRITICAL,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG
    }
    
    log_level = level_map.get(verbosity, logging.INFO)
    
    handlers = []
    
    handlers.append(
        logging.FileHandler(f'semantic_diff_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    )
    
    if verbosity > 0:
        handlers.append(logging.StreamHandler(sys.stdout))
    
    logging.basicConfig(
        level=logging.DEBUG,  
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True
    )
    
    if verbosity > 0:
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(log_level)
    
    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)


class State(TypedDict):
    code1: str
    code2: str
    confidence: float
    differences: List[Dict[str, Any]]
    analysis: str
    test_cases: List[Dict[str, Any]]
    iteration: int
    max_iterations: int
    coverage_gaps: List[str]


def execute_code_with_input(code, test_input, timeout=5):
    """Execute code with specific input in isolated subprocess and return output."""
    logger.debug(f"Executing code with input: {test_input}")
    
    wrapped_code = f"""
import json
import sys

{code}

import ast
tree = ast.parse('''{code}''')
func_name = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)][0]

test_input = {repr(test_input)}
try:
    result = eval(f"{{func_name}}(*test_input)" if isinstance(test_input, (list, tuple)) else f"{{func_name}}(test_input)")
    print(json.dumps({{"result": result, "error": None}}))
except Exception as e:
    print(json.dumps({{"result": None, "error": str(e)}}))
"""
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode='w') as temp_file:
        temp_file.write(wrapped_code)
        temp_file_path = temp_file.name
    
    try:
        result = subprocess.run(
            [sys.executable, temp_file_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        os.remove(temp_file_path)
        
        if result.returncode == 0 and result.stdout:
            output = json.loads(result.stdout.strip())
            logger.debug(f"Execution successful: {output}")
            return output
        
        logger.warning(f"Execution failed with return code {result.returncode}: {result.stderr}")
        return {"result": None, "error": result.stderr or "Execution failed"}
    
    except subprocess.TimeoutExpired:
        os.remove(temp_file_path)
        logger.warning(f"Execution timeout after {timeout}s for input: {test_input}")
        return {"result": None, "error": "Timeout"}
    except Exception as e:
        logger.error(f"Execution error: {str(e)}", exc_info=True)
        return {"result": None, "error": str(e)}


def analyze_code(state):
    """Analyze code for semantic differences and potential edge cases."""
    if VERBOSITY >= 3:
        logger.info(f"Starting analysis iteration {state['iteration']}/{state['max_iterations']}")
    
    previous_analysis = f"\n\nPrevious analysis:\n{state['analysis']}" if state['analysis'] else ""
    coverage_info = f"\n\nCoverage gaps identified:\n{json.dumps(state['coverage_gaps'], indent=2)}" if state['coverage_gaps'] else ""
    
    prompt = PROMPTS['analysis']['template'].format(
        code1=state['code1'],
        code2=state['code2'],
        previous_analysis=previous_analysis,
        coverage_info=coverage_info
    )
    
    try:
        response = model.generate_content(prompt)
        state["analysis"] = response.text
        if VERBOSITY >= 3:
            logger.debug(f"Full analysis: {response.text}")
            logger.info(f"Analysis complete. Response length: {len(response.text)} chars")
    except Exception as e:
        logger.error(f"Failed to generate analysis: {str(e)}", exc_info=True)
        raise
    
    return state


def generate_tests(state):
    """Generate targeted test cases based on analysis."""
    if VERBOSITY >= 3:
        logger.info("Generating targeted test cases")
    
    # Extract function signature to understand parameters
    try:
        tree = ast.parse(state['code1'])
        func_def = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)][0]
        func_name = func_def.name
        num_params = len(func_def.args.args)
        logger.debug(f"Detected function '{func_name}' with {num_params} parameters")
    except Exception as e:
        logger.warning(f"Failed to parse function signature: {str(e)}")
        func_name = "unknown"
        num_params = 1
    
    prompt = PROMPTS['test_generation']['template'].format(
        analysis=state['analysis'],
        num_tests=5,
        num_params=num_params
    )
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Extract JSON from markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        new_tests = json.loads(response_text)
        if isinstance(new_tests, dict) and "tests" in new_tests:
            new_tests = new_tests["tests"]
        
        state["test_cases"].extend(new_tests)
        if VERBOSITY >= 3:
            logger.info(f"Generated {len(new_tests)} new test cases (total: {len(state['test_cases'])})")
        
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to parse test cases: {str(e)}")
        if VERBOSITY >= 3:
            logger.debug(f"Response text: {response_text}")
    except Exception as e:
        logger.error(f"Error generating tests: {str(e)}", exc_info=True)
    
    return state


def run_tests(state):
    """Execute all test cases and compare results."""
    if VERBOSITY >= 3:
        logger.info(f"Running {len(state['test_cases'])} test cases")
    
    differences_found = []
    passed = 0
    failed = 0
    
    for i, test in enumerate(state["test_cases"]):
        test_input = test.get("input", [])
        description = test.get("description", "")
        
        logger.debug(f"Test {i+1}: {description} with input {test_input}")
        
        result1 = execute_code_with_input(state['code1'], test_input)
        result2 = execute_code_with_input(state['code2'], test_input)
        
        if result1["result"] != result2["result"] or result1["error"] != result2["error"]:
            diff = {
                "test_input": test_input,
                "description": description,
                "code1_result": result1,
                "code2_result": result2
            }
            differences_found.append(diff)
            failed += 1
            
            if VERBOSITY >= 2:
                logger.warning(f"Test {i+1} FAILED: {description}")
            if VERBOSITY >= 3:
                logger.info(f"  Input: {test_input}")
                logger.info(f"  Code1: {result1}")
                logger.info(f"  Code2: {result2}")
        else:
            passed += 1
            if VERBOSITY >= 3:
                logger.debug(f"Test {i+1} passed: {description}")
    
    if VERBOSITY >= 3:
        logger.info(f"Test results: {passed} passed, {failed} failed")
    elif VERBOSITY >= 2 and failed > 0:
        logger.warning(f"{failed} tests failed")
    
    if differences_found:
        state["differences"].extend(differences_found)
        if VERBOSITY >= 3:
            logger.info(f"Total differences found so far: {len(state['differences'])}")
    
    return state


def evaluate_coverage(state):
    """Adversarially evaluate test coverage and identify gaps."""
    if VERBOSITY >= 3:
        logger.info("Evaluating test coverage")
    
    prompt = PROMPTS['coverage_evaluation']['template'].format(
        analysis=state['analysis'],
        num_differences=len(state['differences'])
    )
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Extract JSON from markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        gaps = json.loads(response_text)
        state["coverage_gaps"] = [g.get("gap", str(g)) for g in gaps]
        if VERBOSITY >= 3:
            logger.info(f"Identified {len(state['coverage_gaps'])} coverage gaps")
        
        # Calculate confidence
        num_tests = len(state["test_cases"])
        num_gaps = len(state["coverage_gaps"])
        num_diffs = len(state["differences"])
        
        base_confidence = min(0.95, num_tests / 15)
        gap_penalty = min(0.3, num_gaps * 0.05)
        
        state["confidence"] = max(0.1, base_confidence - gap_penalty)
        
        if VERBOSITY >= 3:
            logger.info(f"Confidence score: {state['confidence']:.2%} "
                       f"(tests: {num_tests}, gaps: {num_gaps}, diffs: {num_diffs})")
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse coverage gaps: {str(e)}")
        if VERBOSITY >= 3:
            logger.debug(f"Response text: {response_text}")
        state["confidence"] = 0.6
    except Exception as e:
        logger.error(f"Error evaluating coverage: {str(e)}", exc_info=True)
        state["confidence"] = 0.6
    
    state["iteration"] += 1
    return state


def should_continue(state):
    """Decide whether to continue iterating or finalize."""
    if state["iteration"] >= state["max_iterations"]:
        if VERBOSITY >= 3:
            logger.info(f"Max iterations ({state['max_iterations']}) reached, finalizing")
        return "finalize"
    
    if state["confidence"] > 0.85 and len(state["coverage_gaps"]) < 2:
        if VERBOSITY >= 3:
            logger.info(f"High confidence ({state['confidence']:.2%}) and good coverage achieved, finalizing")
        return "finalize"
    
    if state["confidence"] < 0.85:
        if VERBOSITY >= 3:
            logger.info(f"Confidence below threshold ({state['confidence']:.2%}), continuing iteration")
        return "continue"
    
    if VERBOSITY >= 3:
        logger.info("Continuing iteration for better coverage")
    return "finalize"


def finalize(state):
    """Produce final report and request semantic diff from Gemini."""
    if VERBOSITY >= 2:
        logger.info("="*60)
        logger.info("SEMANTIC DIFF ANALYSIS COMPLETE")
        logger.info("="*60)
        logger.info(f"Iterations: {state['iteration']}")
        logger.info(f"Test cases run: {len(state['test_cases'])}")
        logger.info(f"Differences found: {len(state['differences'])}")
        logger.info(f"Final confidence: {state['confidence']:.2%}")
    
    if state['differences']:
        if VERBOSITY >= 2:
            logger.warning(f"SEMANTIC DIFFERENCES DETECTED ({len(state['differences'])} total)")
        
        if VERBOSITY >= 2:
            for i, diff in enumerate(state['differences'], 1):
                logger.warning(f"  {i}. {diff['description']}")
        
        if VERBOSITY >= 3:
            for i, diff in enumerate(state['differences'], 1):
                logger.info(f"     Input: {diff['test_input']}")
                logger.info(f"     Code 1: {diff['code1_result']}")
                logger.info(f"     Code 2: {diff['code2_result']}")
    else:
        if VERBOSITY >= 2:
            logger.info("✓ No semantic differences detected (within tested scope)")
    
    if state['coverage_gaps'] and VERBOSITY >= 3:
        logger.info("Coverage gaps identified:")
        for gap in state['coverage_gaps']:
            logger.info(f"  - {gap}")
    
    if VERBOSITY >= 3:
        logger.info("Requesting semantic diff from Gemini...")

    prompt = PROMPTS['semantic_diff']['template'].format(
        code1=state['code1'],
        code2=state['code2'],
        num_differences=len(state['differences']),
        num_tests=len(state['test_cases']),
        coverage_gaps=json.dumps(state['coverage_gaps'], indent=2),
        test_cases=state['test_cases']
    )
    
    try:
        response = model.generate_content(prompt)
        diff_result = response.text.strip()
        
        if diff_result:
            # At verbose=1, ONLY print the diff (no headers, no formatting)
            if VERBOSITY >= 1:
                print(diff_result)
        else:
            if VERBOSITY >= 2:
                logger.warning("No semantic diff result returned from Gemini")
    except Exception as e:
        logger.error(f"Failed to generate semantic diff: {str(e)}", exc_info=True)

    return state


# Build LangGraph workflow
graph = StateGraph(State)
graph.add_node("analyze", analyze_code)
graph.add_node("generate_tests", generate_tests)
graph.add_node("run_tests", run_tests)
graph.add_node("evaluate_coverage", evaluate_coverage)
graph.add_node("finalize", finalize)

graph.add_edge(START, "analyze")
graph.add_edge("analyze", "generate_tests")
graph.add_edge("generate_tests", "run_tests")
graph.add_edge("run_tests", "evaluate_coverage")
graph.add_conditional_edges(
    "evaluate_coverage", 
    should_continue, 
    {"continue": "analyze", "finalize": "finalize"}
)
graph.add_edge("finalize", END)

app = graph.compile()


def run_analysis(code1, code2, max_iterations=5, verbose=1):
    """Run semantic diff analysis with specified verbosity.
    
    Args:
        code1 (str): First code snippet
        code2 (str): Second code snippet
        max_iterations (int): Maximum analysis iterations
        verbose (int): Verbosity level (0=silent, 1=low, 2=medium, 3=full)
    
    Returns:
        dict: Analysis results including differences and confidence
    """
    global logger
    logger = setup_logging(verbose)
    
    if verbose >= 3:
        logger.info("Starting Semantic Code Diff Analyzer")
        logger.info(f"Verbosity level: {verbose}")
    
    try:
        result = app.invoke({
            "code1": code1,
            "code2": code2,
            "confidence": 0.0,
            "differences": [],
            "analysis": "",
            "test_cases": [],
            "iteration": 1,
            "max_iterations": max_iterations,
            "coverage_gaps": []
        })
        
        if verbose >= 3:
            logger.info("Analysis completed successfully")
        
        return result
        
    except Exception as e:
        logger.critical(f"Analysis failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Semantic Code Diff Analyzer')
    parser.add_argument(
        '--verbose', '-v',
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help='Verbosity level: 0=silent, 1=final output only, 2=basic info, 3=all details (default: 1)'
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=5,
        help='Maximum number of analysis iterations'
    )
    
    args = parser.parse_args()
    
    code1 = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""

    code2 = """
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
"""

    result = run_analysis(code1, code2, max_iterations=args.max_iterations, verbose=args.verbose)
