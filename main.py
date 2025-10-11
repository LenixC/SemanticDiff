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

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel(
    'gemini-2.5-flash',
    generation_config=genai.GenerationConfig(
        max_output_tokens=1_000_000,
        temperature=0.7,
    )
)

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
    """Execute code with specific input and return output."""
    # Wrap code to accept input and return output
    wrapped_code = f"""
import json
import sys

{code}

# Get the function name from the code
import ast
tree = ast.parse('''{code}''')
func_name = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)][0]

# Execute with test input
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
            return json.loads(result.stdout.strip())
        return {"result": None, "error": result.stderr or "Execution failed"}
    
    except subprocess.TimeoutExpired:
        os.remove(temp_file_path)
        return {"result": None, "error": "Timeout"}
    except Exception as e:
        return {"result": None, "error": str(e)}


def analyze_code(state):
    """Analyze code for semantic differences and potential edge cases."""
    print(f"\n[Iteration {state['iteration']}] Analyzing code with Gemini...")
    
    previous_analysis = f"\n\nPrevious analysis:\n{state['analysis']}" if state['analysis'] else ""
    coverage_info = f"\n\nCoverage gaps identified:\n{json.dumps(state['coverage_gaps'], indent=2)}" if state['coverage_gaps'] else ""
    
    prompt = f"""
    Compare these two Python code snippets for semantic differences:
    
    Code 1:
    ```python
    {state['code1']}
    ```
    
    Code 2:
    ```python
    {state['code2']}
    ```
    
    {previous_analysis}
    {coverage_info}
    
    Analyze:
    1. What are the semantic differences (behavior, not just syntax)?
    2. What edge cases might expose differences?
    3. What input ranges should be tested?
    4. Are there boundary conditions, error cases, or special values to consider?
    
    Be specific and technical.
    """
    
    response = model.generate_content(prompt)
    state["analysis"] = response.text
    print(f"Analysis: {response.text[:300]}...")
    return state


def generate_tests(state):
    """Generate targeted test cases based on analysis."""
    print("Generating targeted test cases...")
    
    # Extract function signature to understand parameters
    try:
        tree = ast.parse(state['code1'])
        func_def = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)][0]
        func_name = func_def.name
        num_params = len(func_def.args.args)
    except:
        func_name = "unknown"
        num_params = 1
    
    test_prompt = f"""
    Based on this analysis:
    {state['analysis']}

    Generate exactly 5 test cases for a function with {num_params} parameter(s).

    Return ONLY valid JSON array, no markdown, no explanation:
    [{{"input": [5], "description": "normal"}}, {{"input": [0], "description": "zero"}}]
    """
    
    response = model.generate_content(test_prompt)
    response_text = response.text.strip()
    
    # Extract JSON from markdown code blocks if present
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0].strip()
    
    try:
        new_tests = json.loads(response_text)
        if isinstance(new_tests, dict) and "tests" in new_tests:
            new_tests = new_tests["tests"]
        state["test_cases"].extend(new_tests)
        print(f"Generated {len(new_tests)} new test cases (total: {len(state['test_cases'])})")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse test cases: {e}")
        print(f"Response text: {response_text}")
    
    return state


def run_tests(state):
    """Execute all test cases and compare results."""
    print("Running test cases...")
    
    differences_found = []
    
    for i, test in enumerate(state["test_cases"]):
        test_input = test.get("input", [])
        description = test.get("description", "")
        
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
            print(f"  ❌ Test {i+1} FAILED: {description}")
            print(f"     Input: {test_input}")
            print(f"     Code1: {result1}")
            print(f"     Code2: {result2}")
        else:
            print(f"  ✓ Test {i+1} passed: {description}")
    
    if differences_found:
        state["differences"].extend(differences_found)
    
    return state


def evaluate_coverage(state):
    """Adversarially evaluate test coverage and identify gaps."""
    print("Evaluating test coverage...")
    
    coverage_prompt = f"""
    Given this code comparison analysis:
    {state['analysis']}
    
    And these test cases that were run:
    [{{"gap": "description of what's not tested", "why_important": "why this matters"}}]
    
    Differences found: {len(state['differences'])} differences
    
    Adversarially evaluate:
    1. What edge cases are NOT covered?
    2. What boundary conditions are missing?
    3. What error scenarios haven't been tested?
    4. Are there input combinations that might reveal hidden differences?
    
    Be critical and specific. Return a JSON array of coverage gaps:
    [{{"gap": "description of what's not tested", "why_important": "why this matters"}}]
    """
    
    response = model.generate_content(coverage_prompt)
    response_text = response.text.strip()
    
    # Extract JSON
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0].strip()
    
    try:
        gaps = json.loads(response_text)
        state["coverage_gaps"] = [g.get("gap", str(g)) for g in gaps]
        print(f"Identified {len(state['coverage_gaps'])} coverage gaps")
        
        # Calculate confidence based on coverage and differences
        num_tests = len(state["test_cases"])
        num_gaps = len(state["coverage_gaps"])
        num_diffs = len(state["differences"])
        
        # Confidence increases with more tests, decreases with more gaps
        base_confidence = min(0.95, num_tests / 15)  # Max at 15+ tests
        gap_penalty = min(0.3, num_gaps * 0.05)      # -5% per gap, max -30%
        
        state["confidence"] = max(0.1, base_confidence - gap_penalty)
        
        print(f"Confidence: {state['confidence']:.2f} (tests: {num_tests}, gaps: {num_gaps}, diffs: {num_diffs})")
        
    except json.JSONDecodeError as e:
        print(f"Failed to parse coverage gaps: {e}")
        state["confidence"] = 0.6  # Moderate confidence if analysis fails
    
    state["iteration"] += 1
    return state


def should_continue(state):
    """Decide whether to continue iterating or finalize."""
    if state["iteration"] >= state["max_iterations"]:
        print("Max iterations reached.")
        return "finalize"
    
    if state["confidence"] > 0.85 and len(state["coverage_gaps"]) < 2:
        print("High confidence and good coverage achieved.")
        return "finalize"
    
    if state["confidence"] < 0.85:
        print(f"Confidence too low ({state['confidence']:.2f}), continuing...")
        return "continue"
    
    return "finalize"


def finalize(state):
    """Produce final report and request semantic diff from Gemini based on the final state."""
    print("\n" + "="*60)
    print("SEMANTIC DIFF ANALYSIS COMPLETE")
    print("="*60)
    print(f"Iterations: {state['iteration']}")
    print(f"Test cases run: {len(state['test_cases'])}")
    print(f"Differences found: {len(state['differences'])}")
    print(f"Final confidence: {state['confidence']:.2%}")
    
    # If there are semantic differences, print them
    if state['differences']:
        print("\n❌ SEMANTIC DIFFERENCES DETECTED:")
        for i, diff in enumerate(state['differences'], 1):
            print(f"\n  {i}. {diff['description']}")
            print(f"     Input: {diff['test_input']}")
            print(f"     Code 1: {diff['code1_result']}")
            print(f"     Code 2: {diff['code2_result']}")
    else:
        print("\n✓ No semantic differences detected (within tested scope)")
    
    # If there are coverage gaps, print them
    if state['coverage_gaps']:
        for gap in state['coverage_gaps']:
            print(f"  - {gap}")
    
    print("\nRequesting semantic diff from Gemini...")

    diff_prompt = f"""
    Compare the following two Python code snippets for **semantic differences**—that is, how their execution behavior and results might differ, based on the final analysis and test results. 

    Code 1:
    ```python
    {state['code1']}
    ```

    Code 2:
    ```python
    {state['code2']}
    ```

    **Final Analysis Context:**
    - Differences identified: {len(state['differences'])} semantic differences found.
    - Test cases run: {len(state['test_cases'])}
    - Coverage gaps: {json.dumps(state['coverage_gaps'], indent=2)}
    - Edge cases explored: {state['test_cases']}
    
    **Instructions:**
    Please output a diff-like format (diff or git diff) that highlights lines that
    are semantically different in concept. Do not add any additional commentary.
    Only reply with the diff-like final output.
    """
    
    # Ask Gemini to return the semantic diff
    response = model.generate_content(diff_prompt)
    diff_result = response.text.strip()
    
    if diff_result:
        print("\nSemantic Code Diff Result (from Gemini):\n")
        print(diff_result)
    else:
        print("\nNo semantic diff result returned from Gemini.")

    return state


# Build graph
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


if __name__ == "__main__":
    # Test with factorial implementations
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

    # Run the semantic diff
    result = app.invoke({
        "code1": code1,
        "code2": code2,
        "confidence": 0.0,
        "differences": [],
        "analysis": "",
        "test_cases": [],
        "iteration": 1,
        "max_iterations": 5,
        "coverage_gaps": []
    })
    
    # Visualize the graph
    #try:
    #    from IPython.display import Image, display
    #    display(Image(app.get_graph().draw_mermaid_png()))
    #except:
    #    print("\nGraph structure:")
    #    print(app.get_graph().draw_ascii())
