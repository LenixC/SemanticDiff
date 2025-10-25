# Semantic Code Diff Analyzer

An LLM-powered agentic testing framework that identifies **semantic differences** between functionally similar code snippets through adversarial test generation and execution.

## Overview

Unlike traditional diff tools that compare syntax, this tool detects behavioral differences by:
1. Analyzing code semantically using Gemini 2.5 Flash
2. Generating targeted test cases based on identified edge cases
3. Executing both code versions in isolated subprocesses
4. Iteratively refining test coverage through adversarial evaluation
5. Producing a confidence-scored semantic diff report

## Architecture

Built on [LangGraph](https://github.com/langchain-ai/langgraph), the tool implements an agentic, stateful directed graph that autonomously decides when to continue testing or finalize based on confidence and coverage metrics.

```
START → analyze → generate_tests → run_tests → evaluate_coverage → [continue/finalize] → END
           ↑                                              |
           └──────────────────────────────────────────────┘
```

### Node Descriptions

- **analyze**: LLM-powered semantic analysis identifying potential behavioral differences, edge cases, and boundary conditions
- **generate_tests**: Generates 5 targeted test cases per iteration based on analysis and coverage gaps
- **run_tests**: Executes both code versions in isolated subprocesses with timeout protection, comparing outputs
- **evaluate_coverage**: Adversarially evaluates test coverage, identifies gaps, calculates confidence score
- **finalize**: Produces comprehensive report and requests final semantic diff visualization from LLM

### State Management

```python
class State(TypedDict):
    code1: str                          # First code snippet
    code2: str                          # Second code snippet
    confidence: float                   # Current confidence score (0-1)
    differences: List[Dict[str, Any]]   # Semantic differences found
    analysis: str                       # Cumulative analysis text
    test_cases: List[Dict[str, Any]]    # All generated test cases
    iteration: int                      # Current iteration number
    max_iterations: int                 # Maximum iterations allowed
    coverage_gaps: List[str]            # Identified testing gaps
```

## Installation

```bash
# Clone repository
git clone <repository-url>
cd semantic-diff-analyzer

# Install dependencies
pip install langgraph google-generativeai python-dotenv

# Configure environment
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

## Usage

### Basic Usage

```python
from semantic_diff import run_analysis

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

result = run_analysis(code1, code2, max_iterations=5, verbose=1)
```

## How It Works

### 1. Semantic Analysis Phase

The LLM analyzes both code snippets for:
- Behavioral differences (not just syntax)
- Edge cases that might expose differences
- Input ranges requiring testing
- Boundary conditions and error cases

### 2. Test Generation Phase

Based on analysis, generates test cases covering:
- Normal operation ranges
- Boundary conditions (min/max values, empty inputs)
- Error scenarios (invalid inputs, type mismatches)
- Special values (zero, negative numbers, None)

### 3. Execution Phase

Each test case is executed in an isolated subprocess with:
- **Timeout protection**: 5-second execution limit
- **Output capture**: Both stdout and stderr
- **Error handling**: Graceful failure on exceptions
- **Comparison**: Exact output and error matching

### 4. Coverage Evaluation Phase

Adversarial analysis identifies:
- Untested edge cases
- Missing boundary conditions
- Unexplored error scenarios
- Input combinations that might reveal hidden differences

### 5. Confidence Scoring

```python
base_confidence = min(0.95, num_tests / 15)    # Increases with test count
gap_penalty = min(0.3, num_gaps * 0.05)        # Decreases with coverage gaps
final_confidence = max(0.1, base_confidence - gap_penalty)
```

### 6. Termination Conditions

The agentic system autonomously decides to terminate when:
- **High confidence**: confidence > 0.85 AND coverage_gaps < 2
- **Max iterations**: iteration >= max_iterations
- Otherwise continues refining tests

## Output Format

### Console Output

```
[Iteration 1] Analyzing code with Gemini...
Analysis: The main semantic difference lies in...
Generating targeted test cases...
Generated 5 new test cases (total: 5)
Running test cases...
  ✓ Test 1 passed: normal positive integer
  ✓ Test 2 passed: zero input
  ❌ Test 3 FAILED: negative input
     Input: [-5]
     Code1: {'result': 1, 'error': None}
     Code2: {'result': 1, 'error': None}
Evaluating test coverage...
Confidence: 0.65 (tests: 5, gaps: 4, diffs: 0)

============================================================
SEMANTIC DIFF ANALYSIS COMPLETE
============================================================
Iterations: 3
Test cases run: 15
Differences found: 2
Final confidence: 87.00%

❌ SEMANTIC DIFFERENCES DETECTED:
  1. negative input handling
     Input: [-5]
     Code 1: {'result': 1, 'error': None}
     Code 2: {'result': None, 'error': 'range error'}
```

### Semantic Diff Result

The final output includes a Git-style diff highlighting semantically different sections:

```diff
- if n <= 1:
-     return 1
+ if n < 0:
+     raise ValueError("n must be non-negative")
+ if n <= 1:
+     return 1
```

## Technical Considerations

### Security

- **Subprocess isolation**: Code executes in separate processes
- **Timeout protection**: Prevents infinite loops
- **No shell injection**: Direct Python execution only
- **Tempfile cleanup**: Automatic cleanup even on errors

### Limitations

1. **Function-level analysis only**: Doesn't handle classes or modules
2. **Simple signatures**: Assumes positional arguments, no keyword args
3. **Single function per snippet**: Multiple functions not supported
4. **Synchronous execution**: Tests run sequentially (could be parallelized)
5. **AST parsing fragility**: Complex code structures may fail

### Performance Characteristics

- **Typical runtime**: 90-180 seconds for 3-5 iterations
- **LLM calls**: 3N + 1 calls (N = number of iterations)
- **Subprocess overhead**: ~100-200ms per test case
- **Token usage**: ~2K-5K tokens per iteration
