"""
Best Practices Module - Helper functions for prompt building and validation.

Uses one-rule-per-generation architecture to avoid:
- Cross-rule contamination
- JSON drift
- Overconfidence bias
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any

from .schemas import Rule

logger = logging.getLogger(__name__)

# Path to the rules JSON file
RULES_FILE = Path(__file__).parent / "rules.json"


def load_predefined_rules() -> List[Rule]:
    """
    Load predefined rules from the JSON file.
    
    Returns:
        List of Rule objects
    """
    try:
        with open(RULES_FILE, "r", encoding="utf-8") as f:
            rules_data = json.load(f)
        return [Rule.model_validate(r) for r in rules_data]
    except Exception as e:
        logger.error(f"Failed to load predefined rules: {e}")
        return []


def get_rules_by_ids(rule_ids: List[str]) -> List[Rule]:
    """
    Get predefined rules by their IDs.
    
    Args:
        rule_ids: List of rule IDs to fetch
        
    Returns:
        List of matching Rule objects
    """
    all_rules = load_predefined_rules()
    rules_map = {r.id: r for r in all_rules}
    
    result = []
    for rid in rule_ids:
        if rid in rules_map:
            result.append(rules_map[rid])
        else:
            logger.warning(f"Rule ID '{rid}' not found in predefined rules")
    
    return result


def build_single_rule_messages(language: str, code: str, rule: Rule) -> List[Dict[str, str]]:
    """
    Build chat messages for evaluating a SINGLE rule.
    
    This is the one-rule-per-generation architecture that avoids:
    - Cross-rule contamination
    - JSON drift
    - Overconfidence bias
    
    Args:
        language: Programming language of the code
        code: The code to evaluate
        rule: Single rule to evaluate
        
    Returns:
        List of message dictionaries for chat template
    """
    # Optimized system prompt
    # Key: Concise role + structured guidelines + strict output format
    system_msg = """You are an expert code quality evaluator specializing in academic programming assignments.

<identity>
- Expert in code review and best practices across multiple languages
- Fair and evidence-based evaluator for student work
- Focused on educational feedback, not production-grade expectations
</identity>

<evaluation_approach>
Before evaluating, you MUST:
1. READ the rule description carefully to understand what to look for
2. SCAN the code for relevant patterns (both positive and negative)
3. IDENTIFY specific lines or constructs as evidence
4. DECIDE pass/fail based on evidence strength only
5. FORMULATE actionable suggestion if failing
</evaluation_approach>

<confidence_calibration>
- HIGH: Unambiguous evidence of compliance or violation (80%+ certainty)
- MEDIUM: Reasonable evidence but some interpretation required (50-80%)
- LOW: Edge case or insufficient information to judge (<50%)
</confidence_calibration>

<output_requirements>
Output EXACTLY ONE valid JSON object. No text before or after.
Schema:
{
  "rule_id": "string - exact ID from input",
  "status": "PASS" | "FAIL",
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "evidence": "string - specific code quote or line reference",
  "suggestion": "string | null - improvement if FAIL, null if PASS"
}
</output_requirements>

<critical_rules>
- Evaluate ONLY the specified rule, ignore other issues
- Never invent rule IDs or evaluate unspecified rules
- Always provide evidence from the actual code
- Output raw JSON only, no markdown formatting
</critical_rules>"""

    # Optimized user prompt
    # Key: Clear task + structured content + explicit action
    user_msg = f"""<task>
Evaluate this {language} code against the following rule.
</task>

<rule id="{rule.id}">
Name: {rule.name}
Description: {rule.description}
Category: {rule.category or "general"}
</rule>

<code language="{language}">
{code}
</code>

<action>
1. Reason about whether the code satisfies or violates rule "{rule.id}"
2. Find specific evidence in the code
3. Output your evaluation as JSON
</action>"""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def extract_json(text: str) -> str:
    """
    Extract JSON object from text, handling markdown code blocks.
    
    Args:
        text: Text potentially containing JSON
        
    Returns:
        Extracted JSON string
        
    Raises:
        ValueError: If no valid JSON found
    """
    text = text.strip()
    
    # Try markdown code block first
    json_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
    match = re.search(json_pattern, text)
    if match:
        return match.group(1)
    
    # Try standalone JSON
    start = text.find("{")
    end = text.rfind("}") + 1
    
    if start != -1 and end > start:
        return text[start:end]
    
    raise ValueError("No valid JSON found in model output")


def validate_single_rule_output(parsed: Dict[str, Any], rule: Rule) -> None:
    """
    Validate the LLM output for a single rule evaluation.
    
    Args:
        parsed: Parsed JSON from LLM
        rule: The rule that was evaluated
        
    Raises:
        ValueError: If validation fails
    """
    # Check rule_id matches
    if parsed.get("rule_id") != rule.id:
        logger.warning(f"Rule ID mismatch: expected {rule.id}, got {parsed.get('rule_id')}")
        # Fix it
        parsed["rule_id"] = rule.id
    
    # Check status
    if parsed.get("status") not in ("PASS", "FAIL"):
        raise ValueError(f"Invalid status for rule {rule.id}: {parsed.get('status')}")
    
    # Check confidence
    if parsed.get("confidence") not in ("HIGH", "MEDIUM", "LOW"):
        # Default to MEDIUM if missing
        parsed["confidence"] = "MEDIUM"
    
    # Ensure evidence exists
    if not parsed.get("evidence"):
        parsed["evidence"] = "No specific evidence provided"


def calculate_overall_status(results: List[Dict[str, Any]]) -> str:
    """
    Calculate overall status as passed/total format.
    
    Args:
        results: List of rule evaluation results
        
    Returns:
        String in format "X/Y" (e.g., "8/10")
    """
    total = len(results)
    passed = sum(1 for r in results if r.get("status") == "PASS")
    return f"{passed}/{total}"



# Function Calling Support


def get_evaluate_rule_tool():
    """
    Get the tool definition for rule evaluation.
    
    This leverages native function calling capabilities
    for more reliable structured output.
    
    Returns:
        ToolDefinition for rule evaluation
    """
    from llm_service.core.function_calling import ToolDefinition, ToolParameter
    
    return ToolDefinition(
        name="report_evaluation",
        description="Report the evaluation result for a coding rule",
        parameters=[
            ToolParameter(
                name="rule_id",
                description="The exact ID of the rule being evaluated",
                type="string",
                required=True
            ),
            ToolParameter(
                name="status",
                description="PASS if code follows the rule, FAIL if it violates",
                type="string",
                required=True
            ),
            ToolParameter(
                name="confidence",
                description="Confidence level: HIGH, MEDIUM, or LOW",
                type="string",
                required=True
            ),
            ToolParameter(
                name="evidence",
                description="Specific code quote or reference supporting the evaluation",
                type="string",
                required=True
            ),
            ToolParameter(
                name="suggestion",
                description="Improvement suggestion if FAIL, or null if PASS",
                type="string",
                required=False
            )
        ]
    )


def build_single_rule_messages_with_tools(
    language: str, 
    code: str, 
    rule: Rule
) -> List[Dict[str, str]]:
    """
    Build messages for function calling mode.
    
    Uses a streamlined prompt since output structure is enforced
    by the function calling mechanism.
    
    Args:
        language: Programming language
        code: Code to evaluate
        rule: Rule to evaluate against
        
    Returns:
        List of message dictionaries
    """
    # Function calling optimized system prompt
    # Simpler since structure is enforced by the tool schema
    system_msg = """You are an expert code quality evaluator.

<task>
Evaluate student code against a specific coding rule, then call the report_evaluation function with your findings.
</task>

<evaluation_process>
1. Read the rule description carefully
2. Scan the code for relevant patterns
3. Identify specific evidence (quote lines if possible)
4. Decide PASS/FAIL based on evidence
5. Call report_evaluation with your assessment
</evaluation_process>

<confidence_guide>
- HIGH: Clear evidence, 80%+ certain
- MEDIUM: Reasonable evidence, 50-80% certain
- LOW: Edge case or borderline, <50% certain
</confidence_guide>"""

    # Function calling optimized user prompt
    user_msg = f"""<rule id="{rule.id}">
{rule.name}: {rule.description}
</rule>

<code language="{language}">
{code}
</code>

Evaluate this code against rule "{rule.id}" and call report_evaluation with your result."""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]


def parse_function_call_result(function_call) -> Dict[str, Any]:
    """
    Parse a function call result into evaluation result format.
    
    Args:
        function_call: FunctionCall object from the model
        
    Returns:
        Evaluation result dictionary
    """
    args = function_call.arguments
    
    return {
        "rule_id": args.get("rule_id", ""),
        "status": args.get("status", "FAIL"),
        "confidence": args.get("confidence", "MEDIUM"),
        "evidence": args.get("evidence", "No evidence provided"),
        "suggestion": args.get("suggestion")
    }

