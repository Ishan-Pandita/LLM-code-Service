"""
Best Practices Module - Main module implementation.

Uses ONE-RULE-PER-GENERATION architecture:
- Each rule is evaluated in a separate LLM call
- Avoids cross-rule contamination
- Avoids JSON drift
- Avoids overconfidence bias
- Results are merged at the end
"""

import json
import logging
import asyncio
from typing import Any, Dict, List, Optional

from llm_service.core.inference_engine import GenerationConfig, InferenceEngine
from llm_service.modules.base import BaseModule, ModuleInput, ModuleOutput, get_module_registry

from .schemas import Rule, EvaluationResponse, RuleResult
from .helpers import (
    load_predefined_rules,
    get_rules_by_ids,
    build_single_rule_messages,
    extract_json,
    validate_single_rule_output,
    calculate_overall_status
)

logger = logging.getLogger(__name__)


class BestPracticesInput(ModuleInput):
    """Input schema for best practices module (used by base router)."""
    
    language: str
    code: str
    predefined_rules: Optional[List[str]] = None
    custom_rules: Optional[List[Dict[str, Any]]] = None


class BestPracticesOutput(ModuleOutput):
    """Output schema for best practices module."""
    
    overall_status: str  # Format: "X/Y" (e.g., "8/10")
    rules: List[Dict[str, Any]]


class BestPracticesModule(BaseModule):
    """
    Best Practices Module - Evaluates code against coding best practice rules.
    
    Uses ONE-RULE-PER-GENERATION architecture:
    - Each rule is evaluated in a separate LLM call
    - Results are merged at the end
    - Overall status is "passed/total" format
    
    Benefits:
    - No cross-rule contamination
    - No JSON drift
    - No overconfidence bias
    - More reliable per-rule evaluation
    """
    
    module_id = "best_practices"
    input_schema = BestPracticesInput
    output_schema = BestPracticesOutput
    
    def __init__(self, inference_engine: InferenceEngine) -> None:
        super().__init__(inference_engine)
        self._predefined_rules = None
    
    @property
    def predefined_rules(self) -> List[Rule]:
        """Lazy load predefined rules."""
        if self._predefined_rules is None:
            self._predefined_rules = load_predefined_rules()
        return self._predefined_rules
    
    def get_rules_for_evaluation(
        self,
        predefined_rule_ids: Optional[List[str]] = None,
        custom_rules: Optional[List[Dict[str, Any]]] = None
    ) -> List[Rule]:
        """
        Get the list of rules to use for evaluation.
        
        Args:
            predefined_rule_ids: List of predefined rule IDs to use
            custom_rules: List of custom rule dictionaries
            
        Returns:
            Combined list of Rule objects
        """
        rules = []
        
        # Add predefined rules
        if predefined_rule_ids is not None:
            # Use only specified predefined rules
            rules.extend(get_rules_by_ids(predefined_rule_ids))
        
        # Add custom rules
        if custom_rules:
            for cr in custom_rules:
                try:
                    rules.append(Rule.model_validate(cr))
                except Exception as e:
                    logger.warning(f"Invalid custom rule: {e}")
        
        return rules
    
    def build_system_prompt(self) -> str:
        """Build system prompt (not used directly, using helpers)."""
        return ""
    
    def build_user_prompt(self, payload: Dict[str, Any]) -> str:
        """Build user prompt (not used directly, using helpers)."""
        return ""
    
    def get_generation_config(self) -> GenerationConfig:
        """Get generation config optimized for single-rule JSON output."""
        return GenerationConfig(
            max_new_tokens=512,  # Smaller since only one rule
            temperature=0.0,     # Deterministic
            top_p=1.0,
            top_k=-1,
            repetition_penalty=1.0,
            do_sample=False,
        )
    
    async def evaluate_single_rule(
        self,
        language: str,
        code: str,
        rule: Rule,
        use_function_calling: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate code against a SINGLE rule.
        
        Uses function calling for more reliable structured output,
        with fallback to JSON parsing.
        
        Args:
            language: Programming language
            code: Code to evaluate
            rule: Single rule to evaluate
            use_function_calling: Whether to use function calling
            
        Returns:
            Rule evaluation result dictionary
        """
        logger.debug(f"Evaluating rule: {rule.id} - {rule.name}")
        
        config = self.get_generation_config()
        
        # Try function calling first
        if use_function_calling:
            try:
                from .helpers import (
                    get_evaluate_rule_tool,
                    build_single_rule_messages_with_tools,
                    parse_function_call_result
                )
                
                # Build tool-aware messages
                messages = build_single_rule_messages_with_tools(language, code, rule)
                tool = get_evaluate_rule_tool()
                
                # Generate with tools
                result = await self.inference_engine.generate_with_tools(
                    messages, 
                    tools=[tool],
                    config=config,
                    chat_template_params={"enable_thinking": False}
                )
                
                # Check if model used function calling
                if result.has_function_calls:
                    for fc in result.function_calls:
                        if fc.name == "report_evaluation":
                            parsed = parse_function_call_result(fc)
                            # Ensure rule_id matches
                            parsed["rule_id"] = rule.id
                            logger.debug(f"Got function call result for {rule.id}")
                            return parsed
                
                # If no function call, fall through to fallback
                logger.debug(f"No function call for {rule.id}, trying text extraction")
                
            except Exception as e:
                logger.warning(f"Function calling failed for {rule.id}, using fallback: {e}")
        
        # Fallback: Traditional JSON parsing
        messages = build_single_rule_messages(language, code, rule)
        result = await self.inference_engine.generate(
            messages, 
            config,
            chat_template_params={"enable_thinking": False}
        )
        
        try:
            json_str = extract_json(result.generated_text)
            parsed = json.loads(json_str)
            
            # Validate and fix if needed
            validate_single_rule_output(parsed, rule)
            
            return parsed
            
        except Exception as e:
            logger.error(f"Failed to parse output for rule {rule.id}: {e}")
            # Return a failed result with error
            return {
                "rule_id": rule.id,
                "status": "FAIL",
                "confidence": "LOW",
                "evidence": f"Evaluation error: {str(e)}",
                "suggestion": "Manual review required"
            }
    
    async def evaluate(
        self,
        language: str,
        code: str,
        predefined_rule_ids: Optional[List[str]] = None,
        custom_rules: Optional[List[Dict[str, Any]]] = None
    ) -> EvaluationResponse:
        """
        Evaluate code against best practice rules using ONE-RULE-PER-GENERATION.
        
        Each rule is evaluated in a separate LLM call.
        All calls are executed in parallel using `asyncio.gather` for maximum throughput
        with vLLM's continuous batching.
        
        Args:
            language: Programming language
            code: Code to evaluate
            predefined_rule_ids: Optional list of predefined rule IDs
            custom_rules: Optional list of custom rules
            
        Returns:
            EvaluationResponse with merged results
        """
        # Get rules to evaluate
        rules = self.get_rules_for_evaluation(predefined_rule_ids, custom_rules)
        
        if not rules:
            raise ValueError("No rules specified for evaluation")
        
        logger.info(f"Evaluating code against {len(rules)} rules (parallel)")
        
        # Create async tasks for each rule
        tasks = [
            self.evaluate_single_rule(language, code, rule)
            for rule in rules
        ]
        
        # Execute all tasks in parallel
        rule_results = await asyncio.gather(*tasks)
        
        # Calculate overall status (passed/total)
        overall_status = calculate_overall_status(rule_results)
        
        logger.info(f"Evaluation complete: {overall_status}")
        
        # Convert to response model
        validated_results = [
            RuleResult.model_validate(r) for r in rule_results
        ]
        
        return EvaluationResponse(
            overall_status=overall_status,
            rules=validated_results
        )
    
    def parse_output(self, raw_output: str) -> BestPracticesOutput:
        """Parse output (used by base router if called via standard route)."""
        json_str = extract_json(raw_output)
        parsed = json.loads(json_str)
        
        return BestPracticesOutput(
            overall_status=parsed.get("overall_status", "0/0"),
            rules=parsed.get("rules", [])
        )
    
    async def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute via base module interface."""
        validated = self.validate_input(payload)
        
        result = await self.evaluate(
            language=validated.language,
            code=validated.code,
            predefined_rule_ids=validated.predefined_rules,
            custom_rules=validated.custom_rules
        )
        
        return {
            "module_id": self.module_id,
            "output": result.model_dump(),
            "usage": {}  # Would be populated by inference engine
        }


# Register the module
get_module_registry().register(BestPracticesModule)
