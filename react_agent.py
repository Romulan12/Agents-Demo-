"""
ReAct Agent - Reasoning + Acting Pattern
Implements autonomous multi-step reasoning for advanced agentic behavior
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Fix SSL certificate verification issues
import ssl
import certifi
ssl._create_default_https_context = ssl._create_unverified_context

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import BaseTool


class ActionType(Enum):
    """Types of actions the agent can take"""
    TOOL_USE = "tool_use"
    FINISH = "finish"
    REFLECT = "reflect"


@dataclass
class ReActStep:
    """Represents one step in the ReAct loop"""
    iteration: int
    thought: str
    action_type: ActionType
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "iteration": self.iteration,
            "thought": self.thought,
            "action_type": self.action_type.value,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation
        }


class ReActAgent:
    """
    ReAct (Reasoning + Acting) Agent
    
    Implements the ReAct pattern for autonomous multi-step reasoning:
    1. Thought: Reason about what to do next
    2. Action: Select and execute a tool
    3. Observation: Observe the result
    4. Repeat until task is complete
    
    This enables:
    - Multi-step problem solving
    - Self-directed tool chaining
    - Explicit reasoning traces
    - Iterative refinement
    """
    
    def __init__(
        self,
        tools: List[BaseTool],
        llm: OpenAI,
        max_iterations: int = 5,
        verbose: bool = True
    ):
        """
        Initialize ReAct agent.
        
        Args:
            tools: List of available tools
            llm: Language model for reasoning
            max_iterations: Maximum reasoning steps (default: 5)
            verbose: Whether to print reasoning steps (default: True)
        """
        self.tools = {tool.metadata.name: tool for tool in tools}
        self.llm = llm
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # Build tool descriptions for prompting
        self.tool_descriptions = self._build_tool_descriptions()
    
    def _build_tool_descriptions(self) -> str:
        """Build formatted tool descriptions for the prompt"""
        descriptions = []
        for name, tool in self.tools.items():
            desc = f"- {name}: {tool.metadata.description}"
            descriptions.append(desc)
        return "\n".join(descriptions)
    
    def _create_react_prompt(
        self,
        question: str,
        history: List[ReActStep]
    ) -> str:
        """
        Create the ReAct reasoning prompt.
        
        Args:
            question: User's question
            history: Previous reasoning steps
            
        Returns:
            Formatted prompt for the LLM
        """
        # Build history section
        history_text = ""
        if history:
            history_text = "\n\nPrevious steps:\n"
            for step in history:
                history_text += f"\nThought {step.iteration}: {step.thought}\n"
                if step.action:
                    history_text += f"Action {step.iteration}: {step.action}\n"
                    history_text += f"Observation {step.iteration}: {step.observation}\n"
        
        prompt = f"""You are a research assistant using the ReAct (Reasoning + Acting) pattern.

Question: {question}

Available tools:
{self.tool_descriptions}

Instructions:
1. Think step-by-step about what information you need
2. Choose ONE tool to use (or FINISH if you have enough information)
3. After seeing the observation, think about what to do next

CRITICAL RULES - MUST FOLLOW:
- ONLY report information that is EXPLICITLY in the tool output
- If a tool returns no relevant information, acknowledge it and try another tool
- DO NOT use your general knowledge to fill in gaps or make assumptions
- If you cannot find information in papers, use web search tools (web_search or scrape_webpage)
- ALWAYS cite the source of your information (which tool/paper provided it)
- If all paper tools fail to find relevant information, you MUST try web search before finishing

Verification After Each Observation:
- Ask yourself: "Does this observation actually contain relevant information about my query?"
- If observation is irrelevant, empty, or off-topic, acknowledge it and try a different tool
- If you've tried 3+ paper tools without finding relevant info, switch to web search
- Never claim to have found information if the observation doesn't explicitly contain it

Format your response EXACTLY as:
Thought: [your reasoning about what to do next, including verification of previous observation if applicable]
Action: [tool_name OR "FINISH"]
Action Input: [query for the tool, or "N/A" if FINISH]

Example of good thought process:
"The previous observation from paper_X did not contain relevant information about [topic]. 
I should try searching paper_Y or use web search to find current information."

{history_text}

Now, what is your next step?"""
        
        return prompt
    
    async def _parse_llm_response(self, response: str) -> Tuple[str, str, str]:
        """
        Parse LLM response into thought, action, and action_input.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Tuple of (thought, action, action_input)
        """
        lines = response.strip().split('\n')
        thought = ""
        action = ""
        action_input = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith("Thought:"):
                thought = line.replace("Thought:", "").strip()
            elif line.startswith("Action:"):
                action = line.replace("Action:", "").strip()
            elif line.startswith("Action Input:"):
                action_input = line.replace("Action Input:", "").strip()
        
        return thought, action, action_input
    
    async def _execute_tool(self, tool_name: str, query: str) -> str:
        """
        Execute a tool and return the observation.
        
        Args:
            tool_name: Name of the tool to execute
            query: Query to pass to the tool
            
        Returns:
            Tool execution result
        """
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"
        
        try:
            tool = self.tools[tool_name]
            result = await tool.acall(query)
            return str(result)
        except Exception as e:
            return f"Error executing tool: {str(e)}"
    
    async def run(self, question: str) -> Dict[str, Any]:
        """
        Run the ReAct reasoning loop.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary containing:
                - answer: Final answer
                - reasoning_steps: List of ReActStep objects
                - total_iterations: Number of iterations used
        """
        history: List[ReActStep] = []
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"ReAct Agent Starting")
            print(f"Question: {question}")
            print(f"{'='*60}\n")
        
        for iteration in range(1, self.max_iterations + 1):
            # Generate reasoning prompt
            prompt = self._create_react_prompt(question, history)
            
            # Get LLM response
            response = await self.llm.acomplete(prompt)
            response_text = response.text
            
            # Parse response
            thought, action, action_input = await self._parse_llm_response(response_text)
            
            if self.verbose:
                print(f"Iteration {iteration}:")
                print(f"  Thought: {thought}")
                print(f"  Action: {action}")
                print(f"  Action Input: {action_input}")
            
            # Check if we should finish
            if action.upper() == "FINISH" or "FINISH" in action.upper():
                step = ReActStep(
                    iteration=iteration,
                    thought=thought,
                    action_type=ActionType.FINISH,
                    action="FINISH",
                    action_input=action_input
                )
                history.append(step)
                
                if self.verbose:
                    print(f"  → Agent decided to FINISH\n")
                
                break
            
            # Execute the tool
            observation = await self._execute_tool(action, action_input)
            
            if self.verbose:
                print(f"  Observation: {observation[:200]}...")
                print()
            
            # Record step
            step = ReActStep(
                iteration=iteration,
                thought=thought,
                action_type=ActionType.TOOL_USE,
                action=action,
                action_input={"query": action_input},
                observation=observation
            )
            history.append(step)
        
        # Generate final answer
        final_answer = await self._generate_final_answer(question, history)
        
        if self.verbose:
            print(f"{'='*60}")
            print(f"Final Answer Generated")
            print(f"{'='*60}\n")
        
        return {
            "answer": final_answer,
            "reasoning_steps": [step.to_dict() for step in history],
            "total_iterations": len(history)
        }
    
    async def _generate_final_answer(
        self,
        question: str,
        history: List[ReActStep]
    ) -> str:
        """
        Generate final answer based on reasoning history.
        
        Args:
            question: Original question
            history: List of reasoning steps
            
        Returns:
            Final synthesized answer
        """
        # Check if web search was actually used
        web_search_used = any(
            step.action in ['web_search', 'scrape_webpage'] 
            for step in history 
            if step.action
        )
        
        # Check if all observations were "No relevant information"
        all_no_info = all(
            'No relevant information' in str(step.observation)
            for step in history 
            if step.observation
        )
        
        # Build context from observations
        context = []
        for step in history:
            if step.observation:
                context.append(f"From {step.action}: {step.observation}")
        
        context_text = "\n\n".join(context)
        
        # Build prompt with appropriate instructions
        prompt = f"""Based on the following information gathered through multiple steps, 
provide a comprehensive answer to the question.

Question: {question}

Information gathered:
{context_text}

IMPORTANT REQUIREMENTS:
- Provide a clear, well-structured answer that synthesizes all the information above
- You MUST cite the specific source (tool/paper name) for each piece of information
- Use format: "According to [source_name], ..." or "From [tool_name]: ..."
- DO NOT add any information that wasn't in the observations above
- If observations were empty or irrelevant, state that information was not found
"""
        
        # Add specific instructions based on what happened
        if all_no_info and not web_search_used:
            prompt += "\n\nIMPORTANT: All paper tools returned 'No relevant information found'. State clearly that the provided papers/documents do not contain information about this topic. DO NOT mention web search since it was not used."
        elif web_search_used:
            prompt += "\n\nNote: Web search was used. Clearly indicate which information came from web search vs papers."
        
        prompt += "\n\nAnswer:"
        
        response = await self.llm.acomplete(prompt)
        return response.text


class ReActResearchAssistant:
    """
    Research Assistant using ReAct agent for advanced reasoning.
    
    This is a wrapper that integrates ReAct agent with the existing
    ResearchAssistant infrastructure.
    """
    
    def __init__(
        self,
        base_assistant,
        max_iterations: int = 5,
        verbose: bool = True
    ):
        """
        Initialize ReAct Research Assistant.
        
        Args:
            base_assistant: Existing ResearchAssistant instance
            max_iterations: Max reasoning steps (default: 5)
            verbose: Whether to print reasoning (default: True)
        """
        self.base_assistant = base_assistant
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.react_agent = None
    
    def setup(self):
        """Setup the ReAct agent with tools from base assistant"""
        # Get all tools from base assistant
        all_tools = []
        for tools in self.base_assistant.paper_to_tools_dict.values():
            all_tools.extend(tools)
        
        # Add web search tools if available
        if hasattr(self.base_assistant, 'web_tools'):
            all_tools.extend(self.base_assistant.web_tools)
        
        # Create ReAct agent
        self.react_agent = ReActAgent(
            tools=all_tools,
            llm=self.base_assistant.llm,
            max_iterations=self.max_iterations,
            verbose=self.verbose
        )
        
        print(f"ReAct agent initialized with {len(all_tools)} tools")
    
    async def query(self, question: str) -> Dict[str, Any]:
        """
        Query using ReAct reasoning.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer and reasoning trace
        """
        if self.react_agent is None:
            raise ValueError("ReAct agent not initialized. Call setup() first.")
        
        return await self.react_agent.run(question)
    
    async def query_simple(self, question: str) -> str:
        """
        Query and return just the answer (for compatibility).
        
        Args:
            question: User's question
            
        Returns:
            Just the answer text
        """
        result = await self.query(question)
        return result["answer"]


# Example usage
async def demo_react():
    """Demo of ReAct agent"""
    from research_agent import ResearchAssistant
    
    # Create base assistant
    papers = ["./datasets/AutonomousDataAgents.pdf"]
    base_assistant = ResearchAssistant(papers)
    base_assistant.setup()
    
    # Wrap with ReAct
    react_assistant = ReActResearchAssistant(
        base_assistant=base_assistant,
        max_iterations=5,
        verbose=True
    )
    react_assistant.setup()
    
    # Test query
    question = "What are the main contributions of the paper and how do they compare to current industry practices?"
    result = await react_assistant.query(question)
    
    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nTotal iterations: {result['total_iterations']}")
    print(f"\nReasoning steps: {len(result['reasoning_steps'])}")


if __name__ == "__main__":
    asyncio.run(demo_react())
