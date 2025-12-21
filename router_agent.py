#!/usr/bin/env python3
"""
Router Agent - Autonomous query routing using LLM reasoning
"""

from typing import Dict, List
import json
import asyncio
import httpx
from llama_index.llms.openai import OpenAI


class RouterAgent:
    """
    Intelligent router that autonomously decides which specialized 
    agent should handle a user query using LLM reasoning.
    """
    
    def __init__(self, openai_api_key: str):
        # Create BOTH sync and async HTTP clients with SSL verification disabled
        # This matches the configuration in research_agent.py
        http_client = httpx.Client(verify=False)
        async_http_client = httpx.AsyncClient(verify=False)
        
        self.llm = OpenAI(
            model="gpt-3.5-turbo",
            api_key=openai_api_key,
            temperature=0.1,  # Low temp for consistent routing
            http_client=http_client,
            async_http_client=async_http_client
        )
        
        # Define available agents and their capabilities
        self.agents = {
            "blog_writer": {
                "description": "Generates blog posts, articles, and written content on topics",
                "capabilities": [
                    "Create comprehensive blog posts",
                    "Generate articles with proper structure",
                    "Write content based on research papers",
                    "Produce engaging written content"
                ],
                "keywords": ["write", "blog", "article", "post", "create", "generate", "draft", "compose"],
                "examples": [
                    "Write a blog on computer vision",
                    "Create an article about transformers",
                    "Generate a post on RAG systems"
                ]
            },
            "qa_agent": {
                "description": "Answers questions using document analysis and multi-step reasoning",
                "capabilities": [
                    "Answer complex questions",
                    "Analyze and compare information across documents",
                    "Provide detailed explanations",
                    "Synthesize information from multiple sources"
                ],
                "keywords": ["what", "how", "why", "explain", "compare", "analyze", "describe"],
                "examples": [
                    "What are autonomous agents?",
                    "How does RAG work?",
                    "Compare X and Y across papers",
                    "Explain the key concepts"
                ]
            }
        }
    
    async def route(self, user_query: str) -> Dict:
        """
        Autonomously route query to appropriate agent using LLM reasoning.
        
        Args:
            user_query: The user's input query
            
        Returns:
            {
                "agent": "agent_name",
                "reasoning": "explanation",
                "confidence": 0.0-1.0,
                "alternative": "backup_agent" (optional)
            }
        """
        
        # Build routing prompt with agent information
        agents_info = self._format_agents_info()
        
        prompt = f"""You are an intelligent Router Agent. Your job is to analyze the user's query and decide which specialized agent should handle it.

Available Agents:
{agents_info}

User Query: "{user_query}"

Analyze the query step-by-step:
1. What is the user's primary intent? (content creation vs information seeking)
2. What type of output do they expect? (blog/article vs answer/explanation)
3. Which agent is best suited for this task?
4. How confident are you in this decision? (0.0 to 1.0)
5. What would be a good backup agent if the primary choice doesn't work?

Think carefully about the user's intent. Key indicators:

BLOG WRITER (Content Creation):
- Explicit blog/article requests: "write a blog", "create an article", "draft a post"
- Long-form content: "comprehensive post", "detailed article"
- Publishing intent: "for my website", "to publish"

QA AGENT (Information Seeking):
- Questions: "what", "how", "why", "explain", "describe"
- Summary requests: "summarize", "brief summary", "short summary", "overview"
- Comparison requests: "compare", "analyze", "contrast"
- Explanations: "tell me about", "explain"

CRITICAL: "write a summary" or "write a short summary" = QA agent (NOT blog writer!)
The word "write" alone doesn't mean blog - consider the full context.

Respond ONLY with valid JSON in this exact format (no other text):
{{
    "agent": "blog_writer" or "qa_agent",
    "reasoning": "detailed explanation of why this agent was chosen based on the query analysis",
    "confidence": 0.95,
    "alternative": "backup_agent_name"
}}"""

        try:
            response = await self.llm.acomplete(prompt)
            
            # Add rate limiting - sleep 0.2 seconds after API call
            await asyncio.sleep(0.2)
            
            response_text = response.text.strip()
            
            # Try to extract JSON if there's extra text
            if not response_text.startswith('{'):
                # Find JSON in response
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start != -1 and end > start:
                    response_text = response_text[start:end]
            
            decision = json.loads(response_text)
            
            # Validate decision
            if decision["agent"] not in self.agents:
                raise ValueError(f"Invalid agent: {decision['agent']}")
            
            # Ensure confidence is between 0 and 1
            decision["confidence"] = max(0.0, min(1.0, float(decision["confidence"])))
            
            return decision
            
        except Exception as e:
            # Fallback to simple heuristic if LLM fails
            print(f"⚠️  Router LLM failed: {e}, using fallback heuristic")
            return self._fallback_route(user_query)
    
    def _format_agents_info(self) -> str:
        """Format agent information for the prompt."""
        info_lines = []
        for agent_name, agent_info in self.agents.items():
            info_lines.append(f"\n{agent_name.upper()}:")
            info_lines.append(f"  Description: {agent_info['description']}")
            info_lines.append(f"  Capabilities:")
            for cap in agent_info['capabilities']:
                info_lines.append(f"    - {cap}")
            info_lines.append(f"  Common keywords: {', '.join(agent_info['keywords'])}")
            info_lines.append(f"  Example queries:")
            for ex in agent_info['examples']:
                info_lines.append(f"    - \"{ex}\"")
        
        return "\n".join(info_lines)
    
    def _fallback_route(self, query: str) -> Dict:
        """
        Simple keyword-based fallback routing when LLM fails.
        Uses priority-based matching for better accuracy.
        """
        query_lower = query.lower()
        
        # Priority 1: Check for summary/explanation keywords (highest priority)
        summary_keywords = ["summary", "summarize", "brief", "overview", "explain", "describe", "tell me about"]
        if any(kw in query_lower for kw in summary_keywords):
            return {
                "agent": "qa_agent",
                "reasoning": "Fallback: Detected summary/explanation request (high priority)",
                "confidence": 0.75,
                "alternative": "blog_writer"
            }
        
        # Priority 2: Check for explicit blog keywords
        explicit_blog_keywords = ["blog", "article", "post"]
        if any(kw in query_lower for kw in explicit_blog_keywords):
            return {
                "agent": "blog_writer",
                "reasoning": "Fallback: Detected explicit blog/article keywords",
                "confidence": 0.7,
                "alternative": "qa_agent"
            }
        
        # Priority 3: Check for question words
        question_keywords = ["what", "how", "why", "when", "where", "who", "compare", "analyze"]
        if any(kw in query_lower for kw in question_keywords):
            return {
                "agent": "qa_agent",
                "reasoning": "Fallback: Detected question keywords",
                "confidence": 0.7,
                "alternative": "blog_writer"
            }
        
        # Priority 4: Default to Q&A for ambiguous cases
        return {
            "agent": "qa_agent",
            "reasoning": "Fallback: Default to Q&A for informational queries",
            "confidence": 0.6,
            "alternative": "blog_writer"
        }
    
    def explain_routing(self, decision: Dict, user_query: str) -> str:
        """
        Generate human-readable explanation of routing decision.
        """
        agent_name = decision["agent"]
        agent_info = self.agents[agent_name]
        
        explanation = f"""
🎯 ROUTER AGENT DECISION
{'=' * 60}
Query: "{user_query}"

Selected Agent: {agent_name.upper().replace('_', ' ')}
Description: {agent_info['description']}
Confidence: {decision['confidence']:.0%}

Reasoning:
{decision['reasoning']}

Backup Agent: {decision.get('alternative', 'None').upper().replace('_', ' ')}
{'=' * 60}
"""
        return explanation
