"""
AI Narrative Engine

Leverages DeepSeek API for advanced narrative generation and insight extraction
for the Flash DNA Analysis System. This engine transforms quantitative metrics
into qualitative insights that provide unique value to investors.
"""

import logging
import json
import os
import re
import time
from typing import Dict, List, Any, Optional, Tuple, Union
import requests
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logger = logging.getLogger("ai_narrative_engine")
logger.setLevel(logging.INFO)

# API Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1")

# If API key not found in environment, check config file
if not DEEPSEEK_API_KEY:
    try:
        from config import DEEPSEEK_API_KEY as CONFIG_KEY
        DEEPSEEK_API_KEY = CONFIG_KEY
    except (ImportError, AttributeError):
        logger.warning("DeepSeek API key not found in environment or config")

class AINarrativeEngine:
    """
    Main engine for generating AI-powered narrative insights and analyses.
    Serves as the central integration point for all DeepSeek-powered content.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the narrative engine with API credentials."""
        self.api_key = api_key or DEEPSEEK_API_KEY
        self.model = "deepseek-chat"
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        
        if not self.api_key:
            logger.warning("No DeepSeek API key provided. Narrative engine will use fallback methods.")
    
    def _make_api_request(self, prompt: str, temperature: float = 0.3, 
                          max_tokens: int = 1500) -> Dict[str, Any]:
        """
        Make a request to the DeepSeek API with retries and error handling.
        
        Args:
            prompt: The prompt to send to the API
            temperature: Controls randomness (0.0-1.0)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Dict containing the API response
        """
        if not self.api_key:
            return {"error": "No API key provided", "text": self._generate_fallback_response(prompt)}
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{DEEPSEEK_API_URL}/chat/completions", 
                    headers=headers, 
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limit
                    logger.warning(f"Rate limit hit, waiting before retry ({attempt+1}/{self.max_retries})")
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"API request failed with status {response.status_code}: {response.text}")
                    break
                    
            except Exception as e:
                logger.error(f"Error making API request: {str(e)}")
                time.sleep(self.retry_delay)
        
        # If we got here, all retries failed
        return {"error": "Failed to get response from DeepSeek API", 
                "text": self._generate_fallback_response(prompt)}
    
    def _generate_fallback_response(self, prompt: str) -> str:
        """
        Generate a fallback response when the API is unavailable.
        This ensures reports still contain useful content even if the API fails.
        
        Args:
            prompt: The original prompt
            
        Returns:
            A reasonable fallback response
        """
        # Extract the request type from the prompt
        analysis_type = "analysis"
        if "competitive insights" in prompt.lower():
            analysis_type = "competitive insights"
        elif "market trends" in prompt.lower():
            analysis_type = "market trends analysis"
        elif "startup risks" in prompt.lower():
            analysis_type = "risk assessment"
        elif "recommendation" in prompt.lower():
            analysis_type = "recommendations"
            
        return (f"Based on the available data, we can provide a preliminary {analysis_type}. "
                f"A more detailed analysis would require additional processing. "
                f"The current metrics suggest this startup has both promising aspects "
                f"and areas that need attention. We recommend further analysis of "
                f"specific metrics to draw more definitive conclusions.")
    
    def _extract_content(self, response: Dict[str, Any]) -> str:
        """Extract the generated text from the API response."""
        if "error" in response:
            return response.get("text", "Error generating content")
            
        try:
            return response.get("choices", [{}])[0].get("message", {}).get("content", "")
        except (IndexError, KeyError, TypeError):
            logger.error(f"Unexpected API response format: {response}")
            return "Error extracting content from API response"
    
    def generate_executive_summary(self, doc: Dict[str, Any]) -> str:
        """
        Generate an AI-powered executive summary of the startup analysis.
        
        Args:
            doc: The document containing all startup analysis data
            
        Returns:
            A narrative executive summary
        """
        prompt = self._create_executive_summary_prompt(doc)
        response = self._make_api_request(prompt, temperature=0.4)
        return self._extract_content(response)
    
    def _create_executive_summary_prompt(self, doc: Dict[str, Any]) -> str:
        """Create the prompt for generating an executive summary."""
        # Extract key metrics
        name = doc.get("name", "the startup")
        camp_score = doc.get("camp_score", 0)
        capital_score = doc.get("capital_score", 0)
        market_score = doc.get("market_score", 0)
        advantage_score = doc.get("advantage_score", 0)
        people_score = doc.get("people_score", 0)
        success_prob = doc.get("success_prob", 0)
        runway = doc.get("runway_months", 0)
        
        # Extract patterns for strengths/weaknesses
        strengths = []
        weaknesses = []
        for pattern in doc.get("patterns_matched", []):
            if isinstance(pattern, dict):
                if pattern.get("is_positive", False):
                    strengths.append(pattern.get("name", ""))
                else:
                    weaknesses.append(pattern.get("name", ""))
        
        strengths_text = ", ".join(strengths[:3]) if strengths else "No clear strengths identified"
        weaknesses_text = ", ".join(weaknesses[:3]) if weaknesses else "No clear weaknesses identified"
        
        return f"""
        You are an expert startup analyst writing an executive summary for investors.
        Generate a concise, insightful executive summary for {name}.
        
        Key metrics:
        - CAMP Score: {camp_score}/100
        - Capital Score: {capital_score}/100
        - Market Score: {market_score}/100
        - Advantage Score: {advantage_score}/100
        - People Score: {people_score}/100
        - Success Probability: {success_prob}%
        - Runway: {runway} months
        
        Strengths: {strengths_text}
        Weaknesses: {weaknesses_text}
        
        The executive summary should:
        1. Open with a concise overview of the startup and its potential
        2. Highlight the most compelling aspects of the investment opportunity
        3. Identify key risks and how they might be mitigated
        4. Conclude with an overall assessment of the startup's prospects
        
        Write in a clear, professional tone that would appeal to sophisticated investors.
        DO NOT mention that this was AI-generated.
        DO NOT mention DeepSeek or any AI model.
        DO NOT use phrases like "based on the data provided" or "according to the metrics".
        Write as if you are a human analyst who has deeply studied this startup.
        Be concise but insightful, around 3-4 paragraphs total.
        """
    
    def generate_competitive_analysis(self, doc: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate an AI-powered competitive analysis of the startup.
        
        Args:
            doc: The document containing all startup analysis data
            
        Returns:
            Dict with different sections of competitive analysis
        """
        # Extract competitors
        competitors = doc.get("competitors", [])
        competitor_names = [c.get("name", f"Competitor {i+1}") 
                           for i, c in enumerate(competitors) if isinstance(c, dict)]
        
        prompt = self._create_competitive_analysis_prompt(doc, competitor_names)
        response = self._make_api_request(prompt, temperature=0.3, max_tokens=2000)
        content = self._extract_content(response)
        
        # Parse the content into sections
        sections = {}
        current_section = "overview"
        sections[current_section] = ""
        
        for line in content.split('\n'):
            if line.startswith('##'):
                # New section header
                current_section = line.strip('#').strip().lower().replace(' ', '_')
                sections[current_section] = ""
            else:
                sections[current_section] += line + "\n"
        
        # Clean up the sections
        for key in sections:
            sections[key] = sections[key].strip()
            
        return sections
    
    def _create_competitive_analysis_prompt(self, doc: Dict[str, Any], 
                                            competitor_names: List[str]) -> str:
        """Create the prompt for generating a competitive analysis."""
        name = doc.get("name", "the startup")
        sector = doc.get("sector", "technology")
        stage = doc.get("stage", "early stage")
        
        competitors_text = ", ".join(competitor_names) if competitor_names else "unknown competitors"
        
        moat_analysis = doc.get("moat_analysis", {})
        moat_score = moat_analysis.get("overall_score", 0) if isinstance(moat_analysis, dict) else 0
        
        competitive_positioning = doc.get("competitive_positioning", {})
        position = competitive_positioning.get("position", "challenger") if isinstance(competitive_positioning, dict) else "challenger"
        
        return f"""
        You are an expert competitive analyst for venture capital investments.
        Generate a detailed competitive analysis for {name}, a {stage} {sector} startup.
        
        The startup competes with: {competitors_text}
        Current competitive position: {position}
        Moat strength score: {moat_score}/100
        
        Please analyze the competitive landscape and include these sections:
        
        ## Market Position
        Analyze the startup's current position in the market and how it compares to competitors.
        
        ## Competitive Advantages
        Identify the startup's key competitive advantages and how defensible they are.
        
        ## Threats and Challenges
        Analyze the main competitive threats facing the startup.
        
        ## Differentiation Strategy
        Recommend how the startup can further differentiate from competitors.
        
        Write in a clear, professional tone. Be specific and actionable with your analysis.
        DO NOT mention that this was AI-generated.
        DO NOT use generic statements like "based on the information provided."
        Write as if you are a human analyst who has deeply studied this startup and its industry.
        """
    
    def generate_strategic_recommendations(self, doc: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate strategic recommendations based on the startup analysis.
        
        Args:
            doc: The document containing all startup analysis data
            
        Returns:
            List of recommendation dictionaries with title, description, and priority
        """
        prompt = self._create_recommendations_prompt(doc)
        response = self._make_api_request(prompt, temperature=0.4, max_tokens=2000)
        content = self._extract_content(response)
        
        # Parse the recommendations
        recommendations = []
        current_rec = {}
        
        for line in content.split('\n'):
            if line.startswith('##'):
                # Save previous recommendation if it exists
                if current_rec and 'title' in current_rec:
                    recommendations.append(current_rec)
                
                # Start new recommendation
                current_rec = {
                    'title': line.strip('#').strip(),
                    'description': '',
                    'priority': 'Medium'  # Default priority
                }
            elif line.startswith('Priority:'):
                priority = line.replace('Priority:', '').strip()
                current_rec['priority'] = priority
            elif current_rec and 'title' in current_rec:
                current_rec['description'] += line + "\n"
        
        # Add the last recommendation
        if current_rec and 'title' in current_rec:
            recommendations.append(current_rec)
            
        # Clean up descriptions
        for rec in recommendations:
            rec['description'] = rec['description'].strip()
            
        return recommendations
    
    def _create_recommendations_prompt(self, doc: Dict[str, Any]) -> str:
        """Create the prompt for generating strategic recommendations."""
        name = doc.get("name", "the startup")
        camp_score = doc.get("camp_score", 0)
        capital_score = doc.get("capital_score", 0)
        market_score = doc.get("market_score", 0)
        advantage_score = doc.get("advantage_score", 0)
        people_score = doc.get("people_score", 0)
        
        # Find lowest scoring area for focused recommendations
        scores = {
            "Capital Efficiency": capital_score,
            "Market Dynamics": market_score,
            "Advantage Moat": advantage_score,
            "People & Performance": people_score
        }
        weakest_area = min(scores.items(), key=lambda x: x[1])
        
        return f"""
        You are an expert startup advisor generating strategic recommendations for {name}.
        
        CAMP Framework Scores:
        - Overall: {camp_score}/100
        - Capital Efficiency: {capital_score}/100
        - Market Dynamics: {market_score}/100
        - Advantage Moat: {advantage_score}/100
        - People & Performance: {people_score}/100
        
        The startup's weakest area is: {weakest_area[0]} ({weakest_area[1]}/100)
        
        Generate 3-5 specific, actionable strategic recommendations.
        Each recommendation should:
        
        1. Focus on addressing key challenges or capitalizing on opportunities
        2. Be specific to this startup's situation, not generic advice
        3. Include clear next steps or implementation guidance
        4. Have a priority level (High, Medium, or Low)
        
        Format each recommendation as:
        
        ## [Recommendation Title]
        Priority: [Priority Level]
        
        [Detailed description with specific actions and expected outcomes]
        
        Make at least 2 recommendations that address the weakest area.
        DO NOT mention that this was AI-generated.
        DO NOT use clichés or generic startup advice.
        Write as if you are a veteran startup advisor with deep domain expertise.
        """
    
    def generate_market_trends_analysis(self, doc: Dict[str, Any]) -> str:
        """
        Generate an analysis of relevant market trends for the startup.
        
        Args:
            doc: The document containing all startup analysis data
            
        Returns:
            A narrative analysis of market trends
        """
        prompt = self._create_market_trends_prompt(doc)
        response = self._make_api_request(prompt)
        return self._extract_content(response)
    
    def _create_market_trends_prompt(self, doc: Dict[str, Any]) -> str:
        """Create the prompt for generating market trends analysis."""
        sector = doc.get("sector", "technology")
        market_size = doc.get("market_size", 0)
        market_growth_rate = doc.get("market_growth_rate", 0)
        
        market_trends = doc.get("market_trends", {})
        existing_trends = []
        if isinstance(market_trends, dict) and "trends" in market_trends:
            existing_trends = market_trends.get("trends", [])
        
        trends_text = ", ".join(existing_trends) if existing_trends else "No specific trends identified"
        
        return f"""
        You are an expert market analyst specializing in the {sector} sector.
        Generate an insightful analysis of current and emerging market trends relevant to a startup in this space.
        
        Market data:
        - Market size: ${market_size:,.0f}
        - Market growth rate: {market_growth_rate}%/year
        - Identified trends: {trends_text}
        
        Your analysis should:
        1. Identify 3-5 key market trends that are most relevant to this sector
        2. Explain how each trend creates opportunities or threats
        3. Provide specific insights on how a startup could capitalize on these trends
        4. Include a forward-looking perspective on how the market might evolve in the next 2-3 years
        
        Write in a clear, data-driven, and insightful tone that demonstrates deep market expertise.
        DO NOT mention that this was AI-generated.
        DO NOT use phrases like "based on the data provided" or "according to the information."
        Write as if you are a human analyst who has been tracking this market for years.
        """
    
    def analyze_intangible_factors(self, pitch_text: str) -> Dict[str, Any]:
        """
        Perform deep analysis of intangible factors from pitch deck text.
        This goes beyond basic sentiment to extract nuanced signals about team,
        vision, and execution potential.
        
        Args:
            pitch_text: The extracted text from the pitch deck
            
        Returns:
            Dict containing intangible factor analysis
        """
        if not pitch_text or len(pitch_text.strip()) < 100:
            return {
                "intangible_score": 50.0,
                "confidence": "low",
                "team_conviction": 50.0,
                "vision_clarity": 50.0,
                "narrative_strength": 50.0,
                "execution_signals": 50.0,
                "overall_assessment": "Insufficient pitch data for intangible analysis"
            }
        
        prompt = self._create_intangible_analysis_prompt(pitch_text)
        response = self._make_api_request(prompt, temperature=0.1)
        content = self._extract_content(response)
        
        # Parse the JSON response
        try:
            # Find JSON within the response
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            
            # If not properly formatted, try to find JSON block directly
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
                
            # Fallback to default values with the assessment from the response
            return {
                "intangible_score": 50.0,
                "confidence": "low",
                "team_conviction": 50.0,
                "vision_clarity": 50.0,
                "narrative_strength": 50.0,
                "execution_signals": 50.0,
                "overall_assessment": content.strip()
            }
            
        except json.JSONDecodeError:
            logger.error(f"Failed to parse intangible analysis JSON: {content}")
            return {
                "intangible_score": 50.0,
                "confidence": "low",
                "team_conviction": 50.0,
                "vision_clarity": 50.0,
                "narrative_strength": 50.0,
                "execution_signals": 50.0,
                "overall_assessment": "Error parsing intangible analysis"
            }
    
    def _create_intangible_analysis_prompt(self, pitch_text: str) -> str:
        """Create the prompt for analyzing intangible factors from pitch text."""
        # Truncate pitch text if too long
        max_length = 8000
        if len(pitch_text) > max_length:
            pitch_text = pitch_text[:max_length] + "... [truncated]"
            
        return f"""
        You are an expert venture capital analyst who specializes in evaluating the intangible factors
        of startups based on their pitch materials. Analyze the following pitch deck text to extract
        signals about the team, vision, and execution potential.
        
        PITCH DECK TEXT:
        {pitch_text}
        
        Perform a comprehensive analysis of the intangible factors in this pitch. 
        Evaluate the following aspects:
        
        1. Team conviction - How passionate and committed does the team appear?
        2. Vision clarity - How clear and compelling is the vision?
        3. Narrative strength - How well-constructed and persuasive is the narrative?
        4. Execution signals - Are there indications of strong execution capability?
        
        Return your analysis as a JSON object with the following structure:
        
        ```json
        {{
            "intangible_score": <0-100 score>,
            "confidence": "<low|medium|high>",
            "team_conviction": <0-100 score>,
            "vision_clarity": <0-100 score>,
            "narrative_strength": <0-100 score>,
            "execution_signals": <0-100 score>,
            "key_positives": ["<positive signal 1>", "<positive signal 2>", ...],
            "key_concerns": ["<concern 1>", "<concern 2>", ...],
            "overall_assessment": "<1-2 sentence assessment>"
        }}
        ```
        
        Base your scores on patterns you've seen in successful vs. unsuccessful startups.
        Be honest in your assessment - neither overly optimistic nor overly critical.
        """
    
    def generate_report_content(self, doc: Dict[str, Any], 
                                sections: List[str] = None) -> Dict[str, str]:
        """
        Generate multiple report sections in parallel for faster processing.
        
        Args:
            doc: The document containing all startup analysis data
            sections: List of section names to generate
            
        Returns:
            Dict mapping section names to generated content
        """
        if sections is None:
            sections = ["executive_summary", "competitive_analysis", "recommendations", "market_trends"]
            
        content = {}
        
        # Generate content in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            
            if "executive_summary" in sections:
                futures["executive_summary"] = executor.submit(self.generate_executive_summary, doc)
                
            if "competitive_analysis" in sections:
                futures["competitive_analysis"] = executor.submit(self.generate_competitive_analysis, doc)
                
            if "recommendations" in sections:
                futures["recommendations"] = executor.submit(self.generate_strategic_recommendations, doc)
                
            if "market_trends" in sections:
                futures["market_trends"] = executor.submit(self.generate_market_trends_analysis, doc)
            
            # Collect results
            for section, future in futures.items():
                try:
                    content[section] = future.result()
                except Exception as e:
                    logger.error(f"Error generating {section}: {str(e)}")
                    content[section] = f"Error generating {section}: {str(e)}"
        
        return content
    
    def compute_intangible_score(self, doc: Dict[str, Any]) -> float:
        """
        Enhanced replacement for the existing intangible_api.compute_intangible_llm function.
        Analyzes pitch text to derive a deep intangible score.
        
        Args:
            doc: The document containing pitch deck text
            
        Returns:
            Intangible score (0-100)
        """
        pitch_text = doc.get("pitch_deck_text", "")
        if not pitch_text or len(pitch_text.strip()) < 100:
            return 50.0  # Default score for insufficient data
            
        analysis = self.analyze_intangible_factors(pitch_text)
        return analysis.get("intangible_score", 50.0)


def initialize_narrative_engine() -> AINarrativeEngine:
    """Initialize and return the AI narrative engine as a global singleton."""
    global _narrative_engine
    if '_narrative_engine' not in globals():
        _narrative_engine = AINarrativeEngine()
    return _narrative_engine


# Compatibility function to replace existing intangible_api.compute_intangible_llm
def compute_intangible_llm(doc: Dict[str, Any]) -> float:
    """
    Enhanced replacement for the existing compute_intangible_llm function.
    
    Args:
        doc: The document containing pitch deck text
        
    Returns:
        Intangible score (0-100)
    """
    engine = initialize_narrative_engine()
    return engine.compute_intangible_score(doc) 