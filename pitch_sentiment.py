import os
import logging
import re
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import random
import time

# Try to load nltk for advanced sentiment analysis
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Try to load environment variables
try:
    from dotenv import load_dotenv
    for env_file in ['.env', '.env.local', '/app/.env']:
        if os.path.exists(env_file):
            load_dotenv(env_file)
            break
except ImportError:
    pass  # dotenv not available

# Configure logging
logger = logging.getLogger("pitch_sentiment")
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

# Constants
MAX_TEXT_LENGTH = 10000
MIN_CATEGORY_CONFIDENCE = 0.6
DEFAULT_CONFIDENCE = 0.75

# Default NLP models
DEFAULT_SENTIMENT_MODEL = "vader"

# Pitch categories as enum for better type safety
class PitchCategory(Enum):
    TEAM = "team"
    PRODUCT = "product"
    MARKET = "market"
    BUSINESS_MODEL = "business_model"
    FINANCIALS = "financials"
    COMPETITION = "competition"
    VISION = "vision"
    TRACTION = "traction"
    GENERAL = "general"

@dataclass
class CategorySentiment:
    """Sentiment analysis results for a specific pitch category."""
    category: str
    score: float  # -1 to 1 scale
    confidence: float  # 0 to 1 scale
    text_samples: List[str]  # Key text snippets that influenced the score

@dataclass
class SentimentResult:
    """Overall sentiment analysis results for a pitch deck."""
    sentiment_score: float  # -1 to 1 scale
    confidence: float  # 0 to 1 scale
    category_sentiments: Dict[str, CategorySentiment]
    key_phrases: List[Dict[str, Any]]  # Key phrases with sentiment information
    raw_scores: Dict[str, float]  # Raw scoring data
    analysis_method: str  # Method used for analysis

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Convert enums to their values if present in category_sentiments
        if result.get('category_sentiments'):
            result['category_sentiments'] = {
                k.value if hasattr(k, 'value') else k: v 
                for k, v in result['category_sentiments'].items()
            }
        return result

class PitchAnalyzer:
    """
    Advanced analyzer for startup pitch decks that performs sentiment analysis
    and extracts key insights.
    
    This analyzer uses NLTK for NLP tasks and sentiment analysis.
    """
    
    def __init__(self, sentiment_model: str = DEFAULT_SENTIMENT_MODEL, 
                 load_nltk_resources: bool = True):
        """
        Initialize the pitch analyzer with specified models.
        
        Args:
            sentiment_model: Name of the sentiment model to use (default: vader)
            load_nltk_resources: Whether to download required NLTK resources
        """
        self.sentiment_model = sentiment_model
        self.initialized = False
        
        # Try to initialize the models
        try:
            self._initialize_nlp(load_nltk_resources)
            self.initialized = True
        except Exception as e:
            logger.warning(f"Could not initialize NLP models: {e}. Will use fallback methods.")
        
        # Category recognition patterns
        self.category_patterns = self._get_category_patterns()
        
    def _initialize_nlp(self, load_nltk_resources: bool) -> None:
        """Initialize NLP components."""
        # Initialize sentiment analyzer
        if self.sentiment_model == "vader" and NLTK_AVAILABLE:
            try:
                # Download VADER lexicon if needed and requested
                if load_nltk_resources:
                    try:
                        nltk.data.find('vader_lexicon.zip')
                    except LookupError:
                        nltk.download('vader_lexicon', quiet=True)
                        nltk.download('punkt', quiet=True)
                        nltk.download('averaged_perceptron_tagger', quiet=True)
                        nltk.download('maxent_ne_chunker', quiet=True)
                        nltk.download('words', quiet=True)
                
                self.sid = SentimentIntensityAnalyzer()
                logger.info("Initialized VADER sentiment analyzer")
            except Exception as e:
                logger.error(f"Failed to initialize VADER: {e}")
                self.sid = None
    
    def _get_category_patterns(self) -> Dict[PitchCategory, List[str]]:
        """Define regex patterns for identifying pitch categories."""
        return {
            PitchCategory.TEAM: [
                r'\bteam\b', r'\bfounder', r'\bco-founder', r'\bceo\b', r'\bcto\b', 
                r'\bexperience\b', r'(\bour\s+team\b)', r'(\bthe\s+team\b)',
                r'\bbackground\b', r'\bleadership\b', r'\bmanagement\b'
            ],
            PitchCategory.PRODUCT: [
                r'\bproduct\b', r'\bsolution\b', r'\btechnology\b', r'\bplatform\b',
                r'\bfeatures\b', r'\bservice\b', r'\binnovation\b', r'\bapp\b',
                r'\bsoftware\b', r'\bhardware\b', r'\bprototype\b'
            ],
            PitchCategory.MARKET: [
                r'\bmarket\b', r'\bindustry\b', r'\bsegment\b', r'\btam\b', 
                r'\bsam\b', r'\bsom\b', r'\bmarket\s+size\b', r'\bopportunity\b',
                r'\btrend\b', r'\bdemand\b', r'\bcustomer\b'
            ],
            PitchCategory.BUSINESS_MODEL: [
                r'\bbusiness\s+model\b', r'\brevenue\s+model\b', r'\bmonetization\b',
                r'\bpricing\b', r'\bsubscription\b', r'\btransaction\b', r'\bunit\s+economics\b',
                r'\bmargins\b', r'\bcac\b', r'\bltv\b', r'\bcost\s+structure\b'
            ],
            PitchCategory.FINANCIALS: [
                r'\bfinancial', r'\bprojection', r'\bforecast', r'\brevenue', 
                r'\bprofit', r'\bebitda\b', r'\bburn\s+rate\b', r'\bcash\s+flow\b',
                r'\binvestment\b', r'\bfunding\b', r'\bvaluation\b'
            ],
            PitchCategory.COMPETITION: [
                r'\bcompetit', r'\brival', r'\blandscape\b', r'\balternative', 
                r'\bversus\b', r'\bvs\.?\b', r'\bmarket\s+leader\b', r'\bincumbent\b',
                r'\bmoat\b', r'\bdifferent', r'\bunique\s+value'
            ],
            PitchCategory.VISION: [
                r'\bvision\b', r'\bmission\b', r'\bgoal\b', r'\bfuture\b', 
                r'\bstrategy\b', r'\bplan\b', r'\bgrowth\b', r'\bscale\b',
                r'\bexpansion\b', r'\blong-term\b', r'\bimpact\b'
            ],
            PitchCategory.TRACTION: [
                r'\btraction\b', r'\bgrowth\b', r'\bcustomer\b', r'\buser\b', 
                r'\bmetric\b', r'\bkpi\b', r'\bmrr\b', r'\barr\b', r'\bchurn\b',
                r'\bretention\b', r'\bacquisition\b'
            ],
            PitchCategory.GENERAL: [
                r'\bpitch\b', r'\bstartup\b', r'\bcompany\b', r'\bbusiness\b',
                r'\bventure\b', r'\benterprise\b', r'\borganization\b'
            ]
        }
    
    def analyze_sentiment(self, text: str) -> SentimentResult:
        """
        Analyze sentiment in the given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            SentimentResult object containing analysis results
        """
        # Preprocess text
        text = self._preprocess_text(text)
        
        # Use VADER if available
        if self.initialized and self.sentiment_model == "vader":
            return self._analyze_with_vader(text)
        
        # Fallback to basic analysis
        return self._analyze_with_fallback(text)
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
            
        # Truncate if too long
        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH]
        
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        return text
    
    def _analyze_with_vader(self, text: str) -> SentimentResult:
        """Analyze sentiment using VADER."""
        # Split into sentences
        sentences = sent_tokenize(text)
        
        # Get category-specific sentiments
        category_sentiments = self._analyze_categories(sentences)
        
        # Get overall sentiment
        compound_scores = []
        for sentence in sentences:
            scores = self.sid.polarity_scores(sentence)
            compound_scores.append(scores['compound'])
        
        # Calculate overall metrics
        overall_score = np.mean(compound_scores) if compound_scores else 0.0
        confidence = min(1.0, 0.5 + abs(overall_score))
        
        # Extract key phrases
        key_phrases = self._extract_key_phrases(sentences)
        
        return SentimentResult(
            sentiment_score=overall_score,
            confidence=confidence,
            category_sentiments=category_sentiments,
            key_phrases=key_phrases,
            raw_scores={'compound': overall_score},
            analysis_method="vader"
        )
    
    def _analyze_with_fallback(self, text: str) -> SentimentResult:
        """Basic sentiment analysis when VADER is not available."""
        # Split into sentences
        sentences = sent_tokenize(text)
        
        # Simple word lists for sentiment
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'best', 'success',
            'innovative', 'efficient', 'profit', 'growth', 'leading', 'opportunity',
            'successful', 'positive', 'perfect', 'better', 'strong', 'unique'
        }
        
        negative_words = {
            'bad', 'poor', 'terrible', 'awful', 'worst', 'failure', 'inefficient',
            'loss', 'decline', 'weak', 'problem', 'difficult', 'negative', 'hard',
            'impossible', 'wrong', 'fail', 'risk', 'costly'
        }
        
        # Analyze categories
        category_sentiments = self._analyze_categories_fallback(
            sentences, list(positive_words), list(negative_words)
        )
        
        # Calculate overall sentiment
        word_counts = {'pos': 0, 'neg': 0}
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            word_counts['pos'] += sum(1 for w in words if w in positive_words)
            word_counts['neg'] += sum(1 for w in words if w in negative_words)
        
        total = word_counts['pos'] + word_counts['neg']
        if total > 0:
            sentiment_score = (word_counts['pos'] - word_counts['neg']) / total
            confidence = min(1.0, 0.5 + (total / len(sentences)) * 0.1)
        else:
            sentiment_score = 0.0
            confidence = 0.5
        
        # Extract key phrases
        key_phrases = self._extract_key_phrases(sentences)
        
        return SentimentResult(
            sentiment_score=sentiment_score,
            confidence=confidence,
            category_sentiments=category_sentiments,
            key_phrases=key_phrases,
            raw_scores=word_counts,
            analysis_method="basic"
        )
    
    def _analyze_categories(self, sentences: List[str]) -> Dict[str, CategorySentiment]:
        """Analyze sentiment for each category using VADER."""
        results = {}
        
        for category, patterns in self.category_patterns.items():
            relevant_sentences = []
            scores = []
            
            for sentence in sentences:
                if any(re.search(pattern, sentence, re.I) for pattern in patterns):
                    relevant_sentences.append(sentence)
                    scores.append(self.sid.polarity_scores(sentence)['compound'])
            
            if relevant_sentences:
                avg_score = np.mean(scores)
                confidence = min(1.0, 0.5 + (len(relevant_sentences) / len(sentences)))
                
                results[category] = CategorySentiment(
                    category=category.value,
                    score=avg_score,
                    confidence=confidence,
                    text_samples=relevant_sentences[:3]  # Top 3 examples
                )
        
        return results
    
    def _analyze_categories_fallback(self, sentences: List[str], 
                                   positive_words: List[str],
                                   negative_words: List[str]) -> Dict[str, CategorySentiment]:
        """Basic category sentiment analysis without VADER."""
        results = {}
        
        for category, patterns in self.category_patterns.items():
            relevant_sentences = []
            scores = []
            
            for sentence in sentences:
                if any(re.search(pattern, sentence, re.I) for pattern in patterns):
                    relevant_sentences.append(sentence)
                    
                    # Simple word counting for sentiment
                    words = word_tokenize(sentence.lower())
                    pos_count = sum(1 for w in words if w in positive_words)
                    neg_count = sum(1 for w in words if w in negative_words)
                    total = pos_count + neg_count
                    
                    if total > 0:
                        score = (pos_count - neg_count) / total
                    else:
                        score = 0.0
                    
                    scores.append(score)
            
            if relevant_sentences:
                avg_score = np.mean(scores) if scores else 0.0
                confidence = min(1.0, 0.5 + (len(relevant_sentences) / len(sentences)))
                
                results[category] = CategorySentiment(
                    category=category.value,
                    score=avg_score,
                    confidence=confidence,
                    text_samples=relevant_sentences[:3]  # Top 3 examples
                )
        
        return results
    
    def _extract_key_phrases(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """Extract and analyze key phrases using NLTK."""
        key_phrases = []
        
        for sentence in sentences:
            # Tokenize and tag parts of speech
            tokens = word_tokenize(sentence)
            tagged = pos_tag(tokens)
            
            # Extract named entities
            entities = ne_chunk(tagged)
            
            # Process named entities
            current_entity = []
            current_type = None
            
            for chunk in entities:
                if hasattr(chunk, 'label'):
                    # Named entity found
                    entity_text = ' '.join(c[0] for c in chunk.leaves())
                    entity_type = chunk.label()
                    
                    # Get sentiment if VADER is available
                    if self.initialized and self.sentiment_model == "vader":
                        sentiment = self.sid.polarity_scores(entity_text)['compound']
                    else:
                        # Simple positive/negative word counting
                        words = word_tokenize(entity_text.lower())
                        pos_words = sum(1 for w in words if w in {'good', 'great', 'best'})
                        neg_words = sum(1 for w in words if w in {'bad', 'poor', 'worst'})
                        total = pos_words + neg_words
                        sentiment = (pos_words - neg_words) / total if total > 0 else 0.0
                    
                    key_phrases.append({
                        'text': entity_text,
                        'type': entity_type,
                        'sentiment': sentiment,
                        'sentence': sentence
                    })
        
        return key_phrases
    
    def _get_default_result(self) -> SentimentResult:
        """Return default sentiment result for error cases."""
        return SentimentResult(
            sentiment_score=0.0,
            confidence=0.0,
            category_sentiments={},
            key_phrases=[],
            raw_scores={},
            analysis_method="none"
        )
    
    def extract_metrics(self, text: str) -> Dict[str, Any]:
        """Extract numerical metrics and statistics from text."""
        metrics = {
            'numbers': [],
            'percentages': [],
            'currencies': [],
            'dates': []
        }
        
        # Number patterns
        number_pattern = r'\b\d+(?:,\d{3})*(?:\.\d+)?\b'
        percentage_pattern = r'\b\d+(?:\.\d+)?%\b'
        currency_pattern = r'(?:$|USD|€|£)\s*\d+(?:,\d{3})*(?:\.\d+)?\b'
        date_pattern = r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4})\b'
        
        # Extract metrics
        metrics['numbers'] = re.findall(number_pattern, text)
        metrics['percentages'] = re.findall(percentage_pattern, text)
        metrics['currencies'] = re.findall(currency_pattern, text)
        metrics['dates'] = re.findall(date_pattern, text)
        
        # Convert to appropriate types
        metrics['numbers'] = [float(n.replace(',', '')) for n in metrics['numbers']]
        metrics['percentages'] = [float(p.strip('%')) / 100 for p in metrics['percentages']]
        
        # Process currencies
        processed_currencies = []
        for c in metrics['currencies']:
            value = re.search(r'\d+(?:,\d{3})*(?:\.\d+)?', c).group()
            currency = c[0] if c[0] in {'$', '€', '£'} else 'USD'
            processed_currencies.append({
                'value': float(value.replace(',', '')),
                'currency': currency
            })
        metrics['currencies'] = processed_currencies
        
        return metrics
    
    def analyze_overall_sentiment(self, doc: dict) -> Dict[str, Any]:
        """
        Analyze overall sentiment of a document including all its sections.
        
        Args:
            doc: Dictionary containing document sections and metadata
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        # Combine all text sections
        all_text = []
        section_results = {}
        
        for section, content in doc.items():
            if isinstance(content, str) and content.strip():
                # Analyze individual section
                section_sentiment = self.analyze_sentiment(content)
                section_results[section] = section_sentiment.to_dict()
                all_text.append(content)
        
        # Analyze combined text
        combined_text = " ".join(all_text)
        overall_sentiment = self.analyze_sentiment(combined_text)
        
        # Extract metrics
        metrics = self.extract_metrics(combined_text)
        
        return {
            'overall': overall_sentiment.to_dict(),
            'sections': section_results,
            'metrics': metrics
        }