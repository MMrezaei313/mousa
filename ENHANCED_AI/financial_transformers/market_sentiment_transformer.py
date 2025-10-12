# market_sentiment_transformer.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
from torch.utils.data import Dataset, DataLoader
import math
from transformers import AutoTokenizer, AutoModel
import re
from textblob import TextBlob
import requests
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

class SentimentType(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish" 
    NEUTRAL = "neutral"
    VERY_BULLISH = "very_bullish"
    VERY_BEARISH = "very_bearish"

class SentimentSource(Enum):
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    FINANCIAL_REPORTS = "financial_reports"
    ANALYST_RATINGS = "analyst_ratings"
    MARKET_DATA = "market_data"

@dataclass
class SentimentScore:
    overall_score: float
    sentiment_type: SentimentType
    confidence: float
    source_scores: Dict[SentimentSource, float]
    magnitude: float
    subjectivity: float
    key_phrases: List[str]
    timestamp: datetime

@dataclass
class MarketSentiment:
    symbol: str
    current_sentiment: SentimentScore
    sentiment_trend: str
    change_24h: float
    dominant_topics: List[str]
    influential_events: List[Dict]
    market_impact: float

class FinancialNewsDataset(Dataset):
    """Dataset for financial news and sentiment analysis"""
    
    def __init__(self, texts: List[str], labels: List[float], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor([label])
        }

class SentimentAwareAttention(nn.Module):
    """Attention mechanism with sentiment awareness"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super(SentimentAwareAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_sentiment = nn.Linear(d_model, n_heads)  # Sentiment-aware projection
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, sentiment_context: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project inputs
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add sentiment bias if available
        if sentiment_context is not None:
            sentiment_bias = self.w_sentiment(sentiment_context).view(batch_size, 1, 1, self.n_heads)
            attention_scores = attention_scores + sentiment_bias
        
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.layer_norm(output + x)

class MarketSentimentTransformer(nn.Module):
    """
    Transformer model specialized for market sentiment analysis
    """
    
    def __init__(self, 
                 d_model: int = 768,
                 n_heads: int = 12,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 num_sentiment_classes: int = 3,
                 use_pretrained: bool = True):
        super(MarketSentimentTransformer, self).__init__()
        
        self.d_model = d_model
        self.use_pretrained = use_pretrained
        
        # Pretrained language model
        if use_pretrained:
            self.bert = AutoModel.from_pretrained('bert-base-uncased')
            self.bert.resize_token_embeddings(self.bert.config.vocab_size)
        else:
            # Custom transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
            self.embedding = nn.Embedding(30000, d_model)  # Vocabulary size
        
        # Sentiment-aware attention layers
        self.sentiment_attention_layers = nn.ModuleList([
            SentimentAwareAttention(d_model, n_heads, dropout) for _ in range(2)
        ])
        
        # Multi-task output heads
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_sentiment_classes)
        )
        
        self.sentiment_regressor = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1),
            nn.Tanh()  # Output between -1 and 1
        )
        
        self.subjectivity_predictor = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        self.magnitude_predictor = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for sentiment analysis
        
        Returns:
            Dictionary with sentiment predictions
        """
        if self.use_pretrained:
            # Use BERT embeddings
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
        else:
            # Use custom embeddings
            embeddings = self.embedding(input_ids)
            hidden_states = self.transformer_encoder(embeddings)
        
        # Apply sentiment-aware attention
        for attention_layer in self.sentiment_attention_layers:
            hidden_states = attention_layer(hidden_states)
        
        # Use [CLS] token for classification
        cls_output = hidden_states[:, 0, :]
        
        # Multi-task predictions
        sentiment_logits = self.sentiment_classifier(cls_output)
        sentiment_score = self.sentiment_regressor(cls_output)
        subjectivity = self.subjectivity_predictor(cls_output)
        magnitude = self.magnitude_predictor(cls_output)
        
        return {
            'sentiment_logits': sentiment_logits,
            'sentiment_score': sentiment_score,
            'subjectivity': subjectivity,
            'magnitude': magnitude,
            'hidden_states': hidden_states
        }

class MarketSentimentAnalyzer:
    """
    Advanced market sentiment analysis using transformer models
    """
    
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {
            'model_name': 'bert-base-uncased',
            'max_length': 512,
            'batch_size': 16,
            'learning_rate': 2e-5,
            'num_epochs': 10,
            'sentiment_thresholds': {
                'very_bearish': -0.6,
                'bearish': -0.2,
                'neutral': 0.2,
                'bullish': 0.6
            }
        }
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        self.model = None
        self.is_trained = False
        
        # Financial lexicon for rule-based analysis
        self.financial_lexicon = self._load_financial_lexicon()
        
    def _load_financial_lexicon(self) -> Dict[str, float]:
        """Load financial sentiment lexicon"""
        lexicon = {
            # Bullish terms
            'bullish': 0.8, 'rally': 0.7, 'surge': 0.8, 'soar': 0.9, 'jump': 0.6,
            'gain': 0.5, 'profit': 0.4, 'growth': 0.6, 'optimistic': 0.7,
            'positive': 0.5, 'strong': 0.4, 'outperform': 0.7, 'buy': 0.8,
            'upgrade': 0.6, 'beat': 0.5, 'record': 0.4, 'breakout': 0.7,
            
            # Bearish terms
            'bearish': -0.8, 'plunge': -0.9, 'slump': -0.7, 'drop': -0.6,
            'fall': -0.5, 'loss': -0.6, 'decline': -0.5, 'pessimistic': -0.7,
            'negative': -0.5, 'weak': -0.4, 'underperform': -0.7, 'sell': -0.8,
            'downgrade': -0.6, 'miss': -0.5, 'crash': -0.9, 'collapse': -0.9,
            
            # Risk terms
            'risk': -0.3, 'volatility': -0.2, 'uncertainty': -0.4, 'concern': -0.5,
            'warning': -0.6, 'caution': -0.4, 'fear': -0.7, 'worry': -0.5,
            
            # Opportunity terms
            'opportunity': 0.4, 'potential': 0.3, 'recovery': 0.5, 'rebound': 0.6,
            'momentum': 0.4, 'trend': 0.2, 'support': 0.3, 'resistance': -0.2
        }
        return lexicon
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove special characters but keep financial symbols ($, %)
        text = re.sub(r'[^\w\s$%]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_financial_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract financial entities from text"""
        entities = {
            'tickers': re.findall(r'\$[A-Z]+', text),  # $AAPL, $GOOG
            'percentages': re.findall(r'\d+%', text),  # 10%, -5%
            'prices': re.findall(r'\$\d+\.?\d*', text),  # $150, $150.50
            'financial_terms': [],
            'key_phrases': []
        }
        
        # Extract financial terms from lexicon
        for term in self.financial_lexicon.keys():
            if term in text.lower():
                entities['financial_terms'].append(term)
        
        # Simple key phrase extraction (can be enhanced with NLP)
        sentences = text.split('.')
        for sentence in sentences:
            if any(term in sentence.lower() for term in ['earnings', 'revenue', 'profit', 'loss', 'growth']):
                entities['key_phrases'].append(sentence.strip())
        
        return entities
    
    def rule_based_sentiment(self, text: str) -> Dict[str, float]:
        """Rule-based sentiment analysis as fallback"""
        text = self.preprocess_text(text)
        words = text.split()
        
        sentiment_score = 0.0
        word_count = 0
        intensity_multiplier = 1.0
        
        for i, word in enumerate(words):
            if word in self.financial_lexicon:
                # Check for intensifiers
                if i > 0 and words[i-1] in ['very', 'extremely', 'highly']:
                    intensity_multiplier = 1.5
                elif i > 0 and words[i-1] in ['slightly', 'somewhat']:
                    intensity_multiplier = 0.7
                
                sentiment_score += self.financial_lexicon[word] * intensity_multiplier
                word_count += 1
                intensity_multiplier = 1.0  # Reset
        
        # Normalize score
        if word_count > 0:
            sentiment_score /= word_count
        
        # Use TextBlob as additional signal
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Combine scores
        combined_score = (sentiment_score * 0.6 + textblob_sentiment * 0.4)
        
        return {
            'score': combined_score,
            'subjectivity': subjectivity,
            'word_count': word_count,
            'method': 'rule_based'
        }
    
    def train(self, 
              texts: List[str], 
              labels: List[float],
              val_texts: List[str] = None,
              val_labels: List[float] = None) -> Dict[str, List[float]]:
        """
        Train the sentiment transformer model
        
        Args:
            texts: List of text samples
            labels: List of sentiment labels (-1 to 1)
            val_texts: Validation texts
            val_labels: Validation labels
            
        Returns:
            Training history
        """
        
        # Create datasets
        train_dataset = FinancialNewsDataset(texts, labels, self.tokenizer, self.config['max_length'])
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        
        val_loader = None
        if val_texts is not None and val_labels is not None:
            val_dataset = FinancialNewsDataset(val_texts, val_labels, self.tokenizer, self.config['max_length'])
            val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'])
        
        # Initialize model
        self.model = MarketSentimentTransformer(use_pretrained=True).to(self.device)
        
        # Loss functions and optimizer
        regression_criterion = nn.MSELoss()
        classification_criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'])
        
        # Training history
        history = {'train_loss': [], 'val_loss': []}
        
        # Training loop
        for epoch in range(self.config['num_epochs']):
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(input_ids, attention_mask)
                
                # Multi-task loss
                regression_loss = regression_criterion(outputs['sentiment_score'].squeeze(), labels)
                
                # Convert regression labels to classification (simplified)
                class_labels = self._regression_to_classification(labels)
                classification_loss = classification_criterion(outputs['sentiment_logits'], class_labels)
                
                # Combined loss
                total_loss = regression_loss + 0.3 * classification_loss
                total_loss.backward()
                
                optimizer.step()
                
                train_loss += total_loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation
            val_loss = 0.0
            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        
                        outputs = self.model(input_ids, attention_mask)
                        regression_loss = regression_criterion(outputs['sentiment_score'].squeeze(), labels)
                        val_loss += regression_loss.item()
                
                val_loss /= len(val_loader)
                history['val_loss'].append(val_loss)
            
            if epoch % 2 == 0:
                self.logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        self.is_trained = True
        return history
    
    def _regression_to_classification(self, regression_labels: torch.Tensor) -> torch.Tensor:
        """Convert regression labels to classification labels"""
        class_labels = []
        for score in regression_labels:
            if score < -0.33:
                class_labels.append(0)  # Bearish
            elif score > 0.33:
                class_labels.append(2)  # Bullish
            else:
                class_labels.append(1)  # Neutral
        
        return torch.LongTensor(class_labels).to(regression_labels.device)
    
    def analyze_sentiment(self, text: str, source: SentimentSource = None) -> SentimentScore:
        """
        Analyze sentiment of a single text
        
        Args:
            text: Input text
            source: Source of the text
            
        Returns:
            SentimentScore object
        """
        if not text or len(text.strip()) == 0:
            return self._create_neutral_sentiment()
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Extract entities and key phrases
        entities = self.extract_financial_entities(text)
        key_phrases = entities['key_phrases'][:5]  # Top 5 phrases
        
        # Use transformer model if available
        if self.is_trained:
            transformer_score = self._predict_with_transformer(processed_text)
        else:
            transformer_score = None
        
        # Rule-based analysis as fallback
        rule_based_result = self.rule_based_sentiment(processed_text)
        
        # Combine scores
        if transformer_score is not None:
            final_score = transformer_score * 0.7 + rule_based_result['score'] * 0.3
            confidence = 0.8
        else:
            final_score = rule_based_result['score']
            confidence = max(0.5, 1 - rule_based_result['subjectivity'])
        
        # Determine sentiment type
        sentiment_type = self._score_to_sentiment_type(final_score)
        
        # Source scores (simplified - in practice would vary by source)
        source_scores = {
            SentimentSource.NEWS: final_score * 0.9,
            SentimentSource.SOCIAL_MEDIA: final_score * 0.7,
            SentimentSource.FINANCIAL_REPORTS: final_score * 0.95,
            SentimentSource.ANALYST_RATINGS: final_score * 0.85,
            SentimentSource.MARKET_DATA: final_score * 0.8
        }
        
        return SentimentScore(
            overall_score=final_score,
            sentiment_type=sentiment_type,
            confidence=confidence,
            source_scores=source_scores,
            magnitude=abs(final_score),
            subjectivity=rule_based_result['subjectivity'],
            key_phrases=key_phrases,
            timestamp=datetime.now()
        )
    
    def _predict_with_transformer(self, text: str) -> float:
        """Predict sentiment using transformer model"""
        self.model.eval()
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config['max_length'],
            return_tensors='pt'
        )
        
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            outputs = self.model(input_ids, attention_mask)
            sentiment_score = outputs['sentiment_score'].item()
        
        return sentiment_score
    
    def _score_to_sentiment_type(self, score: float) -> SentimentType:
        """Convert numerical score to sentiment type"""
        thresholds = self.config['sentiment_thresholds']
        
        if score <= thresholds['very_bearish']:
            return SentimentType.VERY_BEARISH
        elif score <= thresholds['bearish']:
            return SentimentType.BEARISH
        elif score >= thresholds['bullish']:
            return SentimentType.BULLISH
        elif score >= thresholds['very_bullish']:
            return SentimentType.VERY_BULLISH
        else:
            return SentimentType.NEUTRAL
    
    def _create_neutral_sentiment(self) -> SentimentScore:
        """Create a neutral sentiment score for empty text"""
        return SentimentScore(
            overall_score=0.0,
            sentiment_type=SentimentType.NEUTRAL,
            confidence=0.5,
            source_scores={source: 0.0 for source in SentimentSource},
            magnitude=0.0,
            subjectivity=0.5,
            key_phrases=[],
            timestamp=datetime.now()
        )
    
    def analyze_market_sentiment(self, 
                               symbol: str,
                               news_articles: List[str] = None,
                               social_posts: List[str] = None,
                               analyst_reports: List[str] = None) -> MarketSentiment:
        """
        Comprehensive market sentiment analysis for a symbol
        
        Args:
            symbol: Stock symbol
            news_articles: List of news articles
            social_posts: List of social media posts
            analyst_reports: List of analyst reports
            
        Returns:
            MarketSentiment object
        """
        
        all_texts = []
        source_mapping = []
        
        # Collect texts from different sources
        if news_articles:
            all_texts.extend(news_articles)
            source_mapping.extend([SentimentSource.NEWS] * len(news_articles))
        
        if social_posts:
            all_texts.extend(social_posts)
            source_mapping.extend([SentimentSource.SOCIAL_MEDIA] * len(social_posts))
        
        if analyst_reports:
            all_texts.extend(analyst_reports)
            source_mapping.extend([SentimentSource.ANALYST_RATINGS] * len(analyst_reports))
        
        if not all_texts:
            return self._create_default_market_sentiment(symbol)
        
        # Analyze each text
        sentiment_scores = []
        for text, source in zip(all_texts, source_mapping):
            score = self.analyze_sentiment(text, source)
            sentiment_scores.append(score)
        
        # Aggregate scores
        overall_score = np.mean([s.overall_score for s in sentiment_scores])
        confidence = np.mean([s.confidence for s in sentiment_scores])
        
        # Calculate source-specific scores
        source_scores = {}
        for source in SentimentSource:
            source_sentiments = [s for s, src in zip(sentiment_scores, source_mapping) if src == source]
            if source_sentiments:
                source_scores[source] = np.mean([s.overall_score for s in source_sentiments])
            else:
                source_scores[source] = 0.0
        
        # Determine dominant topics
        dominant_topics = self._extract_dominant_topics([s.key_phrases for s in sentiment_scores])
        
        # Create current sentiment
        current_sentiment = SentimentScore(
            overall_score=overall_score,
            sentiment_type=self._score_to_sentiment_type(overall_score),
            confidence=confidence,
            source_scores=source_scores,
            magnitude=np.mean([s.magnitude for s in sentiment_scores]),
            subjectivity=np.mean([s.subjectivity for s in sentiment_scores]),
            key_phrases=dominant_topics[:3],
            timestamp=datetime.now()
        )
        
        # Calculate trend (simplified - in practice would use historical data)
        sentiment_trend = self._calculate_sentiment_trend(sentiment_scores)
        
        return MarketSentiment(
            symbol=symbol,
            current_sentiment=current_sentiment,
            sentiment_trend=sentiment_trend,
            change_24h=0.0,  # Would require historical data
            dominant_topics=dominant_topics,
            influential_events=self._identify_influential_events(sentiment_scores),
            market_impact=self._estimate_market_impact(current_sentiment)
        )
    
    def _extract_dominant_topics(self, all_key_phrases: List[List[str]]) -> List[str]:
        """Extract dominant topics from key phrases"""
        # Flatten list of phrases
        all_phrases = [phrase for sublist in all_key_phrases for phrase in sublist]
        
        # Simple frequency-based topic extraction
        phrase_freq = {}
        for phrase in all_phrases:
            if phrase in phrase_freq:
                phrase_freq[phrase] += 1
            else:
                phrase_freq[phrase] = 1
        
        # Return top phrases by frequency
        return sorted(phrase_freq.keys(), key=lambda x: phrase_freq[x], reverse=True)[:5]
    
    def _calculate_sentiment_trend(self, sentiment_scores: List[SentimentScore]) -> str:
        """Calculate sentiment trend"""
        if len(sentiment_scores) < 2:
            return "stable"
        
        # Use recent scores for trend calculation
        recent_scores = [s.overall_score for s in sentiment_scores[-10:]]
        
        if len(recent_scores) >= 2:
            # Simple linear trend
            x = np.arange(len(recent_scores))
            slope, _ = np.polyfit(x, recent_scores, 1)
            
            if slope > 0.05:
                return "improving"
            elif slope < -0.05:
                return "deteriorating"
        
        return "stable"
    
    def _identify_influential_events(self, sentiment_scores: List[SentimentScore]) -> List[Dict]:
        """Identify influential events from sentiment scores"""
        influential_events = []
        
        for score in sentiment_scores:
            if abs(score.overall_score) > 0.7 and score.confidence > 0.7:
                event = {
                    'sentiment_score': score.overall_score,
                    'sentiment_type': score.sentiment_type.value,
                    'key_phrases': score.key_phrases[:2],
                    'magnitude': score.magnitude,
                    'timestamp': score.timestamp
                }
                influential_events.append(event)
        
        return influential_events[:3]  # Return top 3 events
    
    def _estimate_market_impact(self, sentiment: SentimentScore) -> float:
        """Estimate potential market impact of sentiment"""
        # Simple impact estimation based on sentiment strength and confidence
        base_impact = abs(sentiment.overall_score) * sentiment.confidence
        
        # Adjust based on sentiment sources
        source_weights = {
            SentimentSource.ANALYST_RATINGS: 1.2,
            SentimentSource.FINANCIAL_REPORTS: 1.1,
            SentimentSource.NEWS: 1.0,
            SentimentSource.MARKET_DATA: 0.9,
            SentimentSource.SOCIAL_MEDIA: 0.7
        }
        
        weighted_source_impact = 0.0
        total_weight = 0.0
        
        for source, score in sentiment.source_scores.items():
            weight = source_weights.get(source, 1.0)
            weighted_source_impact += abs(score) * weight
            total_weight += weight
        
        if total_weight > 0:
            source_impact = weighted_source_impact / total_weight
        else:
            source_impact = base_impact
        
        return (base_impact + source_impact) / 2
    
    def _create_default_market_sentiment(self, symbol: str) -> MarketSentiment:
        """Create default market sentiment when no data is available"""
        default_sentiment = self._create_neutral_sentiment()
        
        return MarketSentiment(
            symbol=symbol,
            current_sentiment=default_sentiment,
            sentiment_trend="stable",
            change_24h=0.0,
            dominant_topics=["No data available"],
            influential_events=[],
            market_impact=0.0
        )

# Example usage
if __name__ == "__main__":
    # Initialize sentiment analyzer
    sentiment_analyzer = MarketSentimentAnalyzer()
    
    # Sample financial texts
    sample_texts = [
        "Apple stock surges to record high after strong earnings report beat expectations",
        "Market fears grow as Fed signals potential interest rate hikes amid inflation concerns",
        "Tesla shares plunge 10% following production delays and supply chain issues",
        "Amazon reports steady growth with cloud division continuing to outperform",
        "Investors remain cautious amid geopolitical tensions and market volatility"
    ]
    
    # Sample labels for training (-1 to 1)
    sample_labels = [0.8, -0.6, -0.9, 0.4, -0.3]
    
    print("Training sentiment model...")
    history = sentiment_analyzer.train(sample_texts, sample_labels)
    
    print("\nAnalyzing sample texts:")
    for text in sample_texts:
        sentiment = sentiment_analyzer.analyze_sentiment(text)
        print(f"Text: {text[:80]}...")
        print(f"Sentiment: {sentiment.sentiment_type.value} (Score: {sentiment.overall_score:.2f})")
        print(f"Confidence: {sentiment.confidence:.2f}")
        print(f"Key Phrases: {sentiment.key_phrases[:2]}")
        print("-" * 80)
    
    # Comprehensive market sentiment analysis
    print("\nComprehensive Market Sentiment Analysis:")
    market_sentiment = sentiment_analyzer.analyze_market_sentiment(
        symbol="AAPL",
        news_articles=sample_texts[:3],
        social_posts=sample_texts[3:],
        analyst_reports=sample_texts[2:4]
    )
    
    print(f"Symbol: {market_sentiment.symbol}")
    print(f"Overall Sentiment: {market_sentiment.current_sentiment.sentiment_type.value}")
    print(f"Sentiment Score: {market_sentiment.current_sentiment.overall_score:.2f}")
    print(f"Trend: {market_sentiment.sentiment_trend}")
    print(f"Dominant Topics: {market_sentiment.dominant_topics}")
    print(f"Market Impact: {market_sentiment.market_impact:.2f}")
