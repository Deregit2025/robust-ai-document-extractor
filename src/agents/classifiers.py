"""
Domain Classifier Interface and Implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import pdfplumber

class BaseDomainClassifier(ABC):
    """Pluggable interface for document domain classification."""
    
    @abstractmethod
    def classify_domain(self, text_content: str) -> Tuple[str, float]:
        """
        Classifies the domain of a document based on its text content.
        
        Args:
            text_content (str): The extracted text from the document.
            
        Returns:
            Tuple[str, float]: The domain name and confidence score (0.0 to 1.0).
        """
        pass

class KeywordDomainClassifier(BaseDomainClassifier):
    """Simple keyword frequency-based domain classifier."""
    
    def __init__(self, domain_keywords: Dict[str, List[str]]):
        """
        Args:
            domain_keywords: Mapping of domain names to lists of keywords.
        """
        self.domain_keywords = {k: [kw.lower() for kw in v] for k, v in domain_keywords.items()}

    def classify_domain(self, text_content: str) -> Tuple[str, float]:
        text_content_lower = text_content.lower()
        
        domain_scores = {domain: 0 for domain in self.domain_keywords}
        total_hits = 0
        
        for domain, keywords in self.domain_keywords.items():
            for kw in keywords:
                # Count occurrences of keyword in text
                count = text_content_lower.count(kw)
                domain_scores[domain] += count
                total_hits += count
                
        # Find highest scoring domain
        if total_hits == 0:
            return "general", 0.5  # Default with low confidence if no keywords found
            
        best_domain = max(domain_scores, key=domain_scores.get)
        confidence = domain_scores[best_domain] / total_hits
        
        # Ensure minimum confidence based on at least some keyword matches
        # If confidence is 1.0 but only 1 keyword hit, it's artificially high. Cap based on hits.
        confidence = min(confidence, min(total_hits / 5.0, 1.0)) 
        
        if confidence < 0.2:
            return "general", 1.0 - confidence
            
        return best_domain, round(confidence, 2)
