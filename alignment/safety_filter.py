"""Safety filtering for harmful content detection and mitigation."""

import re
from typing import List, Dict, Any, Optional
from pathlib import Path


class SafetyFilter:
    """Rule-based safety filter for content moderation.
    
    Detects and filters potentially harmful content including:
    - Hate speech and discrimination
    - Violence and graphic content
    - Sexual content involving minors
    - Self-harm and suicide
    - Illegal activities
    """
    
    def __init__(
        self,
        enable_filter: bool = True,
        keywords_file: Optional[str] = None,
        custom_rules: Optional[Dict[str, List[str]]] = None
    ):
        """Initialize safety filter.
        
        Args:
            enable_filter: Whether to enable filtering
            keywords_file: Path to file with additional keywords
            custom_rules: Custom filtering rules
        """
        self.enable_filter = enable_filter
        
        # Initialize default rules
        self.rules = self._get_default_rules()
        
        # Load additional keywords if provided
        if keywords_file and Path(keywords_file).exists():
            self._load_keywords(keywords_file)
            
        # Add custom rules if provided
        if custom_rules:
            for category, patterns in custom_rules.items():
                if category not in self.rules:
                    self.rules[category] = []
                self.rules[category].extend(patterns)
                
        # Compile regex patterns for efficiency
        self.compiled_patterns = {}
        for category, patterns in self.rules.items():
            self.compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
            
    def _get_default_rules(self) -> Dict[str, List[str]]:
        """Get default safety rules.
        
        Returns:
            Dictionary of safety categories and patterns
        """
        return {
            'hate_speech': [
                r'\b(hate|kill|destroy)\s+(all\s+)?(jews|muslims|christians|blacks|whites|asians)',
                r'\b(n[i1]gg[e3]r|f[a4]gg[o0]t|k[i1]k[e3]|ch[i1]nk|sp[i1]c)\b',
                r'\bsubhuman\b',
                r'\bgenocide\b',
                r'\bethnic\s+cleansing\b'
            ],
            'violence': [
                r'\b(how\s+to\s+)?(make|build|create)\s+(a\s+)?(bomb|explosive|weapon)',
                r'\b(kill|murder|assassinate)\s+(yourself|someone|people)',
                r'\b(school|mass)\s+shooting',
                r'\bterrorist\s+attack',
                r'\btorture\s+(methods|techniques)',
                r'\bexecute\s+someone\b'
            ],
            'self_harm': [
                r'\b(how\s+to\s+)?(commit\s+)?suicide\b',
                r'\b(cut|harm)\s+(yourself|myself)',
                r'\bself[\s-]harm',
                r'\bend\s+my\s+life\b',
                r'\bkill\s+myself\b',
                r'\boverdose\s+on\b'
            ],
            'sexual_minors': [
                r'\b(sex|sexual)\s+(with\s+)?(children|minors|kids)',
                r'\bchild\s+porn',
                r'\bunderage\s+(sex|nude)',
                r'\bpedophil',
                r'\bloli(ta|con)\b',
                r'\bminor\s+.*\s+sexual\b'
            ],
            'illegal_activity': [
                r'\b(how\s+to\s+)?(buy|sell|make)\s+(drugs|cocaine|heroin|meth)',
                r'\b(hack|breach)\s+(into\s+)?(system|network|account)',
                r'\b(steal|rob)\s+(from\s+)?(bank|store|person)',
                r'\bmoney\s+laundering\b',
                r'\btax\s+evasion\b',
                r'\bidentity\s+theft\b'
            ],
            'personal_info': [
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
                r'\b\d{16}\b',  # Credit card pattern
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                r'\b\d{10,11}\b',  # Phone number
                r'\b\d+\s+[A-Za-z]+\s+(Street|St|Avenue|Ave|Road|Rd)\b'  # Address
            ]
        }
        
    def _load_keywords(self, keywords_file: str):
        """Load additional keywords from file.
        
        Args:
            keywords_file: Path to keywords file
        """
        with open(keywords_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Format: category:keyword
                    if ':' in line:
                        category, keyword = line.split(':', 1)
                        if category not in self.rules:
                            self.rules[category] = []
                        self.rules[category].append(re.escape(keyword))
                        
    def check_safety(self, text: str) -> Dict[str, Any]:
        """Check text for safety violations.
        
        Args:
            text: Text to check
            
        Returns:
            Dictionary with safety check results
        """
        if not self.enable_filter:
            return {
                'is_safe': True,
                'violations': [],
                'categories': []
            }
            
        violations = []
        categories = []
        
        # Check each category
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    match = pattern.search(text)
                    violations.append({
                        'category': category,
                        'match': match.group(0),
                        'position': match.span()
                    })
                    if category not in categories:
                        categories.append(category)
                        
        is_safe = len(violations) == 0
        
        return {
            'is_safe': is_safe,
            'violations': violations,
            'categories': categories
        }
        
    def filter_text(self, text: str, replacement: str = "[FILTERED]") -> str:
        """Filter unsafe content from text.
        
        Args:
            text: Text to filter
            replacement: Replacement for filtered content
            
        Returns:
            Filtered text
        """
        if not self.enable_filter:
            return text
            
        filtered_text = text
        
        # Apply each filter
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                filtered_text = pattern.sub(replacement, filtered_text)
                
        return filtered_text
        
    def get_safe_response(self, category: str) -> str:
        """Get safe response for unsafe content.
        
        Args:
            category: Violation category
            
        Returns:
            Safe response message
        """
        responses = {
            'hate_speech': "I cannot and will not generate content that promotes hate or discrimination against any group of people.",
            'violence': "I cannot provide information about violence or harmful activities. If you're in crisis, please contact emergency services or a crisis helpline.",
            'self_harm': "I'm concerned about your wellbeing. If you're having thoughts of self-harm, please reach out to a mental health professional or crisis helpline immediately.",
            'sexual_minors': "I cannot generate any content involving minors in inappropriate contexts.",
            'illegal_activity': "I cannot provide assistance with illegal activities.",
            'personal_info': "I cannot process or share personal identifying information.",
            'default': "I cannot generate this content as it may violate safety guidelines."
        }
        
        return responses.get(category, responses['default'])
        
    def check_prompt_safety(self, prompt: str) -> Dict[str, Any]:
        """Check if a prompt is safe to process.
        
        Args:
            prompt: User prompt
            
        Returns:
            Safety check results
        """
        result = self.check_safety(prompt)
        
        if not result['is_safe']:
            # Get appropriate response
            category = result['categories'][0] if result['categories'] else 'default'
            result['suggested_response'] = self.get_safe_response(category)
            
        return result
        
    def check_response_safety(self, response: str) -> Dict[str, Any]:
        """Check if a model response is safe.
        
        Args:
            response: Model response
            
        Returns:
            Safety check results
        """
        result = self.check_safety(response)
        
        if not result['is_safe']:
            # Filter the response
            result['filtered_response'] = self.filter_text(response)
            
        return result
        
    def moderate_conversation(
        self,
        prompt: str,
        response: str
    ) -> Dict[str, Any]:
        """Moderate a full conversation turn.
        
        Args:
            prompt: User prompt
            response: Model response
            
        Returns:
            Moderation results
        """
        # Check prompt
        prompt_check = self.check_prompt_safety(prompt)
        
        # If prompt is unsafe, return early with safe response
        if not prompt_check['is_safe']:
            return {
                'is_safe': False,
                'prompt_safe': False,
                'response_safe': True,
                'action': 'block_prompt',
                'safe_response': prompt_check['suggested_response'],
                'violations': prompt_check['violations']
            }
            
        # Check response
        response_check = self.check_response_safety(response)
        
        if not response_check['is_safe']:
            return {
                'is_safe': False,
                'prompt_safe': True,
                'response_safe': False,
                'action': 'filter_response',
                'safe_response': response_check.get('filtered_response', response),
                'violations': response_check['violations']
            }
            
        # Both safe
        return {
            'is_safe': True,
            'prompt_safe': True,
            'response_safe': True,
            'action': 'allow',
            'safe_response': response,
            'violations': []
        }