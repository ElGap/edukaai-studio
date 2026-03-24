"""
Core utilities and exception handling.
"""

import json
from typing import Any, Dict, List, Optional
from datetime import datetime


class EdukaAIException(Exception):
    """Base exception for EdukaAI Studio."""
    
    def __init__(
        self, 
        detail: str, 
        status_code: int = 500, 
        error_code: str = "internal_error"
    ):
        self.detail = detail
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(self.detail)


class ValidationError(EdukaAIException):
    """Validation error."""
    
    def __init__(self, detail: str):
        super().__init__(detail, status_code=422, error_code="validation_error")


class NotFoundError(EdukaAIException):
    """Resource not found error."""
    
    def __init__(self, detail: str = "Resource not found"):
        super().__init__(detail, status_code=404, error_code="not_found")


class ResourceLimitError(EdukaAIException):
    """Resource limit exceeded error."""
    
    def __init__(self, detail: str = "Resource limit exceeded"):
        super().__init__(detail, status_code=429, error_code="resource_limit")


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks."""
    import re
    # Remove path separators and null bytes
    filename = re.sub(r'[\\/]', '', filename)
    filename = filename.replace('\x00', '')
    # Limit length
    return filename[:255]


def validate_jsonl_format(content: str) -> tuple[bool, List[Dict], List[Dict]]:
    """
    Validate JSONL format and return (is_valid, valid_samples, errors).
    Also handles JSON arrays and common formatting issues.
    
    Returns:
        tuple: (is_valid, valid_samples, errors)
    """
    valid_samples = []
    errors = []
    
    # First, try to parse entire content as JSON array (common for Alpaca format)
    content_stripped = content.strip()
    if content_stripped.startswith('[') and content_stripped.endswith(']'):
        try:
            json_array = json.loads(content_stripped)
            if isinstance(json_array, list):
                for i, sample in enumerate(json_array, 1):
                    if isinstance(sample, dict):
                        valid_samples.append({
                            "line": i,
                            "data": sample
                        })
                return len(valid_samples) > 0, valid_samples, []
        except json.JSONDecodeError:
            pass  # Not a valid JSON array, fall back to JSONL parsing
    
    # Try standard JSONL (line-by-line)
    lines = content.split('\n')
    
    # Collect potential JSON objects that might span multiple lines (pretty-printed)
    current_object = ""
    in_object = False
    brace_count = 0
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        
        # Try standard single-line JSON first
        if line.startswith('{') and line.endswith('}'):
            try:
                sample = json.loads(line)
                valid_samples.append({
                    "line": i,
                    "data": sample
                })
                continue
            except json.JSONDecodeError:
                pass
        
        # Handle multi-line JSON objects
        if line.startswith('{') and not in_object:
            in_object = True
            current_object = line
            brace_count = line.count('{') - line.count('}')
        elif in_object:
            current_object += " " + line
            brace_count += line.count('{') - line.count('}')
            
            if brace_count == 0 and line.endswith('}'):
                # Complete object
                try:
                    sample = json.loads(current_object)
                    valid_samples.append({
                        "line": i,
                        "data": sample
                    })
                    in_object = False
                    current_object = ""
                    brace_count = 0
                    continue
                except json.JSONDecodeError as e:
                    errors.append({
                        "line": i,
                        "error": f"Invalid JSON object: {str(e)}",
                        "preview": current_object[:200]
                    })
                    in_object = False
                    current_object = ""
                    brace_count = 0
        else:
            # Line doesn't start with { and we're not in an object
            errors.append({
                "line": i,
                "error": "Invalid JSON: Line doesn't contain a valid JSON object",
                "preview": line[:100]
            })
    
    # Handle any remaining object
    if in_object and current_object:
        try:
            sample = json.loads(current_object)
            valid_samples.append({
                "line": len(lines),
                "data": sample
            })
        except json.JSONDecodeError as e:
            errors.append({
                "line": len(lines),
                "error": f"Unclosed JSON object: {str(e)}",
                "preview": current_object[:200]
            })
    
    return len(errors) == 0, valid_samples, errors


def detect_format(valid_samples: List[Dict]) -> str:
    """Detect dataset format from samples."""
    if not valid_samples:
        return "unknown"
    
    sample = valid_samples[0]["data"]
    
    # Check for chat format
    if "messages" in sample:
        return "chat"
    
    # Check for completion format (Alpaca-style)
    if "instruction" in sample or "prompt" in sample:
        return "completion"
    
    # Check for text format
    if "text" in sample:
        return "text"
    
    return "unknown"


# =============================================================================
# Security: Dataset Content Sanitization
# =============================================================================

import re
from typing import Tuple

# Maximum allowed length for any text field in dataset
MAX_SAMPLE_LENGTH = 10000

# Dangerous patterns to strip from training data
DANGEROUS_PATTERNS = [
    (re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL), '[SCRIPT_REMOVED]'),
    (re.compile(r'<iframe[^>]*>.*?</iframe>', re.IGNORECASE | re.DOTALL), '[IFRAME_REMOVED]'),
    (re.compile(r'<object[^>]*>.*?</object>', re.IGNORECASE | re.DOTALL), '[OBJECT_REMOVED]'),
    (re.compile(r'<embed[^>]*>.*?</embed>', re.IGNORECASE | re.DOTALL), '[EMBED_REMOVED]'),
    (re.compile(r'javascript:', re.IGNORECASE), '[JS_REMOVED]'),
    (re.compile(r'data:text/html', re.IGNORECASE), '[DATA_URI_REMOVED]'),
    (re.compile(r'on\w+\s*=', re.IGNORECASE), '[EVENT_HANDLER_REMOVED]'),
]


def strip_html_tags(text: str) -> str:
    """Remove HTML tags from text."""
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    return text


def sanitize_text_content(text: str, max_length: int = MAX_SAMPLE_LENGTH) -> Tuple[str, List[str]]:
    """
    Sanitize text content for safe storage in datasets.
    
    Returns:
        Tuple of (sanitized_text, warnings)
    """
    if not text:
        return "", []
    
    warnings = []
    
    # Check length and truncate if needed
    if len(text) > max_length:
        warnings.append(f"Text exceeds {max_length} characters, truncated")
        text = text[:max_length]
    
    # Remove dangerous patterns
    for pattern, replacement in DANGEROUS_PATTERNS:
        if pattern.search(text):
            text = pattern.sub(replacement, text)
            warnings.append(f"Removed dangerous content pattern")
    
    # Strip remaining HTML tags
    text = strip_html_tags(text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text, warnings


def detect_pii(text: str) -> List[Tuple[str, str]]:
    """
    Detect potential PII (Personally Identifiable Information) in text.
    
    Returns:
        List of tuples: (pii_type, matched_text)
    """
    if not text:
        return []
    
    pii_patterns = {
        # Contact Information
        'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        'phone': re.compile(r'\b(?:\+?1[-.]?)?\s*\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),
        
        # Government IDs
        'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        'ssn_no_dashes': re.compile(r'\b\d{9}\b'),
        'passport': re.compile(r'\b[A-Z]{1,2}\d{6,9}\b'),
        'drivers_license': re.compile(r'\b[A-Z]{1,2}\d{6,8}\b'),
        
        # Financial
        'credit_card': re.compile(r'\b(?:\d{4}[- ]?){3}\d{4}\b'),
        'credit_card_amex': re.compile(r'\b3[47]\d{13}\b'),
        'bank_account': re.compile(r'\b\d{8,17}\b'),
        'routing_number': re.compile(r'\b\d{9}\b'),
        
        # Authentication
        'api_key': re.compile(r'\b(?:api[_-]?key|token|secret|password)["\']?\s*[:=]\s*["\']?[a-zA-Z0-9_\-]{16,}\b', re.IGNORECASE),
        'jwt_token': re.compile(r'\beyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*\b'),
        'uuid': re.compile(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', re.IGNORECASE),
        'ip_address': re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
        
        # Personal
        'date_of_birth': re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),
        'zip_code': re.compile(r'\b\d{5}(-\d{4})?\b'),
        
        # Medical
        'medical_record': re.compile(r'\bMR[N]?[\s-]?\d{6,10}\b', re.IGNORECASE),
        'health_plan': re.compile(r'\b[Hh][Pp][\s-]?\d{8,12}\b'),
    }
    
    findings = []
    for pii_type, pattern in pii_patterns.items():
        for match in pattern.finditer(text):
            matched_text = match.group()
            # Basic validation to reduce false positives
            if pii_type == 'ip_address' and not _is_valid_ip(matched_text):
                continue
            if pii_type == 'zip_code' and not _is_valid_zip(matched_text):
                continue
            findings.append((pii_type, matched_text))
    
    return findings


def _is_valid_ip(ip: str) -> bool:
    """Validate IP address format."""
    parts = ip.split('.')
    if len(parts) != 4:
        return False
    for part in parts:
        try:
            num = int(part)
            if num < 0 or num > 255:
                return False
        except ValueError:
            return False
    return True


def _is_valid_zip(zip_code: str) -> bool:
    """Validate US zip code."""
    if '-' in zip_code:
        parts = zip_code.split('-')
        if len(parts) != 2:
            return False
        if not (parts[0].isdigit() and len(parts[0]) == 5):
            return False
        if not (parts[1].isdigit() and len(parts[1]) == 4):
            return False
    else:
        if not (zip_code.isdigit() and len(zip_code) == 5):
            return False
    return True


def anonymize_pii(text: str, findings: List[Tuple[str, str]]) -> Tuple[str, Dict[str, Any]]:
    """
    Anonymize PII in text by replacing with tokens.
    
    Args:
        text: Original text containing PII
        findings: List of (pii_type, matched_text) from detect_pii()
        
    Returns:
        Tuple of (anonymized_text, stats)
        stats contains: {
            'replacements_made': int,
            'types_found': Dict[str, int],
            'example_replacements': List[Tuple[str, str]]
        }
    """
    if not findings:
        return text, {'replacements_made': 0, 'types_found': {}, 'example_replacements': []}
    
    anonymized = text
    counters = {}
    types_found = {}
    example_replacements = []
    
    # Sort by length (longest first) to avoid partial replacements
    sorted_findings = sorted(findings, key=lambda x: len(x[1]), reverse=True)
    
    for pii_type, original in sorted_findings:
        # Skip if already replaced (avoid double counting)
        if original not in anonymized:
            continue
            
        counters[pii_type] = counters.get(pii_type, 0) + 1
        idx = counters[pii_type]
        types_found[pii_type] = types_found.get(pii_type, 0) + 1
        
        # Generate replacement token
        if pii_type == 'email':
            replacement = f"[EMAIL_{idx}@ANON.COM]"
        elif pii_type == 'ssn':
            replacement = f"[SSN_{idx}]"
        elif pii_type == 'ssn_no_dashes':
            replacement = f"[SSN_ND_{idx}]"
        elif pii_type == 'credit_card':
            replacement = f"[CC_{idx}]"
        elif pii_type == 'credit_card_amex':
            replacement = f"[CC_AMEX_{idx}]"
        elif pii_type == 'phone':
            replacement = f"[PHONE_{idx}]"
        elif pii_type == 'api_key':
            replacement = f"[API_KEY_{idx}]"
        elif pii_type == 'jwt_token':
            replacement = f"[JWT_TOKEN_{idx}]"
        elif pii_type == 'uuid':
            replacement = f"[UUID_{idx}]"
        elif pii_type == 'ip_address':
            replacement = f"[IP_{idx}]"
        elif pii_type == 'date_of_birth':
            replacement = f"[DOB_{idx}]"
        elif pii_type == 'zip_code':
            replacement = f"[ZIP_{idx}]"
        elif pii_type == 'passport':
            replacement = f"[PASSPORT_{idx}]"
        elif pii_type == 'drivers_license':
            replacement = f"[DL_{idx}]"
        elif pii_type == 'bank_account':
            replacement = f"[BANK_ACCT_{idx}]"
        elif pii_type == 'routing_number':
            replacement = f"[ROUTE_{idx}]"
        elif pii_type == 'medical_record':
            replacement = f"[MRN_{idx}]"
        elif pii_type == 'health_plan':
            replacement = f"[HPID_{idx}]"
        else:
            replacement = f"[{pii_type.upper()}_{idx}]"
        
        # Replace only first occurrence (we're going in order)
        anonymized = anonymized.replace(original, replacement, 1)
        
        # Store example (limit to 5 for privacy)
        if len(example_replacements) < 5:
            # Show only first 3 chars of original for logging
            masked_original = original[:3] + "***" if len(original) > 3 else "***"
            example_replacements.append((masked_original, replacement))
    
    stats = {
        'replacements_made': sum(counters.values()),
        'types_found': types_found,
        'example_replacements': example_replacements
    }
    
    return anonymized, stats


def sanitize_dataset_sample(sample: Dict[str, Any], sample_index: int = 0) -> Tuple[Dict[str, Any], List[str], Dict[str, Any]]:
    """
    Sanitize a single dataset sample with PII anonymization.
    
    Args:
        sample: The dataset sample dictionary
        sample_index: Index for error reporting
        
    Returns:
        Tuple of (sanitized_sample, warnings, pii_stats)
        pii_stats contains anonymization tracking data
    """
    warnings = []
    pii_stats = {
        'total_replacements': 0,
        'types_found': {},
        'fields_affected': [],
        'sample_index': sample_index
    }
    sanitized = sample.copy()
    
    # Fields to sanitize
    text_fields = ['instruction', 'input', 'output', 'text', 'prompt', 'response']
    
    for field in text_fields:
        if field in sanitized and isinstance(sanitized[field], str):
            original_text = sanitized[field]
            
            # Step 1: Sanitize content (strip HTML, dangerous patterns)
            sanitized_text, content_warnings = sanitize_text_content(original_text)
            for warning in content_warnings:
                warnings.append(f"Sample {sample_index}, field '{field}': {warning}")
            
            # Step 2: Detect PII
            pii_findings = detect_pii(sanitized_text)
            
            if pii_findings:
                # Step 3: ANONYMIZE PII (Option 2 implementation)
                anonymized_text, anon_stats = anonymize_pii(sanitized_text, pii_findings)
                
                # Track stats
                pii_stats['total_replacements'] += anon_stats['replacements_made']
                pii_stats['fields_affected'].append(field)
                
                for pii_type, count in anon_stats['types_found'].items():
                    pii_stats['types_found'][pii_type] = pii_stats['types_found'].get(pii_type, 0) + count
                
                # Build anonymization summary
                pii_types = list(anon_stats['types_found'].keys())
                total_replaced = anon_stats['replacements_made']
                
                warnings.append(
                    f"Sample {sample_index}, field '{field}': "
                    f"Anonymized {total_replaced} PII instances "
                    f"({', '.join(pii_types)})"
                )
                
                # Replace with anonymized version
                sanitized[field] = anonymized_text
            else:
                # No PII found, use sanitized version
                sanitized[field] = sanitized_text
    
    return sanitized, warnings, pii_stats


def sanitize_dataset_content(content: str) -> Tuple[str, List[str], Dict[str, Any]]:
    """
    Sanitize entire dataset content before storage with full PII anonymization.
    
    Args:
        content: Raw JSONL or JSON array content
        
    Returns:
        Tuple of (sanitized_content, all_warnings, anonymization_report)
        anonymization_report contains complete stats for logging and display
    """
    all_warnings = []
    total_stats = {
        'total_samples': 0,
        'samples_with_pii': 0,
        'total_replacements': 0,
        'types_found': {},
        'fields_affected': set(),
        'sample_details': []
    }
    
    # Parse content
    is_valid, samples, errors = validate_jsonl_format(content)
    
    if not is_valid or not samples:
        return content, ["Failed to parse dataset for sanitization"], total_stats
    
    total_stats['total_samples'] = len(samples)
    
    # Sanitize each sample
    sanitized_samples = []
    for i, sample_data in enumerate(samples):
        sample = sample_data.get("data", sample_data) if isinstance(sample_data, dict) else sample_data
        
        if not isinstance(sample, dict):
            continue
            
        sanitized_sample, warnings, pii_stats = sanitize_dataset_sample(sample, sample_index=i+1)
        all_warnings.extend(warnings)
        sanitized_samples.append(sanitized_sample)
        
        # Aggregate stats
        if pii_stats['total_replacements'] > 0:
            total_stats['samples_with_pii'] += 1
            total_stats['total_replacements'] += pii_stats['total_replacements']
            total_stats['fields_affected'].update(pii_stats['fields_affected'])
            
            for pii_type, count in pii_stats['types_found'].items():
                total_stats['types_found'][pii_type] = total_stats['types_found'].get(pii_type, 0) + count
            
            # Store sample detail (limited to first 10 for brevity)
            if len(total_stats['sample_details']) < 10:
                total_stats['sample_details'].append({
                    'sample_index': i + 1,
                    'replacements': pii_stats['total_replacements'],
                    'types': list(pii_stats['types_found'].keys())
                })
    
    # Convert set to list for JSON serialization
    total_stats['fields_affected'] = list(total_stats['fields_affected'])
    
    # Serialize back to JSONL
    try:
        sanitized_content = '\n'.join(json.dumps(s) for s in sanitized_samples)
        return sanitized_content, all_warnings, total_stats
    except Exception as e:
        return content, [f"Error serializing sanitized content: {e}"], total_stats


# =============================================================================


def format_datetime(dt: datetime) -> str:
    """Format datetime for JSON serialization."""
    return dt.isoformat() if dt else None
