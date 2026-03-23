"""
HuggingFace Model Validator and Manager

Validates and fetches metadata for custom HuggingFace models.
"""

import json
import ssl
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

# Try to import certifi for better SSL certificate handling
try:
    import certifi
    CERTIFI_AVAILABLE = True
except ImportError:
    CERTIFI_AVAILABLE = False


@dataclass
class HFModelInfo:
    """Container for HuggingFace model information."""
    model_id: str
    name: str
    description: str
    size_gb: float
    format: str
    tags: List[str]
    architecture: str
    mlx_compatible: bool
    quantized: bool
    downloads: int
    last_modified: str
    error: Optional[str] = None


class HFModelValidator:
    """Validates HuggingFace models for MLX compatibility."""
    
    def __init__(self, timeout: int = 30, token: Optional[str] = None):
        self.timeout = timeout
        self.token = token
        self.headers = {
            'User-Agent': 'EdukaAI-Studio/1.0'
        }
        if token:
            self.headers['Authorization'] = f'Bearer {token}'
    
    def _make_request(self, url: str) -> Tuple[int, Dict, Optional[str]]:
        """Make HTTP request using urllib with SSL handling."""
        
        # First try with proper SSL certificates
        try:
            req = urllib.request.Request(url, headers=self.headers)
            
            # Create SSL context with certifi certificates if available
            if CERTIFI_AVAILABLE:
                ssl_context = ssl.create_default_context(cafile=certifi.where())
            else:
                ssl_context = ssl.create_default_context()
            
            with urllib.request.urlopen(req, timeout=self.timeout, context=ssl_context) as response:
                data = json.loads(response.read().decode('utf-8'))
                return response.getcode(), data, None
                
        except urllib.error.HTTPError as e:
            error_msg = f"HTTP {e.code}: {e.reason}"
            return e.code, {}, error_msg
            
        except (urllib.error.URLError, ssl.SSLError) as e:
            # SSL or connection error - try with unverified SSL
            error_reason = str(e.reason) if hasattr(e, 'reason') else str(e)
            
            if "certificate" in error_reason.lower() or "ssl" in error_reason.lower() or isinstance(e, ssl.SSLError):
                print(f"[HF VALIDATOR] SSL certificate error, trying with unverified SSL...")
                
                try:
                    # Create unverified SSL context (for testing/development only)
                    ssl_context = ssl.create_unverified_context()
                    req = urllib.request.Request(url, headers=self.headers)
                    
                    with urllib.request.urlopen(req, timeout=self.timeout, context=ssl_context) as response:
                        data = json.loads(response.read().decode('utf-8'))
                        print(f"[HF VALIDATOR] Success with unverified SSL (using system trust)")
                        return response.getcode(), data, None
                        
                except Exception as fallback_e:
                    error_msg = f"SSL error even with unverified context: {str(fallback_e)}"
                    print(f"[HF VALIDATOR] {error_msg}")
                    return -1, {}, error_msg
            
            # Other URL errors
            if "timeout" in error_reason.lower():
                error_msg = f"Connection timeout"
            elif "name" in error_reason.lower():
                error_msg = f"DNS resolution failed - check internet connection"
            else:
                error_msg = f"Network error: {error_reason}"
            print(f"[HF VALIDATOR] URLError: {error_msg}")
            return -1, {}, error_msg
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"[HF VALIDATOR] Exception: {error_msg}")
            return -2, {}, error_msg
    
    def parse_model_input(self, user_input: str) -> Optional[str]:
        """Parse model ID from URL or direct input."""
        if not user_input or not user_input.strip():
            return None
        
        user_input = user_input.strip()
        
        # Check if it's a full URL
        if 'huggingface.co' in user_input:
            parts = user_input.split('huggingface.co/')
            if len(parts) > 1:
                path = parts[1].split('?')[0]
                path_parts = path.split('/')
                if len(path_parts) >= 2:
                    return f"{path_parts[0]}/{path_parts[1]}"
        
        # Direct ID format
        if '/' in user_input and len(user_input.split('/')) == 2:
            return user_input
        
        return None
    
    def validate_model(self, model_id: str) -> HFModelInfo:
        """Validate model and fetch metadata."""
        try:
            api_url = f"https://huggingface.co/api/models/{model_id}"
            status_code, data, error_msg = self._make_request(api_url)
            
            if status_code == 404:
                return HFModelInfo(
                    model_id=model_id, name="", description="", size_gb=0.0,
                    format="unknown", tags=[], architecture="unknown",
                    mlx_compatible=False, quantized=False, downloads=0,
                    last_modified="", error="Model not found on HuggingFace"
                )
            
            if status_code == 401:
                return HFModelInfo(
                    model_id=model_id, name="", description="", size_gb=0.0,
                    format="unknown", tags=[], architecture="unknown",
                    mlx_compatible=False, quantized=False, downloads=0,
                    last_modified="", error="Authentication required - model may be gated"
                )
            
            if status_code < 0:
                # Network/connection errors
                error_text = error_msg or "Network error"
                return HFModelInfo(
                    model_id=model_id, name="", description="", size_gb=0.0,
                    format="unknown", tags=[], architecture="unknown",
                    mlx_compatible=False, quantized=False, downloads=0,
                    last_modified="", error=f"Connection failed: {error_text}"
                )
            
            if status_code != 200 or not data:
                return HFModelInfo(
                    model_id=model_id, name="", description="", size_gb=0.0,
                    format="unknown", tags=[], architecture="unknown",
                    mlx_compatible=False, quantized=False, downloads=0,
                    last_modified="", error=f"API error: {status_code}"
                )
            
            tags = data.get('tags', [])
            siblings = data.get('siblings', [])
            
            is_mlx = self._check_mlx_compatibility(model_id, tags, siblings)
            is_quantized = self._check_quantized(model_id, tags)
            size_gb = self._estimate_size(siblings)
            architecture = self._detect_architecture(model_id, tags)
            name = self._format_model_name(model_id)
            description = self._extract_description(data)
            
            return HFModelInfo(
                model_id=model_id, name=name, description=description,
                size_gb=size_gb, format=self._detect_format(model_id, tags, is_mlx),
                tags=tags, architecture=architecture, mlx_compatible=is_mlx,
                quantized=is_quantized, downloads=data.get('downloads', 0),
                last_modified=data.get('lastModified', '')
            )
            
        except Exception as e:
            return HFModelInfo(
                model_id=model_id, name="", description="", size_gb=0.0,
                format="unknown", tags=[], architecture="unknown",
                mlx_compatible=False, quantized=False, downloads=0,
                last_modified="", error=f"Error: {str(e)}"
            )
    
    def _check_mlx_compatibility(self, model_id: str, tags: List[str], siblings: List[Dict]) -> bool:
        """Check if model is MLX-compatible."""
        mlx_tags = ['mlx', 'mlx-community', 'apple-silicon']
        has_mlx_tag = any(tag in tags for tag in mlx_tags)
        is_mlx_community = model_id.startswith('mlx-community/')
        filenames = [s.get('rfilename', '') for s in siblings]
        has_safetensors = any(f.endswith('.safetensors') for f in filenames)
        return (has_mlx_tag or is_mlx_community) and has_safetensors
    
    def _check_quantized(self, model_id: str, tags: List[str]) -> bool:
        """Check if model is quantized."""
        keywords = ['4bit', '8bit', 'quantized', '4-bit', '8-bit', 'q4', 'q8']
        id_lower = model_id.lower()
        has_id = any(kw in id_lower for kw in keywords)
        has_tag = any('quant' in tag.lower() for tag in tags)
        return has_id or has_tag
    
    def _estimate_size(self, siblings: List[Dict]) -> float:
        """Estimate model size from file list."""
        weight_files = [s for s in siblings if s.get('rfilename', '').endswith('.safetensors')]
        if not weight_files:
            return 0.0
        num_files = len(weight_files)
        if num_files == 1:
            return 0.8
        elif num_files <= 2:
            return 2.0
        elif num_files <= 4:
            return 4.5
        else:
            return 8.0
    
    def _detect_architecture(self, model_id: str, tags: List[str]) -> str:
        """Detect model architecture."""
        id_lower = model_id.lower()
        arch_map = {'llama': 'llama', 'mistral': 'mistral', 'phi': 'phi', 
                    'qwen': 'qwen', 'gemma': 'gemma'}
        for key, val in arch_map.items():
            if key in id_lower:
                return val
        for tag in tags:
            tag_lower = tag.lower()
            if tag_lower in arch_map.values():
                return tag_lower
        return 'unknown'
    
    def _format_model_name(self, model_id: str) -> str:
        """Format model ID into display name."""
        name = model_id.split('/')[-1] if '/' in model_id else model_id
        name = name.replace('-4bit', '').replace('-8bit', '')
        name = name.replace('-instruct', '').replace('-chat', '')
        name = name.replace('_', ' ').replace('-', ' ')
        return ' '.join(word.capitalize() for word in name.split())
    
    def _extract_description(self, data: Dict) -> str:
        """Extract description from model card."""
        try:
            card_data = data.get('cardData', {})
            if isinstance(card_data, dict):
                desc = card_data.get('description', '')
                if desc:
                    return desc[:200] + '...' if len(desc) > 200 else desc
            tags = data.get('tags', [])
            if tags:
                return f"Tags: {', '.join(tags[:5])}"
            return "No description available"
        except:
            return "No description available"
    
    def _detect_format(self, model_id: str, tags: List[str], is_mlx: bool) -> str:
        """Detect model format."""
        id_lower = model_id.lower()
        if '4bit' in id_lower or '-4bit' in id_lower or '4-bit' in id_lower:
            return 'MLX 4-bit' if is_mlx else '4-bit'
        elif '8bit' in id_lower or '-8bit' in id_lower or '8-bit' in id_lower:
            return 'MLX 8-bit' if is_mlx else '8-bit'
        elif any('quant' in tag.lower() for tag in tags):
            return 'MLX Quantized' if is_mlx else 'Quantized'
        else:
            return 'MLX' if is_mlx else 'Standard'


class UserModelsManager:
    """Manages user's custom models."""
    
    def __init__(self, file_path: str = "user_models.json"):
        self.file_path = Path(file_path)
        self.models = []
        self._load()
    
    def _load(self):
        """Load user models from file."""
        if self.file_path.exists():
            try:
                with open(self.file_path, 'r') as f:
                    data = json.load(f)
                    self.models = data.get('models', [])
            except:
                self.models = []
        else:
            self.models = []
    
    def _save(self):
        """Save user models to file."""
        data = {'models': self.models}
        with open(self.file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_model(self, model_info: HFModelInfo) -> bool:
        """Add a validated model to user's list."""
        existing = [m for m in self.models if m['model_id'] == model_info.model_id]
        if existing:
            return False
        
        model_data = {
            'model_id': model_info.model_id,
            'name': model_info.name,
            'source': 'huggingface',
            'added_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'hf_data': {
                'size_gb': model_info.size_gb,
                'format': model_info.format,
                'tags': model_info.tags,
                'architecture': model_info.architecture,
                'mlx_compatible': model_info.mlx_compatible,
                'quantized': model_info.quantized,
                'downloads': model_info.downloads,
                'last_modified': model_info.last_modified
            }
        }
        
        self.models.append(model_data)
        self._save()
        return True
    
    def remove_model(self, model_id: str) -> bool:
        """Remove a model from user's list."""
        original_count = len(self.models)
        self.models = [m for m in self.models if m['model_id'] != model_id]
        if len(self.models) < original_count:
            self._save()
            return True
        return False
    
    def get_all_models(self) -> List[Dict]:
        """Get all user models."""
        return self.models.copy()
    
    def model_exists(self, model_id: str) -> bool:
        """Check if model already in list."""
        return any(m['model_id'] == model_id for m in self.models)


def format_model_info_for_display(info: HFModelInfo) -> str:
    """Format model info for UI display."""
    if info.error:
        return f"✗ {info.error}"
    
    lines = [
        f"✓ {info.name}",
        "",
        f"📦 Size: {info.size_gb:.1f} GB",
        f"🔧 Format: {info.format}",
    ]
    
    if info.mlx_compatible:
        lines.append("✅ MLX Compatible")
    else:
        lines.append("⚠️ Not optimized for MLX")
    
    if info.tags:
        lines.append(f"🏷️ Tags: {', '.join(info.tags[:5])}")
    
    lines.append("")
    lines.append(f"ℹ️ {info.description}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    validator = HFModelValidator()
    
    test_inputs = [
        "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "invalid-model-id",
    ]
    
    for test_input in test_inputs:
        print(f"\n{'='*60}")
        print(f"Testing: {test_input}")
        print('='*60)
        
        model_id = validator.parse_model_input(test_input)
        if model_id:
            print(f"Parsed ID: {model_id}")
            info = validator.validate_model(model_id)
            print(format_model_info_for_display(info))
        else:
            print("✗ Invalid input format")
