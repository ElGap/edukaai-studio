#!/usr/bin/env python3
"""
Verify predefined models exist on HuggingFace.
Run this to check all curated models are valid.
"""

import urllib.request
import ssl
import json
import sys

# Create SSL context that doesn't verify certificates (for testing)
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# Predefined models from config.py
PREDEFINED_MODELS = [
    ("phi-3-mini", "mlx-community/Phi-3-mini-4k-instruct-4bit"),
    ("mistral-7b", "mlx-community/Mistral-7B-Instruct-v0.3-4bit"),
    ("llama-3.2-3b", "mlx-community/Llama-3.2-3B-Instruct-4bit"),
    ("qwen-2.5-7b", "mlx-community/Qwen2.5-7B-Instruct-4bit"),
    ("gemma-3-4b", "mlx-community/gemma-3-4b-it-4bit"),
]


def check_model_exists(model_id):
    """Check if model exists on HuggingFace."""
    try:
        url = f"https://huggingface.co/api/models/{model_id}"
        req = urllib.request.Request(url, headers={
            'User-Agent': 'EdukaAI-Model-Validator/1.0'
        })
        
        with urllib.request.urlopen(req, timeout=30, context=ssl_context) as response:
            data = json.loads(response.read().decode('utf-8'))
            
            # Check for MLX-specific indicators
            tags = data.get('tags', [])
            siblings = data.get('siblings', [])
            
            has_mlx_tag = any(tag in ['mlx', 'mlx-community'] for tag in tags)
            is_mlx_community = model_id.startswith('mlx-community/')
            has_safetensors = any(s.get('rfilename', '').endswith('.safetensors') for s in siblings)
            
            is_mlx_compatible = (has_mlx_tag or is_mlx_community) and has_safetensors
            
            return {
                'exists': True,
                'mlx_compatible': is_mlx_compatible,
                'tags': tags[:5],  # First 5 tags
                'downloads': data.get('downloads', 0),
                'last_modified': data.get('lastModified', 'N/A'),
                'error': None
            }
            
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {'exists': False, 'error': 'Not found on HuggingFace'}
        return {'exists': False, 'error': f'HTTP {e.code}'}
    except Exception as e:
        return {'exists': False, 'error': str(e)}


def main():
    print("=" * 70)
    print("VERIFYING PREDEFINED MODELS ON HUGGINGFACE")
    print("=" * 70)
    print()
    
    all_valid = True
    results = []
    
    for model_key, model_id in PREDEFINED_MODELS:
        print(f"Checking: {model_id}...")
        result = check_model_exists(model_id)
        results.append((model_key, model_id, result))
        
        if result['exists']:
            status = "✓ EXISTS"
            mlx_status = "✓ MLX" if result['mlx_compatible'] else "⚠ Non-MLX"
            print(f"  {status} | {mlx_status} | Downloads: {result['downloads']:,}")
            if result['tags']:
                print(f"  Tags: {', '.join(result['tags'])}")
        else:
            print(f"  ✗ FAILED: {result['error']}")
            all_valid = False
        print()
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    valid_count = sum(1 for _, _, r in results if r['exists'])
    mlx_count = sum(1 for _, _, r in results if r['exists'] and r['mlx_compatible'])
    
    print(f"Total models: {len(PREDEFINED_MODELS)}")
    print(f"Valid models: {valid_count}")
    print(f"MLX compatible: {mlx_count}")
    print()
    
    if all_valid:
        print("✅ All predefined models are valid!")
        return 0
    else:
        print("⚠️ Some models failed validation:")
        for model_key, model_id, result in results:
            if not result['exists']:
                print(f"  - {model_id}: {result['error']}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
