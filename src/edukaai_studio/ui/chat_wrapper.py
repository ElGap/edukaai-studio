"""
Chat Wrapper Module

Wraps the MLX-LM chat functionality for the UI.
Uses model's tokenizer to ensure proper chat format.
"""

import subprocess
import json
import re
from pathlib import Path
from typing import List, Dict, Optional

# Import configuration from edukaai_studio package
from edukaai_studio.config import UI

class ChatWrapper:
    """Handles chat interactions with the trained model using proper chat format."""
    
    def __init__(self, model_path: str, adapter_path: Optional[str] = None):
        """
        Initialize chat wrapper.
        
        Args:
            model_path: Path to base model or fused model
            adapter_path: Path to LoRA adapter directory or file (if not using fused model)
                       If file path is given (e.g., .../adapters.safetensors), parent directory is used
        """
        self.model_path = model_path
        
        # Fix adapter path - mlx_lm expects directory, not file
        if adapter_path:
            adapter_path_obj = Path(adapter_path)
            if adapter_path_obj.is_file():
                # If file path given (e.g., .../adapters.safetensors), use parent directory
                self.adapter_path = str(adapter_path_obj.parent)
                print(f"[CHAT] Adapter file path detected, using directory: {self.adapter_path}")
            else:
                # Already a directory path
                self.adapter_path = adapter_path
                print(f"[CHAT] Using adapter directory: {self.adapter_path}")
        else:
            self.adapter_path = None
        
        self.conversation_history: List[Dict[str, str]] = []
        self.system_prompt = "You are a helpful assistant trained on specific data. Answer questions accurately based on your training."
        
        # Store last chat metrics for accurate stats
        self.last_metrics = {
            'prompt_tokens': 0,
            'generation_tokens': 0,
            'peak_memory_gb': 0.0,
            'prompt_tps': 0.0,
            'generation_tps': 0.0
        }
        
        # Load tokenizer to get proper chat format
        self.tokenizer = None
        self.has_chat_template = False
        self.eos_token = "<|end|>"  # Default for Phi-3
        self.special_tokens = []  # Will be populated from tokenizer
        self._load_tokenizer()
    
    def _load_tokenizer(self):
        """Load tokenizer to get proper chat format, EOS token, and special tokens."""
        try:
            from transformers import AutoTokenizer
            print(f"[CHAT] Loading tokenizer from {self.model_path}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Get EOS token
            if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token:
                self.eos_token = self.tokenizer.eos_token
                print(f"[CHAT] Using EOS token: {self.eos_token}")
            
            # Get all special tokens from tokenizer
            self.special_tokens = []
            if hasattr(self.tokenizer, 'all_special_tokens'):
                self.special_tokens = list(self.tokenizer.all_special_tokens)
                print(f"[CHAT] Loaded {len(self.special_tokens)} special tokens from tokenizer")
            elif hasattr(self.tokenizer, 'special_tokens_map'):
                # Fallback: extract from special_tokens_map
                special_map = self.tokenizer.special_tokens_map
                for key in ['eos_token', 'bos_token', 'unk_token', 'pad_token', 'cls_token', 'sep_token', 'mask_token']:
                    if key in special_map and special_map[key]:
                        token = special_map[key]
                        if isinstance(token, str) and token not in self.special_tokens:
                            self.special_tokens.append(token)
                print(f"[CHAT] Loaded {len(self.special_tokens)} special tokens from special_tokens_map")
            
            # Always add common formatting markers that aren't model-specific
            common_formatting = ["==========", "---", "###"]
            for marker in common_formatting:
                if marker not in self.special_tokens:
                    self.special_tokens.append(marker)
            
            # Check for chat template
            if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
                self.has_chat_template = True
                print(f"[CHAT] Model has chat template - using apply_chat_template")
            else:
                print(f"[CHAT] No chat template found - using fallback formatting")
                
        except Exception as e:
            print(f"[CHAT] Error loading tokenizer: {e}")
            print(f"[CHAT] Using default format (may not be optimal)")
            self.tokenizer = None
            # Fallback special tokens
            self.special_tokens = ["<|endoftext|>", "<|assistant|>", "<|user|>", "<|system|>", 
                                  "<|im_start|>", "<|end|>", "oes", "</s>", "---", "###", 
                                  "[INST]", "[/INST]", "<s>", "</s>", "<bos>", "<eos>", "=========="]
    
    def _build_prompt(self, message: str) -> str:
        """Build prompt using model's chat template if available."""
        if self.tokenizer and self.has_chat_template:
            # Build messages list
            messages = []
            
            # Try with system prompt first
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            
            # Add conversation history
            for h in self.conversation_history[-5:]:
                messages.append({"role": "user", "content": h['user']})
                messages.append({"role": "assistant", "content": h['assistant']})
            
            # Add current message
            messages.append({"role": "user", "content": message})
            
            # Use tokenizer's chat template
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True  # Add assistant prefix
                )
                return prompt
            except Exception as e:
                error_str = str(e).lower()
                if "roles must alternate" in error_str or "conversation roles" in error_str:
                    # Model doesn't support system role (e.g., Mistral)
                    # Retry without system prompt, prepend to first user message instead
                    print(f"[CHAT] Model doesn't support system role, retrying without it...")
                    messages_no_system = []
                    
                    # Add conversation history
                    for h in self.conversation_history[-5:]:
                        messages_no_system.append({"role": "user", "content": h['user']})
                        messages_no_system.append({"role": "assistant", "content": h['assistant']})
                    
                    # Add current message with system prepended if no history
                    if not self.conversation_history and self.system_prompt:
                        user_content = f"{self.system_prompt}\n\n{message}"
                    else:
                        user_content = message
                    
                    messages_no_system.append({"role": "user", "content": user_content})
                    
                    try:
                        prompt = self.tokenizer.apply_chat_template(
                            messages_no_system,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        return prompt
                    except Exception as e2:
                        print(f"[CHAT] Error applying chat template (retry): {e2}")
                else:
                    print(f"[CHAT] Error applying chat template: {e}")
                # Fall through to fallback
        
        # Fallback: use basic format
        return self._fallback_build_prompt(message)
    
    def _fallback_build_prompt(self, message: str) -> str:
        """Fallback prompt builder when tokenizer unavailable."""
        prompt_parts = []
        
        # Add system prompt
        if self.system_prompt:
            prompt_parts.append(f"System: {self.system_prompt}")
        
        # Add conversation history (last 5 exchanges)
        for h in self.conversation_history[-5:]:
            prompt_parts.append(f"User: {h['user']}")
            prompt_parts.append(f"Assistant: {h['assistant']}")
        
        # Add current message
        prompt_parts.append(f"User: {message}")
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
         
    def chat(self, message: str, system_prompt: str = None, temperature: float = UI.DEFAULT_CHAT_TEMPERATURE, top_p: float = None, max_tokens: int = UI.DEFAULT_MAX_TOKENS) -> str:
        """
        Send a message and get response.
        
        Args:
            message: User's message
            system_prompt: Optional system prompt to override default
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling parameter (0.0-1.0)
            max_tokens: Maximum response length
            
        Returns:
            str: Model's response
        """
        try:
            # Update system prompt if provided
            if system_prompt is not None:
                original_system = self.system_prompt
                self.system_prompt = system_prompt
                print(f"[CHAT] Using custom system prompt: {system_prompt[:50]}...")
            
            # Build prompt with history
            prompt = self._build_prompt(message)
            print(f"[CHAT] Built prompt: {repr(prompt[:UI.PREVIEW_TEXT_LENGTH])}...")
            
            # Prepare command
            cmd = [
                "python3", "-m", "mlx_lm", "generate",
                "--model", self.model_path,
                "--prompt", prompt,
                "--temp", str(temperature),
                "--max-tokens", str(max_tokens)
            ]
            
            # Add top_p if provided
            if top_p is not None and top_p < 1.0:
                cmd.extend(["--top-p", str(top_p)])
                print(f"[CHAT] Using top_p: {top_p}")
            
            print(f"[CHAT] Model path: {self.model_path}")
            print(f"[CHAT] Has chat template: {self.has_chat_template}")
            print(f"[CHAT] EOS token: {self.eos_token}")
            
            if self.adapter_path:
                cmd.extend(["--adapter-path", self.adapter_path])
                print(f"[CHAT] Using adapter: {self.adapter_path}")
            
            # Use extra-eos-token to stop generation at EOS token
            # Only for models without chat templates - models with chat templates
            # should stop correctly via the template format
            if self.eos_token and not self.has_chat_template:
                cmd.extend(["--extra-eos-token", self.eos_token])
                print(f"[CHAT] Using extra EOS token: {self.eos_token}")
            elif self.has_chat_template:
                print(f"[CHAT] Using chat template - skipping extra EOS token")
            
            # Generate response
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=UI.CHAT_TIMEOUT_SECONDS  # Configurable timeout
            )
            
            if result.returncode != 0:
                error_msg = f"Error: {result.stderr}"
                print(error_msg)
                return "Sorry, I encountered an error generating a response."
            
            response = result.stdout.strip()
            
            # Extract metrics from mlx_lm output before cleaning
            self.last_metrics = self._extract_metrics(response)
            print(f"[CHAT] Metrics: {self.last_metrics['generation_tokens']} tokens, "
                  f"{self.last_metrics['peak_memory_gb']:.2f}GB memory, "
                  f"{self.last_metrics['generation_tps']:.1f} tps")
            
            # CRITICAL: Truncate at <unk> token if present
            if "<unk>" in response:
                print(f"[CHAT] Warning: <unk> detected, truncating")
                response = response.split("<unk>")[0].strip()
            
            # COMPREHENSIVE RESPONSE CLEANING for mlx_lm output
            
            # 1. Remove the entire prompt if present
            full_prompt = self._build_prompt(message)
            if full_prompt in response:
                response = response.replace(full_prompt, "").strip()
            
            # 2. Remove common instruction/role prefixes
            prefixes_to_remove = [
                "Assistant:",
                "AI:",
                "Bot:",
                "Response:",
                "Answer:",
                "Output:",
            ]
            for prefix in prefixes_to_remove:
                if response.startswith(prefix):
                    response = response[len(prefix):].strip()
            
            # 3. Split by "Assistant:" and take the last clean response
            if "Assistant:" in response:
                parts = response.split("Assistant:")
                # Find the part that looks like a response (not empty, not just formatting)
                for part in reversed(parts):
                    clean_part = part.strip()
                    if clean_part and len(clean_part) > 5:  # Must be substantial
                        response = clean_part
                        break
            
            # 4. Remove special tokens and markers (loaded from model tokenizer)
            for token in self.special_tokens:
                response = response.replace(token, "")
            
            # 5. Clean up lines that look like prompts or system messages
            lines = response.split('\n')
            clean_lines = []
            skip_patterns = [
                'System:',
                'User:',
                'Human:',
                'Instruction:',
                'Context:',
                'Input:',
            ]
            for line in lines:
                stripped = line.strip()
                # Skip empty lines and lines that look like prompts
                if stripped and not any(stripped.startswith(pattern) for pattern in skip_patterns):
                    clean_lines.append(line)
            
            response = '\n'.join(clean_lines).strip()
            
            # 6. Remove echoed user message if it appears at the start
            if message.lower() in response[:len(message)+20].lower():
                # Find where the user message ends and response begins
                idx = response.lower().find(message.lower()) + len(message)
                if idx < len(response):
                    response = response[idx:].strip()
            
            # 7. Remove performance metric lines from mlx_lm output
            performance_patterns = [
                r'^Prompt:\s+\d+\s+tokens,',
                r'^Generation:\s+\d+\s+tokens,',
                r'^Peak memory:\s+[\d.]+\s*GB',
                r'.*tokens-per-sec.*',
            ]
            lines = response.split('\n')
            clean_lines = []
            for line in lines:
                stripped = line.strip()
                # Skip lines matching performance patterns
                if not any(re.search(pattern, stripped) for pattern in performance_patterns):
                    clean_lines.append(line)
            response = '\n'.join(clean_lines).strip()
            
            # 8. Remove trailing artifacts
            response = response.rstrip('-').rstrip('=').rstrip('*').strip()
            
            # 8. Final cleanup - ensure no weird unicode or control characters
            response = ''.join(char for char in response if ord(char) >= 32 or char in '\n\t')
            
            # Add to history
            self.conversation_history.append({
                "user": message,
                "assistant": response
            })
            
            # Restore original system prompt if it was temporarily changed
            if system_prompt is not None:
                self.system_prompt = original_system
            
            return response
            
        except subprocess.TimeoutExpired:
            # Restore original system prompt on error too
            if system_prompt is not None:
                self.system_prompt = original_system
            return "Response generation timed out. Please try a shorter message."
        except Exception as e:
            # Restore original system prompt on error too
            if system_prompt is not None:
                self.system_prompt = original_system
            return f"Error: {str(e)}"
    
    def set_system_prompt(self, new_prompt: str):
        """
        Update the system prompt for the chat.
        
        Args:
            new_prompt: New system prompt string
        """
        self.system_prompt = new_prompt
        print(f"[CHAT] System prompt updated: {new_prompt[:100]}...")
    
    def _extract_metrics(self, raw_output: str) -> dict:
        """
        Extract performance metrics from mlx_lm generate output.
        
        Args:
            raw_output: Raw stdout from mlx_lm generate command
            
        Returns:
            dict: Extracted metrics (prompt_tokens, generation_tokens, etc.)
        """
        metrics = {
            'prompt_tokens': 0,
            'generation_tokens': 0,
            'peak_memory_gb': 0.0,
            'prompt_tps': 0.0,
            'generation_tps': 0.0
        }
        
        # Extract prompt tokens and speed
        # Format: "Prompt: 34 tokens, 318.490 tokens-per-sec"
        prompt_match = re.search(r'Prompt:\s+(\d+)\s+tokens,\s+([\d.]+)\s+tokens-per-sec', raw_output)
        if prompt_match:
            metrics['prompt_tokens'] = int(prompt_match.group(1))
            metrics['prompt_tps'] = float(prompt_match.group(2))
        
        # Extract generation tokens and speed
        # Format: "Generation: 431 tokens, 131.587 tokens-per-sec"
        gen_match = re.search(r'Generation:\s+(\d+)\s+tokens,\s+([\d.]+)\s+tokens-per-sec', raw_output)
        if gen_match:
            metrics['generation_tokens'] = int(gen_match.group(1))
            metrics['generation_tps'] = float(gen_match.group(2))
        
        # Extract peak memory
        # Format: "Peak memory: 2.499 GB"
        mem_match = re.search(r'Peak memory:\s+([\d.]+)\s*GB', raw_output)
        if mem_match:
            metrics['peak_memory_gb'] = float(mem_match.group(1))
        
        return metrics
    
    def get_last_metrics(self) -> dict:
        """Get metrics from the last chat generation."""
        return self.last_metrics.copy()
    
    def reset(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.conversation_history.copy()
    
    def save_conversation(self, filepath: str):
        """Save conversation to file."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("Conversation Log\n")
                f.write("=" * 80 + "\n\n")
                
                for i, h in enumerate(self.conversation_history, 1):
                    f.write(f"[{i}] User: {h['user']}\n")
                    f.write(f"    Assistant: {h['assistant']}\n\n")
            
            return True
        except Exception as e:
            print(f"Error saving conversation: {e}")
            return False
    
    def load_from_training(self, output_dir: str) -> bool:
        """
        Load model from training output directory.
        
        Args:
            output_dir: Path to training output directory
            
        Returns:
            bool: True if model loaded successfully
        """
        output_path = Path(output_dir)
        
        # Try fused model first
        fused_model = output_path / "fused_model"
        if fused_model.exists() and (fused_model / "model.safetensors").exists():
            self.model_path = str(fused_model)
            self.adapter_path = None
            return True
        
        # Try adapter format
        best_adapter = output_path / "best_adapter"
        if best_adapter.exists() and (best_adapter / "adapters.safetensors").exists():
            self.adapter_path = str(best_adapter)
            # Need to get base model from summary
            summary_file = output_path / "training_summary.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    data = json.load(f)
                    self.model_path = data.get('model', self.model_path)
            return True
        
        return False
