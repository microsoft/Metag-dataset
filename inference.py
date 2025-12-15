# This file contains classes and functions to run inference with a language model
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch
import yaml
from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm


@dataclass
class InferenceConfig:
    """Configuration for LLM inference."""
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 2048
    prompts: list[str] = field(default_factory=list)
    device: str = "auto"
    torch_dtype: str = "auto"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "InferenceConfig":
        """Load configuration from a YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, config_path: str) -> "InferenceConfig":
        """Load configuration from a JSON file."""
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "InferenceConfig":
        """Load configuration from a dictionary."""
        return cls(**config_dict)


class BaseInference(ABC):
    """Abstract base class for LLM inference."""
    
    def __init__(self, config: InferenceConfig | str | dict):
        """
        Initialize the inference engine.
        
        Args:
            config: Either an InferenceConfig object, a path to a config file,
                   or a dictionary containing configuration parameters.
        """
        if isinstance(config, str):
            if config.endswith(".yaml") or config.endswith(".yml"):
                self.config = InferenceConfig.from_yaml(config)
            else:
                self.config = InferenceConfig.from_json(config)
        elif isinstance(config, dict):
            self.config = InferenceConfig.from_dict(config)
        else:
            self.config = config
    
    @abstractmethod
    def generate(self, prompts: Optional[list[str]] = None) -> list[str]:
        """Generate responses for the given prompts."""
        pass
    
    def generate_single(self, prompt: str) -> str:
        """
        Generate a response for a single prompt.
        
        Args:
            prompt: The prompt to generate a response for.
        
        Returns:
            The generated response.
        """
        return self.generate([prompt])[0]


class HuggingFaceInference(BaseInference):
    """Class to perform inference with an LLM using HuggingFace Transformers."""
    
    def __init__(self, config: InferenceConfig | str | dict):
        """
        Initialize the HuggingFace inference engine.
        
        Args:
            config: Either an InferenceConfig object, a path to a config file,
                   or a dictionary containing configuration parameters.
        """
        super().__init__(config)
        
        # Parse torch dtype
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, "auto")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch_dtype,
            device_map=self.config.device,
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, prompts: Optional[list[str]] = None) -> list[str]:
        """
        Generate responses for the given prompts.
        
        Args:
            prompts: List of prompts to generate responses for.
                    If None, uses prompts from the config.
        
        Returns:
            List of generated responses.
        """
        if prompts is None:
            prompts = self.config.prompts
        
        if not prompts:
            raise ValueError("No prompts provided for inference.")
        
        responses = []
        for prompt in tqdm(prompts, desc="Generating responses"):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=self.config.temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            # Decode only the generated tokens (exclude input)
            generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(response)
        
        return responses
    
    def generate_batch(self, prompts: Optional[list[str]] = None, batch_size: int = 8) -> list[str]:
        """
        Generate responses for prompts using batched inference with HuggingFace pipeline.
        
        Args:
            prompts: List of prompts to generate responses for.
                    If None, uses prompts from the config.
            batch_size: Number of prompts to process in each batch.
        
        Returns:
            List of generated responses.
        """
        if prompts is None:
            prompts = self.config.prompts
        
        if not prompts:
            raise ValueError("No prompts provided for inference.")
        
        # Create text generation pipeline
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
        )
        
        # Generate responses with batching and progress tracking
        responses = []
        for output in tqdm(
            pipe(
                prompts,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                return_full_text=False,  # Only return generated text, not the prompt
            ),
            total=len(prompts),
            desc="Generating responses (batched)",
        ):
            responses.append(output[0]["generated_text"])
        
        return responses
    
    def update_sampling_params(
        self,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        """
        Update the sampling parameters for generation.
        
        Args:
            temperature: New temperature value.
            top_p: New top_p value.
            max_tokens: New max_tokens value.
        """
        if temperature is not None:
            self.config.temperature = temperature
        if top_p is not None:
            self.config.top_p = top_p
        if max_tokens is not None:
            self.config.max_tokens = max_tokens


class UnslothInference(BaseInference):
    """Class to perform inference with an LLM using the Unsloth library for faster inference."""
    
    def __init__(self, config: InferenceConfig | str | dict):
        """
        Initialize the Unsloth inference engine.
        
        Args:
            config: Either an InferenceConfig object, a path to a config file,
                   or a dictionary containing configuration parameters.
        """
        super().__init__(config)
        
        from unsloth import FastLanguageModel
        
        # Load model with Unsloth optimizations
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_tokens,
            dtype=None,  # Auto-detect dtype
            load_in_4bit=True,  # Use 4-bit quantization for efficiency
        )
        import bpdb; bpdb.set_trace()  
        
        # Enable faster inference mode
        FastLanguageModel.for_inference(self.model)
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, prompts: Optional[list[str]] = None) -> list[str]:
        """
        Generate responses for the given prompts.
        
        Args:
            prompts: List of prompts to generate responses for.
                    If None, uses prompts from the config.
        
        Returns:
            List of generated responses.
        """
        if prompts is None:
            prompts = self.config.prompts
        
        if not prompts:
            raise ValueError("No prompts provided for inference.")
        
        responses = []
        for prompt in tqdm(prompts, desc="Generating responses (Unsloth)"):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=self.config.temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                )
            
            # Decode only the generated tokens (exclude input)
            generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(response)
        
        return responses
    
    def generate_batch(self, prompts: Optional[list[str]] = None, batch_size: int = 8) -> list[str]:
        """
        Generate responses for prompts in batches.
        
        Args:
            prompts: List of prompts to generate responses for.
                    If None, uses prompts from the config.
            batch_size: Number of prompts to process in each batch.
        
        Returns:
            List of generated responses.
        """
        if prompts is None:
            prompts = self.config.prompts
        
        if not prompts:
            raise ValueError("No prompts provided for inference.")
        
        responses = []
        num_batches = (len(prompts) + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="Generating responses (Unsloth batched)"):
            batch_prompts = prompts[i * batch_size : (i + 1) * batch_size]
            
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=self.config.temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                )
            
            # Decode each output in the batch
            for j, output in enumerate(outputs):
                input_length = (inputs["attention_mask"][j] == 1).sum()
                generated_tokens = output[input_length:]
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                responses.append(response)
        
        return responses
    
    def update_sampling_params(
        self,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        """
        Update the sampling parameters for generation.
        
        Args:
            temperature: New temperature value.
            top_p: New top_p value.
            max_tokens: New max_tokens value.
        """
        if temperature is not None:
            self.config.temperature = temperature
        if top_p is not None:
            self.config.top_p = top_p
        if max_tokens is not None:
            self.config.max_tokens = max_tokens


if __name__ == "__main__":
    # Example usage with a config file
    # config = InferenceConfig.from_yaml("config.yaml")
    
    # Example usage with a dictionary
    config = InferenceConfig(
        model_name="meta-llama/Llama-2-7b-chat-hf",
        temperature=0.7,
        top_p=0.95,
        max_tokens=512,
        prompts=[
            "What is machine learning?",
            "Explain neural networks in simple terms.",
        ]
    )
    
    # Initialize the inference engine
    inference = HuggingFaceInference(config)
    
    # Generate responses for prompts in config
    responses = inference.generate()
    for prompt, response in zip(config.prompts, responses):
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print("-" * 50)
