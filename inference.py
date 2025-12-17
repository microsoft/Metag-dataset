# This file contains classes and functions to run inference with a language model
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch
import yaml
# from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm


@dataclass
class InferenceConfig:
    """Configuration for LLM inference."""
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 2048
    batch_size: int = 32
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
            inputs = self.tokenizer(text=prompt, return_tensors="pt").to(self.model.device)
            
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


class VLLMInference(BaseInference):
    """Class to perform inference with an LLM using vLLM Docker server for high-throughput serving."""
    
    def __init__(
        self,
        config: InferenceConfig | str | dict,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        port: int = 8000,
        hf_token: Optional[str] = None,
        auto_start_server: bool = True,
    ):
        """
        Initialize the vLLM inference engine with Docker-based server.
        
        Args:
            config: Either an InferenceConfig object, a path to a config file,
                   or a dictionary containing configuration parameters.
            tensor_parallel_size: Number of GPUs to use for tensor parallelism.
            gpu_memory_utilization: Fraction of GPU memory to use (0.0 to 1.0).
            port: Port to expose the vLLM server on.
            hf_token: HuggingFace token for accessing gated models. If None, uses HF_TOKEN env var.
            auto_start_server: Whether to automatically start the Docker server.
        """
        super().__init__(config)
        
        import os
        import subprocess
        import time
        from openai import OpenAI
        
        self.port = port
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self.container_name = f"vllm-server-{port}"
        self._container_id = None
        
        # Initialize OpenAI client pointing to local vLLM server
        self.client = OpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="not-needed"  # vLLM doesn't require a key by default
        )
        
        if auto_start_server:
            self.start_server()
    
    def start_server(self, wait_for_ready: bool = True, timeout: int = 300) -> None:
        """
        Start the vLLM Docker server.
        
        Args:
            wait_for_ready: Whether to wait for the server to be ready.
            timeout: Maximum time to wait for server to be ready (seconds).
        """
        import subprocess
        import time
        
        # Stop any existing container with the same name
        self.stop_server()
        
        # Build the docker run command
        cmd = [
            "docker", "run",
            "-d",  # Run in detached mode
            "--runtime", "nvidia",
            "--gpus", "all",
            "-v", f"{self._get_hf_cache_dir()}:/root/.cache/huggingface",
            "-e", f"HUGGING_FACE_HUB_TOKEN={self.hf_token}",
            "-p", f"{self.port}:8000",
            "--ipc=host",
            "--name", self.container_name,
            "vllm/vllm-openai:latest",
            "--model", self.config.model_name,
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
        ]
        
        print(f"Starting vLLM Docker server with model: {self.config.model_name}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start vLLM Docker container: {result.stderr}")
        
        self._container_id = result.stdout.strip()
        print(f"Container started with ID: {self._container_id[:12]}")
        
        if wait_for_ready:
            self._wait_for_server_ready(timeout)
    
    def _get_hf_cache_dir(self) -> str:
        """Get the HuggingFace cache directory."""
        import os
        return os.path.expanduser("~/.cache/huggingface")
    
    def _wait_for_server_ready(self, timeout: int = 300) -> None:
        """
        Wait for the vLLM server to be ready to accept requests.
        
        Args:
            timeout: Maximum time to wait in seconds.
        """
        import time
        import requests
        
        start_time = time.time()
        print("Waiting for vLLM server to be ready...")
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://localhost:{self.port}/health")
                if response.status_code == 200:
                    print("vLLM server is ready!")
                    return
            except requests.exceptions.ConnectionError:
                pass
            
            # Check if container is still running
            import subprocess
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", self.container_name],
                capture_output=True, text=True
            )
            if result.stdout.strip() != "true":
                # Get container logs for debugging
                logs = subprocess.run(
                    ["docker", "logs", "--tail", "50", self.container_name],
                    capture_output=True, text=True
                )
                raise RuntimeError(f"vLLM container stopped unexpectedly. Logs:\n{logs.stdout}\n{logs.stderr}")
            
            time.sleep(5)
        
        raise TimeoutError(f"vLLM server did not become ready within {timeout} seconds")
    
    def stop_server(self) -> None:
        """Stop and remove the vLLM Docker container."""
        import subprocess
        
        # Stop the container if running
        subprocess.run(
            ["docker", "stop", self.container_name],
            capture_output=True, text=True
        )
        # Remove the container
        subprocess.run(
            ["docker", "rm", self.container_name],
            capture_output=True, text=True
        )
        self._container_id = None
    
    def generate(self, prompts: Optional[list[str]] = None, batch_size: int = 32) -> list[str]:
        """
        Generate responses for the given prompts using batched requests.
        
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
        
        # Batch prompts for efficiency
        for batch_start in tqdm(range(0, len(prompts), batch_size), desc="Generating responses"):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            
            response = self.client.completions.create(
                model=self.config.model_name,
                prompt=batch_prompts,  # Pass list of prompts
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            
            for choice in response.choices:
                responses.append(choice.text)
        
        return responses
    
    def generate_single(self, prompt: str) -> str:
        """
        Generate a response for a single prompt.
        
        Args:
            prompt: The prompt to generate a response for.
        
        Returns:
            The generated response.
        """
        response = self.client.completions.create(
            model=self.config.model_name,
            prompt=prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        return response.choices[0].text
    
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
    
    def __del__(self):
        """Cleanup: stop the Docker container when the object is destroyed."""
        # Note: This may not always be called, so explicit stop_server() is recommended
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: stop the server."""
        self.stop_server()
        return False


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
