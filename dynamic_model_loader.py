#!/usr/bin/env python3
"""
Dynamic Model Loader for Jetson Nano ML System

This module provides efficient, on-demand model loading to save memory on resource-constrained devices.
Instead of loading all models at startup, models are loaded only when needed based on player/device IDs.

Features:
- On-demand model loading based on player/device IDs
- LRU cache with configurable size to balance memory and performance
- Automatic model path resolution
- CUDA/CPU device management
- Error handling and fallback mechanisms
- Memory usage monitoring
"""

import os
import time
import logging
import numpy as np
import torch
from typing import Dict, Optional, Tuple, Any
from collections import OrderedDict
from hummingbird.ml import load
import psutil

class DynamicModelLoader:
    """
    Dynamic model loader with LRU caching for memory-efficient model management.
    
    This class loads models on-demand based on player/device IDs and maintains
    a configurable cache to balance memory usage and performance.
    """
    
    def __init__(self, 
                 models_dir: str = "athlete_models_tensors_updated",
                 cache_size: int = 5,
                 device: str = "cuda",
                 enable_memory_monitoring: bool = True):
        """
        Initialize the dynamic model loader.
        
        Args:
            models_dir: Directory containing model files
            cache_size: Maximum number of models to keep in memory (LRU cache)
            device: Target device for model inference ("cpu" or "cuda")
            enable_memory_monitoring: Whether to log memory usage statistics
        """
        self.models_dir = models_dir
        self.cache_size = cache_size
        self.device = device
        self.enable_memory_monitoring = enable_memory_monitoring
        
        # LRU cache: OrderedDict where keys are model IDs and values are (model, load_time)
        self.model_cache: OrderedDict[int, Tuple[Any, float]] = OrderedDict()
        
        # Statistics
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "models_loaded": 0,
            "models_evicted": 0,
            "total_load_time": 0.0
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Validate models directory
        if not os.path.exists(models_dir):
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
        
        # Discover available models
        self.available_models = self._discover_available_models()
        self.logger.info(f"Discovered {len(self.available_models)} models in {models_dir}")
        
        # Log initial memory usage
        if self.enable_memory_monitoring:
            self._log_memory_usage("initialization")
    
    def _discover_available_models(self) -> Dict[int, str]:
        """Discover available model files and map them to player IDs."""
        available = {}
        
        try:
            for filename in sorted(os.listdir(self.models_dir)):
                if filename.endswith('.zip') or os.path.isdir(os.path.join(self.models_dir, filename)):
                    # Extract player ID from filename (e.g., "hb_1_model.zip" -> 1)
                    if filename.startswith('hb_') and '_model' in filename:
                        try:
                            # Extract number between 'hb_' and '_model'
                            parts = filename.split('_')
                            if len(parts) >= 3:
                                player_id = int(parts[1])
                                model_path = os.path.join(self.models_dir, filename.replace('.zip', ''))
                                available[player_id] = model_path
                        except (ValueError, IndexError):
                            continue
        
        except Exception as e:
            self.logger.error(f"Error discovering models: {e}")
        
        return available
    
    def _log_memory_usage(self, context: str):
        """Log current memory usage for monitoring."""
        if not self.enable_memory_monitoring:
            return
        
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / (1024 * 1024)
            
            # GPU memory if available
            gpu_memory = None
            if torch.cuda.is_available():
                try:
                    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                    reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                    gpu_memory = f"GPU: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved"
                except Exception:
                    pass
            
            gpu_info = f" | {gpu_memory}" if gpu_memory else ""
            self.logger.info(f"Memory usage ({context}): {memory_mb:.1f}MB{gpu_info} | Cache: {len(self.model_cache)}/{self.cache_size} models")
            # Removed console output for cleaner display
            
        except Exception as e:
            self.logger.warning(f"Failed to log memory usage: {e}")
    
    def _load_model_from_disk(self, player_id: int) -> Optional[Any]:
        """Load a model from disk for the given player ID."""
        if player_id not in self.available_models:
            self.logger.warning(f"No model found for player {player_id}")
            return None
        
        model_path = self.available_models[player_id]
        start_time = time.time()
        
        try:
            self.logger.debug(f"Loading model for player {player_id} from {model_path}")
            # Removed console output for cleaner display
            
            # Load the model
            model = load(model_path, override_flag=True)
            
            # Move to target device if needed
            if self.device == "cuda" and torch.cuda.is_available():
                try:
                    # Test model functionality first
                    test_input = np.random.randn(1, 30).astype(np.float32)
                    cpu_result = model.predict(test_input)
                    
                    # Move to CUDA
                    model.to("cuda")
                    cuda_result = model.predict(test_input)
                    
                    self.logger.debug(f"Model {player_id} loaded successfully on CUDA")
                    # Removed console output for cleaner display
                    
                except Exception as cuda_e:
                    self.logger.warning(f"CUDA loading failed for player {player_id}: {cuda_e}")
                    self.logger.info(f"Falling back to CPU for player {player_id}")
                    try:
                        model.to("cpu")
                    except:
                        pass
            else:
                self.logger.debug(f"Model {player_id} loaded successfully on CPU")
                # Removed console output for cleaner display
            
            load_time = time.time() - start_time
            self.stats["models_loaded"] += 1
            self.stats["total_load_time"] += load_time
            
            self.logger.debug(f"Model {player_id} loaded in {load_time:.2f}s")
            # Removed console output for cleaner display
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model for player {player_id}: {e}")
            return None
    
    def _evict_oldest_model(self):
        """Evict the least recently used model from cache."""
        if not self.model_cache:
            return
        
        # Remove the oldest (first) item
        player_id, (model, load_time) = self.model_cache.popitem(last=False)
        
        # Clean up model resources
        try:
            if hasattr(model, 'to'):
                model.to('cpu')  # Move to CPU before deletion
            del model
        except Exception as e:
            self.logger.warning(f"Error cleaning up model {player_id}: {e}")
        
        self.stats["models_evicted"] += 1
        self.logger.debug(f"Evicted model {player_id} from cache (LRU)")
        #print(f"🗑️ Evicted model {player_id} from cache (LRU)")
    
    def get_model(self, player_id: int) -> Optional[Any]:
        """
        Get a model for the given player ID.
        
        This method implements LRU caching:
        1. Check if model is in cache (cache hit)
        2. If not in cache, load from disk (cache miss)
        3. If cache is full, evict oldest model
        4. Add new model to cache
        
        Args:
            player_id: Player/device ID to get model for
            
        Returns:
            Loaded model or None if loading failed
        """
        # Check cache first
        if player_id in self.model_cache:
            # Cache hit - move to end (most recently used)
            model, load_time = self.model_cache.pop(player_id)
            self.model_cache[player_id] = (model, load_time)
            self.stats["cache_hits"] += 1
            
            self.logger.debug(f"Cache hit for player {player_id}")
            return model
        
        # Cache miss - load from disk
        self.stats["cache_misses"] += 1
        self.logger.debug(f"Cache miss for player {player_id}")
        
        model = self._load_model_from_disk(player_id)
        if model is None:
            return None
        
        # Check if cache is full
        if len(self.model_cache) >= self.cache_size:
            self._evict_oldest_model()
        
        # Add to cache
        self.model_cache[player_id] = (model, time.time())
        
        # Log memory usage after loading
        if self.enable_memory_monitoring:
            self._log_memory_usage(f"after loading player {player_id}")
        
        return model
    
    def preload_models(self, player_ids: list, max_concurrent: int = 3):
        """
        Preload models for multiple players to warm up the cache.
        
        Args:
            player_ids: List of player IDs to preload
            max_concurrent: Maximum number of models to load concurrently
        """
        self.logger.info(f"Preloading models for {len(player_ids)} players")
        
        for i, player_id in enumerate(player_ids):
            if len(self.model_cache) >= self.cache_size:
                self.logger.info(f"Cache full, stopping preload at player {player_id}")
                break
            
            self.get_model(player_id)
            
            # Log progress
            if (i + 1) % max_concurrent == 0:
                self._log_memory_usage(f"preload progress ({i + 1}/{len(player_ids)})")
    
    def clear_cache(self):
        """Clear all models from cache to free memory."""
        self.logger.info("Clearing model cache")
        
        for player_id, (model, _) in self.model_cache.items():
            try:
                if hasattr(model, 'to'):
                    model.to('cpu')
                del model
            except Exception as e:
                self.logger.warning(f"Error cleaning up model {player_id}: {e}")
        
        self.model_cache.clear()
        
        if self.enable_memory_monitoring:
            self._log_memory_usage("after cache clear")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the current cache state."""
        cache_hit_rate = 0.0
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        if total_requests > 0:
            cache_hit_rate = self.stats["cache_hits"] / total_requests
        
        avg_load_time = 0.0
        if self.stats["models_loaded"] > 0:
            avg_load_time = self.stats["total_load_time"] / self.stats["models_loaded"]
        
        return {
            "cache_size": len(self.model_cache),
            "max_cache_size": self.cache_size,
            "cached_models": list(self.model_cache.keys()),
            "cache_hit_rate": cache_hit_rate,
            "total_requests": total_requests,
            "models_loaded": self.stats["models_loaded"],
            "models_evicted": self.stats["models_evicted"],
            "average_load_time": avg_load_time,
            "available_models": len(self.available_models)
        }
    
    def get_available_player_ids(self) -> list:
        """Get list of player IDs for which models are available."""
        return sorted(self.available_models.keys())
    
    def is_model_available(self, player_id: int) -> bool:
        """Check if a model is available for the given player ID."""
        return player_id in self.available_models


# Convenience function for backward compatibility
def create_dynamic_model_loader(cache_size: int = 5, device: str = "cpu") -> DynamicModelLoader:
    """
    Create a dynamic model loader with default settings.
    
    Args:
        cache_size: Maximum number of models to keep in memory
        device: Target device for model inference
        
    Returns:
        Configured DynamicModelLoader instance
    """
    return DynamicModelLoader(
        models_dir="athlete_models_tensors_updated",
        cache_size=cache_size,
        device=device,
        enable_memory_monitoring=True
    )


if __name__ == "__main__":
    # Example usage and testing
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create loader
    loader = create_dynamic_model_loader(cache_size=3, device="cpu")
    
    # Test loading some models
    test_players = [1, 2, 3, 4, 5]
    
    print("Testing dynamic model loading...")
    for player_id in test_players:
        model = loader.get_model(player_id)
        if model:
            print(f"✅ Model for player {player_id} loaded successfully")
        else:
            print(f"❌ Failed to load model for player {player_id}")
    
    # Show cache info
    info = loader.get_cache_info()
    print(f"\nCache Info:")
    print(f"  Cache size: {info['cache_size']}/{info['max_cache_size']}")
    print(f"  Cached models: {info['cached_models']}")
    print(f"  Cache hit rate: {info['cache_hit_rate']:.2%}")
    print(f"  Available models: {info['available_models']}")
    
    # Test cache eviction
    print(f"\nTesting cache eviction...")
    for player_id in [6, 7, 8]:
        model = loader.get_model(player_id)
        if model:
            print(f"✅ Model for player {player_id} loaded (should evict oldest)")
    
    # Final cache info
    info = loader.get_cache_info()
    print(f"\nFinal Cache Info:")
    print(f"  Cached models: {info['cached_models']}")
    print(f"  Models evicted: {info['models_evicted']}")
