"""
Decorators utilities for Mousa Trading Bot
Advanced decorators for logging, timing, retry mechanisms, and validation
"""

import time
import functools
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime
import inspect
import asyncio
from threading import Lock
import os

# Thread lock for thread-safe operations
_thread_lock = Lock()

def timer(func: Callable) -> Callable:
    """
    Decorator to measure execution time of a function
    """
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Function {func.__name__} executed in {run_time:.4f} seconds")
        
        return result
    
    @functools.wraps(func)
    async def async_wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Async function {func.__name__} executed in {run_time:.4f} seconds")
        
        return result
    
    return async_wrapper_timer if asyncio.iscoroutinefunction(func) else wrapper_timer

def retry(max_attempts: int = 3, delay: float = 1.0, 
          backoff: float = 2.0, exceptions: tuple = (Exception,),
          logger: Optional[logging.Logger] = None):
    """
    Decorator to retry a function on exception with exponential backoff
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier (e.g., 2.0 for exponential backoff)
        exceptions: Tuple of exceptions to catch and retry on
        logger: Logger instance for logging retries
    """
    def decorator_retry(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper_retry(*args, **kwargs):
            local_logger = logger or logging.getLogger(func.__module__)
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        local_logger.error(
                            f"Function {func.__name__} failed after {max_attempts} attempts. "
                            f"Last error: {str(e)}"
                        )
                        raise
                    
                    local_logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}. "
                        f"Retrying in {current_delay:.2f} seconds. Error: {str(e)}"
                    )
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
        
        @functools.wraps(func)
        async def async_wrapper_retry(*args, **kwargs):
            local_logger = logger or logging.getLogger(func.__module__)
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        local_logger.error(
                            f"Async function {func.__name__} failed after {max_attempts} attempts. "
                            f"Last error: {str(e)}"
                        )
                        raise
                    
                    local_logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}. "
                        f"Retrying in {current_delay:.2f} seconds. Error: {str(e)}"
                    )
                    
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
        
        return async_wrapper_retry if asyncio.iscoroutinefunction(func) else wrapper_retry
    
    return decorator_retry

def validate_input(validation_rules: Dict[str, Callable] = None, 
                  allow_extra: bool = True):
    """
    Decorator to validate function inputs based on rules
    
    Args:
        validation_rules: Dictionary mapping parameter names to validation functions
        allow_extra: Whether to allow extra parameters not in validation_rules
    """
    def decorator_validate(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper_validate(*args, **kwargs):
            if not validation_rules:
                return func(*args, **kwargs)
            
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate parameters
            for param_name, param_value in bound_args.arguments.items():
                if param_name in validation_rules:
                    validator = validation_rules[param_name]
                    if not validator(param_value):
                        raise ValueError(
                            f"Validation failed for parameter '{param_name}' with value: {param_value}"
                        )
                
                elif not allow_extra and param_name not in validation_rules:
                    raise ValueError(f"Unexpected parameter: {param_name}")
            
            return func(*args, **kwargs)
        
        return wrapper_validate
    
    return decorator_validate

def memoize(ttl: Optional[float] = None, maxsize: Optional[int] = 128):
    """
    Decorator to cache function results with optional TTL (Time To Live)
    
    Args:
        ttl: Time to live in seconds (None for no expiration)
        maxsize: Maximum cache size (None for unlimited)
    """
    def decorator_memoize(func: Callable) -> Callable:
        cache = {}
        cache_info = {
            'hits': 0,
            'misses': 0,
            'maxsize': maxsize,
            'currsize': 0
        }
        
        @functools.wraps(func)
        def wrapper_memoize(*args, **kwargs):
            # Create cache key from arguments
            key = functools._make_key(args, kwargs, typed=False)
            
            with _thread_lock:
                current_time = time.time()
                
                # Check if result is in cache and not expired
                if key in cache:
                    result, timestamp = cache[key]
                    if ttl is None or (current_time - timestamp) < ttl:
                        cache_info['hits'] += 1
                        return result
                    else:
                        # Remove expired entry
                        del cache[key]
                        cache_info['currsize'] -= 1
                
                cache_info['misses'] += 1
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                
                # Apply cache size limit
                if maxsize is not None and cache_info['currsize'] >= maxsize:
                    # Remove oldest entry (simple FIFO)
                    oldest_key = next(iter(cache))
                    del cache[oldest_key]
                    cache_info['currsize'] -= 1
                
                cache[key] = (result, current_time)
                cache_info['currsize'] += 1
                
                return result
        
        def clear_cache():
            """Clear the cache"""
            with _thread_lock:
                cache.clear()
                cache_info.update({
                    'hits': 0,
                    'misses': 0,
                    'currsize': 0
                })
        
        def get_cache_info():
            """Get cache statistics"""
            with _thread_lock:
                return cache_info.copy()
        
        # Add cache management methods to wrapper
        wrapper_memoize.clear_cache = clear_cache
        wrapper_memoize.get_cache_info = get_cache_info
        
        return wrapper_memoize
    
    return decorator_memoize

def rate_limit(max_calls: int, period: float):
    """
    Decorator to limit the rate of function calls
    
    Args:
        max_calls: Maximum number of calls allowed in the period
        period: Time period in seconds
    """
    def decorator_rate_limit(func: Callable) -> Callable:
        calls = []
        
        @functools.wraps(func)
        def wrapper_rate_limit(*args, **kwargs):
            nonlocal calls
            current_time = time.time()
            
            with _thread_lock:
                # Remove calls outside the current period
                calls = [call_time for call_time in calls if current_time - call_time < period]
                
                if len(calls) >= max_calls:
                    oldest_call = calls[0]
                    sleep_time = period - (current_time - oldest_call)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        # Update current time after sleep
                        current_time = time.time()
                        # Re-filter calls after sleep
                        calls = [call_time for call_time in calls if current_time - call_time < period]
                
                calls.append(current_time)
            
            return func(*args, **kwargs)
        
        return wrapper_rate_limit
    
    return decorator_rate_limit

def deprecated(reason: str = None, version: str = None):
    """
    Decorator to mark functions as deprecated
    
    Args:
        reason: Reason for deprecation
        version: Version when it was deprecated
    """
    def decorator_deprecated(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper_deprecated(*args, **kwargs):
            message = f"Function {func.__name__} is deprecated"
            if version:
                message += f" since version {version}"
            if reason:
                message += f". Reason: {reason}"
            
            logging.getLogger(func.__module__).warning(message)
            return func(*args, **kwargs)
        
        return wrapper_deprecated
    
    return decorator_deprecated

def singleton(cls):
    """
    Decorator to implement singleton pattern for classes
    """
    instances = {}
    
    @functools.wraps(cls)
    def wrapper_singleton(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return wrapper_singleton

def log_execution(level: int = logging.INFO, log_args: bool = True, 
                 log_result: bool = False, log_time: bool = True):
    """
    Decorator to log function execution details
    
    Args:
        level: Logging level
        log_args: Whether to log function arguments
        log_result: Whether to log function result
        log_time: Whether to log execution time
    """
    def decorator_log(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper_log(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            
            # Log function call with arguments
            if log_args:
                arg_str = ", ".join([str(arg) for arg in args])
                kwarg_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
                all_args = ", ".join(filter(None, [arg_str, kwarg_str]))
                logger.log(level, f"Calling {func.__name__}({all_args})")
            else:
                logger.log(level, f"Calling {func.__name__}")
            
            start_time = time.time() if log_time else None
            
            try:
                result = func(*args, **kwargs)
                
                # Log result
                if log_result and result is not None:
                    result_str = str(result)
                    if len(result_str) > 100:  # Truncate long results
                        result_str = result_str[:100] + "..."
                    logger.log(level, f"Function {func.__name__} returned: {result_str}")
                
                # Log execution time
                if log_time and start_time:
                    execution_time = time.time() - start_time
                    logger.log(level, f"Function {func.__name__} executed in {execution_time:.4f}s")
                
                return result
                
            except Exception as e:
                logger.error(f"Function {func.__name__} failed with error: {str(e)}")
                raise
        
        @functools.wraps(func)
        async def async_wrapper_log(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            
            if log_args:
                arg_str = ", ".join([str(arg) for arg in args])
                kwarg_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
                all_args = ", ".join(filter(None, [arg_str, kwarg_str]))
                logger.log(level, f"Calling async {func.__name__}({all_args})")
            else:
                logger.log(level, f"Calling async {func.__name__}")
            
            start_time = time.time() if log_time else None
            
            try:
                result = await func(*args, **kwargs)
                
                if log_result and result is not None:
                    result_str = str(result)
                    if len(result_str) > 100:
                        result_str = result_str[:100] + "..."
                    logger.log(level, f"Async function {func.__name__} returned: {result_str}")
                
                if log_time and start_time:
                    execution_time = time.time() - start_time
                    logger.log(level, f"Async function {func.__name__} executed in {execution_time:.4f}s")
                
                return result
                
            except Exception as e:
                logger.error(f"Async function {func.__name__} failed with error: {str(e)}")
                raise
        
        return async_wrapper_log if asyncio.iscoroutinefunction(func) else wrapper_log
    
    return decorator_log

def thread_safe(lock: Optional[Lock] = None):
    """
    Decorator to make function thread-safe using a lock
    
    Args:
        lock: Custom lock to use (creates new lock if None)
    """
    def decorator_thread_safe(func: Callable) -> Callable:
        local_lock = lock or Lock()
        
        @functools.wraps(func)
        def wrapper_thread_safe(*args, **kwargs):
            with local_lock:
                return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper_thread_safe(*args, **kwargs):
            with local_lock:
                return await func(*args, **kwargs)
        
        return async_wrapper_thread_safe if asyncio.iscoroutinefunction(func) else wrapper_thread_safe
    
    return decorator_thread_safe

# Common validation functions for use with validate_input decorator
def is_positive(value) -> bool:
    """Check if value is positive"""
    return value > 0 if isinstance(value, (int, float)) else False

def is_non_negative(value) -> bool:
    """Check if value is non-negative"""
    return value >= 0 if isinstance(value, (int, float)) else False

def is_in_range(min_val, max_val):
    """Factory function to create range validator"""
    def validator(value):
        return min_val <= value <= max_val if isinstance(value, (int, float)) else False
    return validator

def is_list_of_type(expected_type):
    """Factory function to create list type validator"""
    def validator(value):
        return isinstance(value, list) and all(isinstance(item, expected_type) for item in value)
    return validator

def is_dict_with_keys(required_keys: List[str], optional_keys: List[str] = None):
    """Factory function to create dictionary structure validator"""
    def validator(value):
        if not isinstance(value, dict):
            return False
        
        # Check required keys
        if not all(key in value for key in required_keys):
            return False
        
        # Check no extra keys if optional_keys specified
        if optional_keys is not None:
            all_allowed_keys = set(required_keys) | set(optional_keys)
            if not all(key in all_allowed_keys for key in value.keys()):
                return False
        
        return True
    return validator
