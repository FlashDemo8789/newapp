"""
ML Pipeline Orchestration

This module provides a framework for creating and executing ML pipelines
with reproducible results. It supports DAG-based workflow execution, 
dependency tracking, and caching of intermediate results.

Features:
- Define pipelines as DAGs of tasks
- Cache intermediate task results
- Track task dependencies
- Log pipeline execution metrics
- Retry failed tasks automatically
"""

import os
import time
import json
import pickle
import logging
import hashlib
import traceback
from typing import Dict, List, Any, Callable, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TaskResult:
    """Results from a pipeline task execution"""
    task_id: str
    status: str  # success, failure, skipped
    start_time: datetime
    end_time: datetime
    result: Any = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        # Convert datetime to string for JSON serialization
        result['start_time'] = self.start_time.isoformat()
        result['end_time'] = self.end_time.isoformat()
        
        # Handle result data (skip if not serializable)
        try:
            json.dumps({"test": self.result})  # Try serializing with a test key
            result['result'] = self.result
        except (TypeError, OverflowError):
            result['result'] = "NON_SERIALIZABLE_RESULT"
        
        return result

class Task:
    """
    A unit of work in a pipeline
    
    Tasks are the building blocks of pipelines, representing individual steps
    that process data and produce results.
    """
    
    def __init__(
        self,
        task_id: str,
        function: Callable,
        dependencies: List[str] = None,
        cache_result: bool = True,
        retry_count: int = 0,
        timeout: Optional[int] = None,
        description: str = ""
    ):
        """
        Initialize a pipeline task
        
        Args:
            task_id: Unique identifier for this task
            function: The function to execute for this task
            dependencies: List of task_ids this task depends on
            cache_result: Whether to cache the result of this task
            retry_count: Number of times to retry on failure
            timeout: Maximum execution time in seconds (None = no limit)
            description: Human-readable description of the task
        """
        self.task_id = task_id
        self.function = function
        self.dependencies = dependencies or []
        self.cache_result = cache_result
        self.retry_count = retry_count
        self.timeout = timeout
        self.description = description
    
    def execute(self, context: Dict[str, Any]) -> Any:
        """
        Execute the task function with context
        
        Args:
            context: Dictionary containing pipeline context,
                    including results from dependent tasks
                    
        Returns:
            Result of the task function
        """
        return self.function(context)
    
    def get_cache_key(self, context: Dict[str, Any]) -> str:
        """
        Generate a cache key for this task
        
        The cache key is based on the task ID and the
        relevant parts of the context that affect execution.
        
        Args:
            context: Pipeline execution context
            
        Returns:
            String cache key
        """
        # Start with task ID
        key_parts = [self.task_id]
        
        # Get dependencies from context
        for dep in self.dependencies:
            # Add hash of dependency result if it exists in context
            if dep in context and 'result' in context[dep]:
                # Try to create a deterministic hash
                dep_result = context[dep]['result']
                try:
                    # For basic types, use string representation
                    if isinstance(dep_result, (int, float, str, bool)):
                        hash_input = str(dep_result)
                    elif isinstance(dep_result, (list, tuple, dict)):
                        hash_input = json.dumps(dep_result, sort_keys=True)
                    else:
                        # For complex objects, use pickle hash
                        hash_input = pickle.dumps(dep_result)
                    
                    # Create hash
                    result_hash = hashlib.md5(hash_input.encode() if isinstance(hash_input, str) else hash_input).hexdigest()
                    key_parts.append(f"{dep}:{result_hash}")
                except:
                    # If hashing fails, use timestamp to prevent incorrect cache hits
                    key_parts.append(f"{dep}:{time.time()}")
        
        # Join all parts with a separator
        return ":".join(key_parts)

class Pipeline:
    """
    ML Pipeline for orchestrating task execution
    
    Pipelines coordinate the execution of interdependent tasks,
    handling data flow between tasks and caching results.
    """
    
    def __init__(
        self,
        name: str,
        storage_dir: str = "./pipeline_data",
        max_workers: int = 4
    ):
        """
        Initialize a pipeline
        
        Args:
            name: Name of the pipeline
            storage_dir: Directory for caching results
            max_workers: Maximum number of concurrent task executions
        """
        self.name = name
        self.storage_dir = storage_dir
        self.max_workers = max_workers
        self.tasks: Dict[str, Task] = {}
        self.graph = nx.DiGraph()
        self.task_results: Dict[str, TaskResult] = {}
        self.cache: Dict[str, Any] = {}
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        os.makedirs(os.path.join(storage_dir, 'cache'), exist_ok=True)
    
    def add_task(self, task: Task) -> 'Pipeline':
        """
        Add a task to the pipeline
        
        Args:
            task: Task to add
            
        Returns:
            Self for chaining
        """
        if task.task_id in self.tasks:
            raise ValueError(f"Task with ID {task.task_id} already exists")
        
        self.tasks[task.task_id] = task
        
        # Add to graph
        self.graph.add_node(task.task_id, task=task)
        
        # Add dependency edges
        for dep_id in task.dependencies:
            if dep_id not in self.tasks:
                raise ValueError(f"Dependency task {dep_id} not found")
            self.graph.add_edge(dep_id, task.task_id)
        
        # Verify no cycles in graph
        if not nx.is_directed_acyclic_graph(self.graph):
            self.graph.remove_node(task.task_id)
            raise ValueError(f"Adding task {task.task_id} would create a cycle in the pipeline")
        
        return self
    
    def get_task_execution_order(self) -> List[str]:
        """
        Get the topological order of tasks for execution
        
        Returns:
            List of task IDs in execution order
        """
        return list(nx.topological_sort(self.graph))
    
    def get_ready_tasks(self, completed_tasks: Set[str]) -> List[str]:
        """
        Get tasks ready for execution
        
        A task is ready when all its dependencies have been completed.
        
        Args:
            completed_tasks: Set of task IDs that have been completed
            
        Returns:
            List of task IDs ready for execution
        """
        ready_tasks = []
        
        for task_id, task in self.tasks.items():
            if task_id not in completed_tasks:
                deps_completed = all(dep in completed_tasks for dep in task.dependencies)
                if deps_completed:
                    ready_tasks.append(task_id)
        
        return ready_tasks
    
    def _get_cache_path(self, cache_key: str) -> str:
        """
        Get cache file path for a given cache key
        
        Args:
            cache_key: The cache key
            
        Returns:
            Path to cache file
        """
        # Create a filename-safe version of the cache key
        safe_key = hashlib.md5(cache_key.encode()).hexdigest()
        return os.path.join(self.storage_dir, 'cache', f"{safe_key}.pkl")
    
    def _load_from_cache(self, cache_key: str) -> Optional[Any]:
        """
        Load result from cache
        
        Args:
            cache_key: Cache key to load
            
        Returns:
            Cached result or None if not in cache
        """
        cache_path = self._get_cache_path(cache_key)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load from cache {cache_key}: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, result: Any) -> bool:
        """
        Save result to cache
        
        Args:
            cache_key: Cache key
            result: Result to cache
            
        Returns:
            True if saved successfully, False otherwise
        """
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            return True
        except Exception as e:
            logger.warning(f"Failed to save to cache {cache_key}: {e}")
            return False
    
    def execute_task(self, task_id: str, context: Dict[str, Any]) -> TaskResult:
        """
        Execute a single task
        
        Args:
            task_id: ID of the task to execute
            context: Pipeline execution context
            
        Returns:
            TaskResult object
        """
        task = self.tasks[task_id]
        logger.info(f"Executing task {task_id}")
        
        start_time = datetime.now()
        result = None
        error = None
        status = "success"
        
        # Check cache first if caching is enabled
        if task.cache_result:
            cache_key = task.get_cache_key(context)
            result = self._load_from_cache(cache_key)
            
            if result is not None:
                logger.info(f"Task {task_id} loaded from cache")
                return TaskResult(
                    task_id=task_id,
                    status="success",
                    start_time=start_time,
                    end_time=datetime.now(),
                    result=result
                )
        
        # Execute the task
        attempts = 0
        while attempts <= task.retry_count:
            try:
                result = task.execute(context)
                break
            except Exception as e:
                attempts += 1
                error = str(e)
                stack_trace = traceback.format_exc()
                logger.error(f"Task {task_id} failed (attempt {attempts}/{task.retry_count + 1}): {e}\n{stack_trace}")
                
                if attempts <= task.retry_count:
                    # Wait before retrying (with increasing backoff)
                    time.sleep(2 ** attempts)
                else:
                    status = "failure"
        
        end_time = datetime.now()
        
        # Cache result if successful and caching is enabled
        if status == "success" and task.cache_result:
            cache_key = task.get_cache_key(context)
            self._save_to_cache(cache_key, result)
        
        return TaskResult(
            task_id=task_id,
            status=status,
            start_time=start_time,
            end_time=end_time,
            result=result,
            error=error
        )
    
    def execute(self, initial_context: Dict[str, Any] = None) -> Dict[str, TaskResult]:
        """
        Execute the pipeline
        
        Args:
            initial_context: Initial context data
            
        Returns:
            Dictionary mapping task IDs to their execution results
        """
        if not self.tasks:
            logger.warning("No tasks in pipeline")
            return {}
        
        # Initialize context and execution state
        context = initial_context or {}
        completed_tasks: Set[str] = set()
        results: Dict[str, TaskResult] = {}
        
        # Get execution order
        execution_order = self.get_task_execution_order()
        logger.info(f"Pipeline execution order: {', '.join(execution_order)}")
        
        # Execute tasks in parallel where possible
        while len(completed_tasks) < len(self.tasks):
            # Get tasks ready for execution
            ready_tasks = [task_id for task_id in self.get_ready_tasks(completed_tasks) if task_id not in completed_tasks]
            
            if not ready_tasks:
                if len(completed_tasks) < len(self.tasks):
                    logger.error("Pipeline deadlocked - no tasks ready but not all completed")
                    break
            
            # Execute ready tasks in parallel
            future_to_task = {}
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for task_id in ready_tasks:
                    future = executor.submit(self.execute_task, task_id, context)
                    future_to_task[future] = task_id
                
                for future in as_completed(future_to_task):
                    task_id = future_to_task[future]
                    try:
                        task_result = future.result()
                        results[task_id] = task_result
                        
                        # Update context with task result
                        context[task_id] = task_result.to_dict()
                        
                        if task_result.status == "success":
                            completed_tasks.add(task_id)
                            logger.info(f"Task {task_id} completed successfully")
                        else:
                            logger.error(f"Task {task_id} failed: {task_result.error}")
                    except Exception as e:
                        logger.error(f"Error processing task {task_id} result: {e}")
                        results[task_id] = TaskResult(
                            task_id=task_id,
                            status="failure",
                            start_time=datetime.now(),
                            end_time=datetime.now(),
                            error=str(e)
                        )
        
        # Store results
        self.task_results = results
        
        # Check if all tasks completed
        if len(completed_tasks) == len(self.tasks):
            logger.info("Pipeline execution completed successfully")
        else:
            failed_tasks = [
                task_id for task_id, result in results.items()
                if result.status != "success"
            ]
            logger.error(f"Pipeline execution failed. Failed tasks: {', '.join(failed_tasks)}")
        
        return results
    
    def get_final_result(self, task_id: str = None) -> Any:
        """
        Get the final result from a specific task or the last task
        
        Args:
            task_id: ID of the task to get result from (if None, uses the last task)
            
        Returns:
            Task result or None if not found
        """
        if not self.task_results:
            return None
        
        if task_id is None:
            # Get the last task in execution order
            execution_order = self.get_task_execution_order()
            task_id = execution_order[-1]
        
        if task_id in self.task_results:
            task_result = self.task_results[task_id]
            if task_result.status == "success":
                return task_result.result
        
        return None
    
    def clear_cache(self) -> None:
        """Clear pipeline cache"""
        cache_dir = os.path.join(self.storage_dir, 'cache')
        
        if os.path.exists(cache_dir):
            for filename in os.listdir(cache_dir):
                if filename.endswith('.pkl'):
                    try:
                        os.remove(os.path.join(cache_dir, filename))
                    except Exception as e:
                        logger.warning(f"Failed to remove cache file {filename}: {e}")
        
        logger.info("Pipeline cache cleared")

class PipelineBuilder:
    """
    Builder for creating pipelines with a fluent interface
    
    This class provides a convenient way to construct pipelines
    with method chaining.
    """
    
    def __init__(self, name: str, storage_dir: str = "./pipeline_data"):
        """
        Initialize a pipeline builder
        
        Args:
            name: Name of the pipeline
            storage_dir: Directory for caching results
        """
        self.pipeline = Pipeline(name, storage_dir)
    
    def add_task(
        self,
        task_id: str,
        function: Callable,
        dependencies: List[str] = None,
        cache_result: bool = True,
        retry_count: int = 0,
        timeout: Optional[int] = None,
        description: str = ""
    ) -> 'PipelineBuilder':
        """
        Add a task to the pipeline
        
        Args:
            task_id: Unique identifier for this task
            function: The function to execute for this task
            dependencies: List of task_ids this task depends on
            cache_result: Whether to cache the result of this task
            retry_count: Number of times to retry on failure
            timeout: Maximum execution time in seconds (None = no limit)
            description: Human-readable description of the task
            
        Returns:
            Self for chaining
        """
        task = Task(
            task_id=task_id,
            function=function,
            dependencies=dependencies,
            cache_result=cache_result,
            retry_count=retry_count,
            timeout=timeout,
            description=description
        )
        
        self.pipeline.add_task(task)
        return self
    
    def build(self) -> Pipeline:
        """
        Build and return the pipeline
        
        Returns:
            The constructed pipeline
        """
        return self.pipeline 