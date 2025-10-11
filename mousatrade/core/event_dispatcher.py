"""
Event Dispatcher System for Trading Engine
Implements event-driven architecture for loose coupling
"""

import logging
import asyncio
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum
import inspect
from concurrent.futures import ThreadPoolExecutor


class EventPriority(Enum):
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class Event:
    name: str
    data: Any
    timestamp: float
    priority: EventPriority = EventPriority.NORMAL
    source: str = None


class EventDispatcher:
    """
    Manages event publishing and subscription
    """
    
    def __init__(self, max_workers: int = 20):
        self.handlers: Dict[str, List[Callable]] = {}
        self.async_handlers: Dict[str, List[Callable]] = {}
        self.event_history: List[Event] = []
        self.max_history_size = 1000
        
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.loop = asyncio.new_event_loop()
        
        self.is_running = True
        self.logger = self._setup_logging()
    
    def _setup_logging(self):
        """Setup event dispatcher logging"""
        logger = logging.getLogger("event_dispatcher")
        return logger
    
    def register_handler(self, event_name: str, handler: Callable, is_async: bool = False):
        """Register an event handler"""
        handler_list = self.async_handlers if is_async else self.handlers
        
        if event_name not in handler_list:
            handler_list[event_name] = []
        
        handler_list[event_name].append(handler)
        self.logger.debug(f"Handler registered for event: {event_name}")
    
    def unregister_handler(self, event_name: str, handler: Callable):
        """Unregister an event handler"""
        for handler_list in [self.handlers, self.async_handlers]:
            if event_name in handler_list and handler in handler_list[event_name]:
                handler_list[event_name].remove(handler)
                self.logger.debug(f"Handler unregistered for event: {event_name}")
    
    def emit(self, event_name: str, data: Any, priority: EventPriority = EventPriority.NORMAL, source: str = None):
        """Emit an event (synchronous)"""
        if not self.is_running:
            return
        
        event = Event(
            name=event_name,
            data=data,
            timestamp=asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0,
            priority=priority,
            source=source
        )
        
        # Add to history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history_size:
            self.event_history.pop(0)
        
        # Process synchronous handlers
        if event_name in self.handlers:
            for handler in self.handlers[event_name]:
                try:
                    handler(data)
                except Exception as e:
                    self.logger.error(f"Handler failed for event {event_name}: {e}")
        
        # Process asynchronous handlers in background
        if event_name in self.async_handlers:
            for handler in self.async_handlers[event_name]:
                self.executor.submit(self._execute_async_handler, handler, data, event_name)
    
    async def emit_async(self, event_name: str, data: Any, priority: EventPriority = EventPriority.NORMAL, source: str = None):
        """Emit an event asynchronously"""
        if not self.is_running:
            return
        
        event = Event(
            name=event_name,
            data=data,
            timestamp=asyncio.get_event_loop().time(),
            priority=priority,
            source=source
        )
        
        # Add to history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history_size:
            self.event_history.pop(0)
        
        # Process synchronous handlers
        if event_name in self.handlers:
            for handler in self.handlers[event_name]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    self.logger.error(f"Handler failed for event {event_name}: {e}")
        
        # Process asynchronous handlers
        if event_name in self.async_handlers:
            tasks = []
            for handler in self.async_handlers[event_name]:
                task = asyncio.create_task(self._execute_async_handler_await(handler, data, event_name))
                tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    def _execute_async_handler(self, handler: Callable, data: Any, event_name: str):
        """Execute async handler in thread pool"""
        try:
            if asyncio.iscoroutinefunction(handler):
                # Run coroutine in event loop
                asyncio.run_coroutine_threadsafe(handler(data), self.loop)
            else:
                handler(data)
        except Exception as e:
            self.logger.error(f"Async handler failed for event {event_name}: {e}")
    
    async def _execute_async_handler_await(self, handler: Callable, data: Any, event_name: str):
        """Execute async handler with await"""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(data)
            else:
                handler(data)
        except Exception as e:
            self.logger.error(f"Async handler failed for event {event_name}: {e}")
    
    def register_middleware(self, middleware: Callable):
        """Register event middleware for preprocessing"""
        # Middleware can modify events before they reach handlers
        # Implementation depends on specific middleware requirements
        pass
    
    def get_event_statistics(self) -> Dict[str, Any]:
        """Get event system statistics"""
        sync_handlers = sum(len(handlers) for handlers in self.handlers.values())
        async_handlers = sum(len(handlers) for handlers in self.async_handlers.values())
        
        event_counts = {}
        for event in self.event_history[-1000:]:  # Last 1000 events
            event_counts[event.name] = event_counts.get(event.name, 0) + 1
        
        return {
            'total_events_processed': len(self.event_history),
            'sync_handlers_count': sync_handlers,
            'async_handlers_count': async_handlers,
            'unique_event_types': len(set(self.handlers.keys()) | set(self.async_handlers.keys())),
            'recent_event_counts': event_counts,
            'is_running': self.is_running
        }
    
    def get_recent_events(self, limit: int = 50) -> List[Event]:
        """Get recent events"""
        return self.event_history[-limit:]
    
    def find_events(self, event_name: str, limit: int = 20) -> List[Event]:
        """Find events by name"""
        return [event for event in self.event_history if event.name == event_name][-limit:]
    
    def clear_history(self):
        """Clear event history"""
        self.event_history.clear()
        self.logger.info("Event history cleared")
    
    def start(self):
        """Start event dispatcher"""
        self.is_running = True
        self.logger.info("Event dispatcher started")
    
    def stop(self):
        """Stop event dispatcher"""
        self.is_running = False
        self.executor.shutdown(wait=True)
        self.logger.info("Event dispatcher stopped")
    
    def wait_for_event(self, event_name: str, timeout: float = 30.0) -> Optional[Event]:
        """Wait for specific event (blocking)"""
        import time
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            for event in reversed(self.event_history):
                if event.name == event_name and event.timestamp >= start_time:
                    return event
            time.sleep(0.1)
        
        return None
