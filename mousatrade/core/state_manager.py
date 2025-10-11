"""
State Management for Trading Engine
Manages engine state, persistence, and recovery
"""

import logging
import json
import pickle
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from enum import Enum
import sqlite3
from dataclasses import dataclass, asdict


class EngineState(Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class EngineSnapshot:
    timestamp: datetime
    state: EngineState
    portfolio: Dict[str, Any]
    active_orders: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    strategy_states: Dict[str, Any]


class StateManager:
    """
    Manages trading engine state and provides persistence
    """
    
    def __init__(self, data_dir: str = "data/state"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_state = EngineState.INITIALIZING
        self.state_history = []
        self.recovery_mode = False
        
        self.logger = self._setup_logging()
        self._init_database()
    
    def _setup_logging(self):
        """Setup state manager logging"""
        logger = logging.getLogger("state_manager")
        return logger
    
    def _init_database(self):
        """Initialize state database"""
        db_path = self.data_dir / "state.db"
        self.conn = sqlite3.connect(db_path)
        
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS engine_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                state TEXT NOT NULL,
                snapshot_data BLOB NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS state_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                from_state TEXT NOT NULL,
                to_state TEXT NOT NULL,
                reason TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
    
    def set_state(self, new_state: EngineState, reason: str = None):
        """Change engine state with transition tracking"""
        old_state = self.current_state
        self.current_state = new_state
        
        # Record state transition
        transition = {
            'timestamp': datetime.now(),
            'from_state': old_state.value,
            'to_state': new_state.value,
            'reason': reason
        }
        
        self.state_history.append(transition)
        self._save_state_transition(transition)
        
        self.logger.info(f"State transition: {old_state.value} -> {new_state.value} ({reason})")
    
    def get_state(self) -> EngineState:
        """Get current engine state"""
        return self.current_state
    
    def create_snapshot(self, 
                       portfolio: Dict[str, Any],
                       active_orders: Dict[str, Any],
                       performance_metrics: Dict[str, Any],
                       strategy_states: Dict[str, Any]) -> str:
        """Create engine snapshot for persistence"""
        snapshot = EngineSnapshot(
            timestamp=datetime.now(),
            state=self.current_state,
            portfolio=portfolio,
            active_orders=active_orders,
            performance_metrics=performance_metrics,
            strategy_states=strategy_states
        )
        
        # Serialize snapshot
        snapshot_data = pickle.dumps(asdict(snapshot))
        
        # Save to database
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO engine_snapshots (timestamp, state, snapshot_data)
            VALUES (?, ?, ?)
        ''', (snapshot.timestamp, snapshot.state.value, snapshot_data))
        
        self.conn.commit()
        snapshot_id = cursor.lastrowid
        
        self.logger.info(f"Snapshot created: {snapshot_id}")
        return str(snapshot_id)
    
    def load_snapshot(self, snapshot_id: str) -> Optional[EngineSnapshot]:
        """Load engine snapshot by ID"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT snapshot_data FROM engine_snapshots WHERE id = ?
            ''', (snapshot_id,))
            
            result = cursor.fetchone()
            if result:
                snapshot_dict = pickle.loads(result[0])
                snapshot = EngineSnapshot(**snapshot_dict)
                self.logger.info(f"Snapshot loaded: {snapshot_id}")
                return snapshot
            else:
                self.logger.warning(f"Snapshot not found: {snapshot_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to load snapshot {snapshot_id}: {e}")
            return None
    
    def get_latest_snapshot(self) -> Optional[EngineSnapshot]:
        """Get the most recent snapshot"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT snapshot_data FROM engine_snapshots 
                ORDER BY timestamp DESC LIMIT 1
            ''')
            
            result = cursor.fetchone()
            if result:
                snapshot_dict = pickle.loads(result[0])
                snapshot = EngineSnapshot(**snapshot_dict)
                return snapshot
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get latest snapshot: {e}")
            return None
    
    def recover_from_failure(self) -> Optional[EngineSnapshot]:
        """Recover engine state from failure"""
        self.recovery_mode = True
        self.logger.info("Initiating failure recovery...")
        
        # Try to load latest snapshot
        snapshot = self.get_latest_snapshot()
        if snapshot:
            self.current_state = snapshot.state
            self.logger.info(f"Recovered to state: {snapshot.state.value}")
            return snapshot
        else:
            self.logger.warning("No snapshot found for recovery")
            return None
    
    def save_strategy_state(self, strategy_id: str, state_data: Dict[str, Any]):
        """Save strategy-specific state"""
        state_file = self.data_dir / f"strategy_{strategy_id}.json"
        try:
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
            self.logger.debug(f"Strategy state saved: {strategy_id}")
        except Exception as e:
            self.logger.error(f"Failed to save strategy state {strategy_id}: {e}")
    
    def load_strategy_state(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Load strategy-specific state"""
        state_file = self.data_dir / f"strategy_{strategy_id}.json"
        try:
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                self.logger.debug(f"Strategy state loaded: {strategy_id}")
                return state_data
            else:
                return None
        except Exception as e:
            self.logger.error(f"Failed to load strategy state {strategy_id}: {e}")
            return None
    
    def _save_state_transition(self, transition: Dict[str, Any]):
        """Save state transition to database"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO state_history (timestamp, from_state, to_state, reason)
            VALUES (?, ?, ?, ?)
        ''', (
            transition['timestamp'],
            transition['from_state'],
            transition['to_state'],
            transition['reason']
        ))
        self.conn.commit()
    
    def get_state_history(self, limit: int = 100) -> list:
        """Get state transition history"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT timestamp, from_state, to_state, reason 
            FROM state_history 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        return cursor.fetchall()
    
    def cleanup_old_snapshots(self, keep_last: int = 10):
        """Clean up old snapshots, keep only recent ones"""
        try:
            cursor = self.conn.cursor()
            
            # Get IDs of snapshots to delete
            cursor.execute('''
                SELECT id FROM engine_snapshots 
                ORDER BY timestamp DESC 
                LIMIT -1 OFFSET ?
            ''', (keep_last,))
            
            old_snapshot_ids = [row[0] for row in cursor.fetchall()]
            
            if old_snapshot_ids:
                # Delete old snapshots
                placeholders = ','.join('?' * len(old_snapshot_ids))
                cursor.execute(f'''
                    DELETE FROM engine_snapshots 
                    WHERE id IN ({placeholders})
                ''', old_snapshot_ids)
                
                self.conn.commit()
                self.logger.info(f"Cleaned up {len(old_snapshot_ids)} old snapshots")
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup old snapshots: {e}")
    
    def get_state_statistics(self) -> Dict[str, Any]:
        """Get state management statistics"""
        cursor = self.conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM engine_snapshots')
        snapshot_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM state_history')
        transition_count = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT COUNT(DISTINCT date(timestamp)) 
            FROM state_history
        ''')
        active_days = cursor.fetchone()[0]
        
        return {
            'current_state': self.current_state.value,
            'snapshot_count': snapshot_count,
            'transition_count': transition_count,
            'active_days': active_days,
            'recovery_mode': self.recovery_mode,
            'state_history_size': len(self.state_history)
        }
    
    def __del__(self):
        """Cleanup on destruction"""
        if hasattr(self, 'conn'):
            self.conn.close()
