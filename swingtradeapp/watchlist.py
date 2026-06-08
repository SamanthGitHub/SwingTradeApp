"""Watchlist management with persistence to JSON."""

import json
import os
from typing import List, Dict, Optional
from pathlib import Path


class WatchlistManager:
    """Manage multiple watchlists with JSON persistence."""
    
    def __init__(self, data_dir: str = '.data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.watchlists_file = self.data_dir / 'watchlists.json'
        self.alerts_file = self.data_dir / 'alerts.json'
        self._load_watchlists()
        self._load_alerts()
    
    def _load_watchlists(self) -> None:
        """Load watchlists from JSON file."""
        if self.watchlists_file.exists():
            try:
                with open(self.watchlists_file, 'r') as f:
                    self.watchlists = json.load(f)
            except Exception:
                self.watchlists = {'Default': []}
        else:
            self.watchlists = {'Default': []}
    
    def _load_alerts(self) -> None:
        """Load price alerts from JSON file."""
        if self.alerts_file.exists():
            try:
                with open(self.alerts_file, 'r') as f:
                    self.alerts = json.load(f)
            except Exception:
                self.alerts = {}
        else:
            self.alerts = {}
    
    def _save_watchlists(self) -> None:
        """Save watchlists to JSON file."""
        with open(self.watchlists_file, 'w') as f:
            json.dump(self.watchlists, f, indent=2)
    
    def _save_alerts(self) -> None:
        """Save alerts to JSON file."""
        with open(self.alerts_file, 'w') as f:
            json.dump(self.alerts, f, indent=2)
    
    def create_watchlist(self, name: str) -> None:
        """Create a new watchlist."""
        if name not in self.watchlists:
            self.watchlists[name] = []
            self._save_watchlists()
    
    def delete_watchlist(self, name: str) -> None:
        """Delete a watchlist."""
        if name in self.watchlists and name != 'Default':
            del self.watchlists[name]
            self._save_watchlists()
    
    def list_watchlists(self) -> List[str]:
        """Get all watchlist names."""
        return sorted(list(self.watchlists.keys()))
    
    def add_symbol(self, watchlist_name: str, symbol: str) -> None:
        """Add symbol to watchlist."""
        if watchlist_name not in self.watchlists:
            self.create_watchlist(watchlist_name)
        if symbol not in self.watchlists[watchlist_name]:
            self.watchlists[watchlist_name].append(symbol)
            self._save_watchlists()
    
    def remove_symbol(self, watchlist_name: str, symbol: str) -> None:
        """Remove symbol from watchlist."""
        if watchlist_name in self.watchlists and symbol in self.watchlists[watchlist_name]:
            self.watchlists[watchlist_name].remove(symbol)
            self._save_watchlists()
    
    def get_watchlist(self, name: str) -> List[str]:
        """Get symbols in a watchlist."""
        return self.watchlists.get(name, [])
    
    def add_alert(self, symbol: str, alert_type: str, value: float, comparison: str = 'above') -> None:
        """Add price or metric alert.
        
        Args:
            symbol: Stock symbol
            alert_type: 'price', 'rsi', 'sma_ratio', etc.
            value: Trigger value
            comparison: 'above' or 'below'
        """
        if symbol not in self.alerts:
            self.alerts[symbol] = []
        
        alert = {
            'type': alert_type,
            'value': float(value),
            'comparison': comparison,
            'active': True
        }
        self.alerts[symbol].append(alert)
        self._save_alerts()
    
    def remove_alert(self, symbol: str, index: int) -> None:
        """Remove an alert by index."""
        if symbol in self.alerts and 0 <= index < len(self.alerts[symbol]):
            del self.alerts[symbol][index]
            self._save_alerts()
    
    def get_alerts(self, symbol: str) -> List[Dict]:
        """Get alerts for a symbol."""
        return self.alerts.get(symbol, [])
    
    def get_all_alerts(self) -> Dict[str, List[Dict]]:
        """Get all alerts."""
        return self.alerts
