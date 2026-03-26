"""
Gesture Manager - Organized gesture loading with caching
Supports category-based organization for scalability
"""

import json
from pathlib import Path
from typing import Dict, Optional, List


class GestureManager:
    """
    Manages gesture loading with category organization and caching.
    
    Organization:
    - actions/: Motion-based gestures (wave_hello, etc.)
    - characters/: Static hand letters (letter_a, letter_b, etc.)
    - numbers/: Static hand numbers (number_0-9)
    
    Supports both old flat structure and new category structure.
    """
    
    CATEGORIES = {
        "actions": "motion-based gestures (wave, clap, etc.)",
        "characters": "static hand letters (A-Y)",
        "numbers": "static hand numbers (0-9)"
    }
    
    def __init__(self, gesture_dir: str = "assets/gestures"):
        self.gesture_dir = Path(gesture_dir)
        self.cache = {}      # gesture_key -> gesture_data
        self.metadata = {}   # gesture_key -> {"category", "name", "type"}
        self._load_metadata()
    
    def _load_metadata(self):
        """
        Scan gesture directory and build metadata index.
        This is fast (JSON headers only, not full content).
        """
        if not self.gesture_dir.exists():
            print(f"[WARNING] Gesture directory not found: {self.gesture_dir}")
            return
        
        # Try new structure first (category-based)
        has_categories = any((self.gesture_dir / cat).exists() for cat in self.CATEGORIES)
        
        if has_categories:
            print(f"[OK] Using category-based structure")
            self._load_from_categories()
        else:
            print(f"[OK] Using flat structure (legacy)")
            self._load_from_flat()
    
    def _load_from_categories(self):
        """Load from category structure"""
        for category, description in self.CATEGORIES.items():
            cat_dir = self.gesture_dir / category
            if not cat_dir.exists():
                continue
            
            for json_file in sorted(cat_dir.glob("*.json")):
                gesture_key = json_file.stem
                self.metadata[gesture_key] = {
                    "category": category,
                    "file": json_file,
                    "loaded": False,
                    "name": None
                }
            
            print(f"[+] Found {len(list(cat_dir.glob('*.json')))} {category}")
    
    def _load_from_flat(self):
        """Load from flat structure (backward compatible)"""
        for json_file in sorted(self.gesture_dir.glob("*.json")):
            gesture_key = json_file.stem
            
            # Classify by naming convention
            if gesture_key.startswith("letter_"):
                category = "characters"
            elif gesture_key.startswith("number_"):
                category = "numbers"
            else:
                category = "actions"
            
            self.metadata[gesture_key] = {
                "category": category,
                "file": json_file,
                "loaded": False,
                "name": None
            }
        
        print(f"[+] Found {len(self.metadata)} gestures (flat structure)")
    
    def get_all(self, category: Optional[str] = None) -> Dict:
        """
        Get all gestures (optionally filtered by category).
        Lazy-loads on first access.
        
        Args:
            category: Optional category filter ("actions", "characters", "numbers")
        
        Returns: {gesture_key: gesture_data, ...}
        """
        result = {}
        
        for gesture_key, meta in self.metadata.items():
            # Filter by category if specified
            if category and meta["category"] != category:
                continue
            
            # Lazy load
            if not meta["loaded"]:
                self._load_gesture(gesture_key)
            
            if gesture_key in self.cache:
                result[gesture_key] = self.cache[gesture_key]
        
        return result
    
    def get(self, gesture_key: str) -> Optional[Dict]:
        """
        Get a single gesture by key.
        Lazy-loads on first access.
        """
        if gesture_key not in self.metadata:
            return None
        
        # Lazy load
        if not self.metadata[gesture_key]["loaded"]:
            self._load_gesture(gesture_key)
        
        return self.cache.get(gesture_key)
    
    def get_names(self, category: Optional[str] = None) -> Dict[str, str]:
        """
        Get gesture display names (gesture_key -> "Letter A", etc.)
        """
        result = {}
        
        for gesture_key, meta in self.metadata.items():
            if category and meta["category"] != category:
                continue
            
            # Lazy load just to get name
            if not meta["loaded"]:
                self._load_gesture(gesture_key)
            
            if gesture_key in self.cache:
                result[gesture_key] = self.cache[gesture_key].get("name", gesture_key.upper())
        
        return result
    
    def _load_gesture(self, gesture_key: str) -> bool:
        """
        Load a single gesture from disk (lazy loading).
        """
        if gesture_key not in self.metadata:
            return False
        
        meta = self.metadata[gesture_key]
        
        try:
            with open(meta["file"], "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.cache[gesture_key] = data
            meta["loaded"] = True
            meta["name"] = data.get("name", gesture_key.upper())
            
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load {gesture_key}: {e}")
            meta["loaded"] = True  # Mark as attempted
            return False
    
    def unload(self, gesture_key: str) -> None:
        """
        Remove gesture from cache (frees memory).
        """
        if gesture_key in self.cache:
            del self.cache[gesture_key]
            self.metadata[gesture_key]["loaded"] = False
    
    def unload_category(self, category: str) -> None:
        """
        Unload entire category (frees memory).
        """
        for gesture_key, meta in self.metadata.items():
            if meta["category"] == category:
                self.unload(gesture_key)
    
    def get_stats(self) -> Dict:
        """
        Get statistics about loaded gestures.
        """
        loaded_count = sum(1 for meta in self.metadata.values() if meta["loaded"])
        total_count = len(self.metadata)
        memory_usage = sum(
            len(json.dumps(self.cache[key]).encode())
            for key in self.cache
        ) / 1024  # KB
        
        by_category = {}
        for gesture_key, meta in self.metadata.items():
            cat = meta["category"]
            if cat not in by_category:
                by_category[cat] = {"total": 0, "loaded": 0}
            by_category[cat]["total"] += 1
            if meta["loaded"]:
                by_category[cat]["loaded"] += 1
        
        return {
            "total_gestures": total_count,
            "loaded_gestures": loaded_count,
            "cache_size_kb": memory_usage,
            "by_category": by_category
        }
    
    def print_stats(self):
        """Pretty print statistics"""
        stats = self.get_stats()
        print(f"\n[STATS] Gesture Manager:")
        print(f"  Total gestures: {stats['total_gestures']}")
        print(f"  Loaded: {stats['loaded_gestures']}")
        print(f"  Cache size: {stats['cache_size_kb']:.1f} KB")
        print(f"  By category:")
        for cat, info in stats['by_category'].items():
            print(f"    {cat:12} {info['loaded']:2}/{info['total']:2} loaded")
