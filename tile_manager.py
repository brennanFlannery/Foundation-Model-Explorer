"""
tile_manager.py
===============

Tile management system for adaptive multi-resolution slide viewing.

This module provides infrastructure for loading, caching, and displaying
whole-slide image tiles at multiple resolution levels. It uses OpenSlide's
DeepZoomGenerator to access the image pyramid and implements an LRU cache
to manage memory usage efficiently.

The TileManager handles the complexity of:
- Determining the appropriate pyramid level for the current zoom
- Calculating which tiles are visible in the viewport
- Asynchronously loading tiles without blocking the UI
- Caching tiles for fast re-display when panning

Classes
-------

TileManager
    Main manager class coordinating tile loading and caching.

TileLoaderWorker
    Worker that runs in a separate thread to load tiles asynchronously.

LRUTileCache
    Memory-limited cache with least-recently-used eviction policy.
"""
from __future__ import annotations

import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from queue import PriorityQueue, Empty
from threading import Lock
from typing import Dict, List, Optional, Set, Tuple

from PySide6.QtCore import QObject, QThread, Signal, QRectF, QPointF, QMutex, QMutexLocker
from PySide6.QtGui import QImage, QPixmap

try:
    import openslide
    from openslide.deepzoom import DeepZoomGenerator
except ImportError:
    openslide = None
    DeepZoomGenerator = None

from PIL import Image
import numpy as np


@dataclass
class TileRequest:
    """Request for a single tile to be loaded."""
    level: int
    col: int
    row: int
    generation: int  # Used to cancel stale requests


@dataclass(order=True)
class PrioritizedTileRequest:
    """Tile request with priority for queue ordering.
    
    Lower priority values are processed first (closer to viewport center).
    """
    priority: float  # Lower = higher priority (distance from center)
    request: TileRequest = field(compare=False)  # Exclude from comparison


@dataclass
class TileResult:
    """Result of loading a tile."""
    level: int
    col: int
    row: int
    generation: int
    image: Optional[Image.Image]
    error: Optional[str] = None


class LRUTileCache:
    """Memory-limited LRU cache for tile pixmaps.
    
    This cache stores QPixmap tiles with a configurable memory limit.
    When the limit is exceeded, the least recently used tiles are evicted.
    
    Attributes
    ----------
    max_bytes : int
        Maximum cache size in bytes.
    current_bytes : int
        Current estimated memory usage.
    """
    
    def __init__(self, max_mb: int = 250):
        """Initialize the cache with a memory limit.
        
        Parameters
        ----------
        max_mb : int
            Maximum cache size in megabytes.
        """
        self.max_bytes = max_mb * 1024 * 1024
        self.current_bytes = 0
        self._cache: OrderedDict[Tuple[int, int, int], QPixmap] = OrderedDict()
        self._sizes: Dict[Tuple[int, int, int], int] = {}
        self._lock = Lock()
    
    def get(self, key: Tuple[int, int, int]) -> Optional[QPixmap]:
        """Get a tile from cache, marking it as recently used.
        
        Parameters
        ----------
        key : Tuple[int, int, int]
            Cache key as (level, col, row).
            
        Returns
        -------
        Optional[QPixmap]
            The cached pixmap, or None if not in cache.
        """
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                return self._cache[key]
            return None
    
    def put(self, key: Tuple[int, int, int], pixmap: QPixmap) -> None:
        """Store a tile in the cache, evicting old tiles if needed.
        
        Parameters
        ----------
        key : Tuple[int, int, int]
            Cache key as (level, col, row).
        pixmap : QPixmap
            The tile pixmap to cache.
        """
        # Estimate pixmap memory: width * height * 4 bytes (RGBA)
        size_bytes = pixmap.width() * pixmap.height() * 4
        
        with self._lock:
            # If already in cache, remove old entry first
            if key in self._cache:
                self.current_bytes -= self._sizes[key]
                del self._cache[key]
                del self._sizes[key]
            
            # Evict LRU entries until we have room
            while self.current_bytes + size_bytes > self.max_bytes and self._cache:
                oldest_key, _ = self._cache.popitem(last=False)
                self.current_bytes -= self._sizes.pop(oldest_key)
            
            # Add new entry
            self._cache[key] = pixmap
            self._sizes[key] = size_bytes
            self.current_bytes += size_bytes
    
    def clear(self) -> None:
        """Clear all tiles from the cache."""
        with self._lock:
            self._cache.clear()
            self._sizes.clear()
            self.current_bytes = 0
    
    def contains(self, key: Tuple[int, int, int]) -> bool:
        """Check if a tile is in the cache."""
        with self._lock:
            return key in self._cache


class TileLoaderWorker(QObject):
    """Worker object that loads tiles in a background thread.
    
    This worker receives tile requests via a priority queue, loads them from
    the DeepZoomGenerator, and emits signals when tiles are ready. Tiles
    closer to the viewport center are processed first.
    
    Signals
    -------
    tile_loaded : Signal(int, int, int, int, QPixmap)
        Emitted when a tile is successfully loaded.
        Parameters: level, col, row, generation, pixmap
    tile_error : Signal(int, int, int, str)
        Emitted when tile loading fails.
        Parameters: level, col, row, error message
    """
    
    tile_loaded = Signal(int, int, int, int, object)  # level, col, row, generation, QPixmap
    tile_error = Signal(int, int, int, str)
    finished = Signal()
    
    def __init__(self, dz_generator: DeepZoomGenerator, request_queue: PriorityQueue):
        """Initialize the worker.
        
        Parameters
        ----------
        dz_generator : DeepZoomGenerator
            The DeepZoom generator to load tiles from.
        request_queue : PriorityQueue
            Priority queue to receive tile requests from.
        """
        super().__init__()
        self.dz = dz_generator
        self.queue = request_queue
        self._running = True
    
    def run(self) -> None:
        """Main worker loop - process tile requests until stopped."""
        while self._running:
            try:
                item = self.queue.get(timeout=0.1)
                
                # Handle both PrioritizedTileRequest and raw TileRequest
                if isinstance(item, PrioritizedTileRequest):
                    request = item.request
                else:
                    request = item
                
                self._load_tile(request)
                
            except Empty:
                continue
            except Exception as e:
                print(f"TileLoaderWorker error: {e}")
        
        self.finished.emit()
    
    def _load_tile(self, request: TileRequest) -> None:
        """Load a single tile and emit the result."""
        try:
            # Get tile from DeepZoom generator
            # DeepZoom levels are numbered differently - level 0 is smallest
            tile = self.dz.get_tile(request.level, (request.col, request.row))
            
            # Convert PIL image to QPixmap
            if tile.mode != 'RGBA':
                tile = tile.convert('RGBA')
            
            data = tile.tobytes('raw', 'RGBA')
            qimage = QImage(data, tile.width, tile.height, QImage.Format_RGBA8888)
            pixmap = QPixmap.fromImage(qimage)
            
            self.tile_loaded.emit(
                request.level, request.col, request.row,
                request.generation, pixmap
            )
            
        except Exception as e:
            self.tile_error.emit(
                request.level, request.col, request.row,
                str(e)
            )
    
    def stop(self) -> None:
        """Signal the worker to stop processing."""
        self._running = False


class TileManager(QObject):
    """Manager for loading and caching multi-resolution slide tiles.
    
    The TileManager coordinates between the OpenSlide image, the DeepZoom
    generator, the tile cache, and the background loading thread. It
    provides the interface for the SlideGraphicsView to request tiles
    and receive them asynchronously.
    
    Signals
    -------
    tile_loaded : Signal(int, int, int, QPixmap)
        Emitted when a tile is ready to display.
        Parameters: level, col, row, pixmap
    loading_started : Signal()
        Emitted when tile loading begins.
    loading_finished : Signal()
        Emitted when all pending tiles are loaded.
    
    Attributes
    ----------
    slide_dimensions : Tuple[int, int]
        Full resolution dimensions (width, height) of the slide.
    level_count : int
        Number of pyramid levels available.
    tile_size : int
        Size of each tile in pixels.
    """
    
    tile_loaded = Signal(int, int, int, object)  # level, col, row, QPixmap
    loading_started = Signal()
    loading_finished = Signal()
    
    def __init__(self, slide_path: str, tile_size: int = 256, cache_mb: int = 250):
        """Initialize the TileManager for a slide.
        
        Parameters
        ----------
        slide_path : str
            Path to the whole-slide image file.
        tile_size : int
            Size of tiles to generate (default 256).
        cache_mb : int
            Maximum cache size in megabytes (default 250).
            
        Raises
        ------
        RuntimeError
            If OpenSlide is not available or slide cannot be opened.
        """
        super().__init__()
        
        if openslide is None:
            raise RuntimeError("OpenSlide is required for adaptive zoom but is not installed.")
        
        self.slide_path = slide_path
        self.tile_size = tile_size
        
        # Open slide and create DeepZoom generator
        self.slide = openslide.OpenSlide(slide_path)
        self.dz = DeepZoomGenerator(self.slide, tile_size=tile_size, overlap=0, limit_bounds=True)
        
        # Store slide properties
        self.slide_dimensions = self.slide.dimensions
        self.level_count = self.dz.level_count
        
        # Initialize cache
        self.cache = LRUTileCache(max_mb=cache_mb)
        
        # Request tracking
        self._generation = 0  # Incremented when view changes significantly
        self._pending_requests: Set[Tuple[int, int, int]] = set()
        self._pending_lock = Lock()
        
        # Background loading thread with priority queue
        self._request_queue: PriorityQueue = PriorityQueue()
        self._worker: Optional[TileLoaderWorker] = None
        self._thread: Optional[QThread] = None
        self._start_worker()
        
        print(f"TileManager initialized: {self.slide_dimensions[0]}x{self.slide_dimensions[1]}, "
              f"{self.level_count} levels, tile size {tile_size}")
    
    def _start_worker(self) -> None:
        """Start the background tile loading thread."""
        self._thread = QThread()
        self._worker = TileLoaderWorker(self.dz, self._request_queue)
        self._worker.moveToThread(self._thread)
        
        # Connect signals
        self._thread.started.connect(self._worker.run)
        self._worker.tile_loaded.connect(self._on_tile_loaded)
        self._worker.tile_error.connect(self._on_tile_error)
        self._worker.finished.connect(self._thread.quit)
        
        self._thread.start()
    
    def get_level_for_scale(self, view_scale: float) -> int:
        """Determine the appropriate DeepZoom level for a view scale.
        
        The view scale is the ratio of screen pixels to scene (level-0) pixels.
        A scale of 1.0 means 1:1 viewing of full resolution.
        A scale of 0.1 means the view is zoomed out 10x.
        
        Parameters
        ----------
        view_scale : float
            Current view scale (screen pixels / level-0 pixels).
            
        Returns
        -------
        int
            The DeepZoom level to use (0 = smallest, level_count-1 = largest).
        """
        # DeepZoom levels: 0 is smallest (1x1), level_count-1 is full resolution
        # Each level is 2x the previous
        
        # At scale 1.0, we want the highest level (full res)
        # At scale 0.5, we want one level down (half res)
        # At scale 0.25, we want two levels down, etc.
        
        # Calculate which level gives us approximately 1:1 screen pixels to tile pixels
        if view_scale <= 0:
            return 0
        
        # log2(1/scale) tells us how many levels down from full res
        import math
        levels_down = max(0, -math.log2(view_scale))
        target_level = self.level_count - 1 - int(levels_down)
        
        # Clamp to valid range
        return max(0, min(self.level_count - 1, target_level))
    
    def get_level_downsample(self, level: int) -> float:
        """Get the downsample factor for a DeepZoom level.
        
        Parameters
        ----------
        level : int
            DeepZoom level (0 = smallest).
            
        Returns
        -------
        float
            Factor by which this level is downsampled from level-0.
        """
        # DeepZoom: level 0 is smallest, each level is 2x larger
        # Level (level_count - 1) is full resolution (downsample = 1)
        levels_from_full = self.level_count - 1 - level
        return 2 ** levels_from_full
    
    def get_level_dimensions(self, level: int) -> Tuple[int, int]:
        """Get the dimensions of a specific DeepZoom level.
        
        Parameters
        ----------
        level : int
            DeepZoom level.
            
        Returns
        -------
        Tuple[int, int]
            Width and height of the level in pixels.
        """
        return self.dz.level_dimensions[level]
    
    def get_level_tiles(self, level: int) -> Tuple[int, int]:
        """Get the tile grid size for a level.
        
        Parameters
        ----------
        level : int
            DeepZoom level.
            
        Returns
        -------
        Tuple[int, int]
            Number of tile columns and rows at this level.
        """
        return self.dz.level_tiles[level]
    
    def get_tile_bounds_lv0(self, level: int, col: int, row: int) -> QRectF:
        """Get the level-0 coordinate bounds for a tile.
        
        Parameters
        ----------
        level : int
            DeepZoom level of the tile.
        col : int
            Column index of the tile.
        row : int
            Row index of the tile.
            
        Returns
        -------
        QRectF
            Bounding rectangle in level-0 coordinates.
        """
        downsample = self.get_level_downsample(level)
        
        # Tile position at this level
        tile_x = col * self.tile_size
        tile_y = row * self.tile_size
        
        # Get actual tile dimensions (edge tiles may be smaller)
        level_dims = self.get_level_dimensions(level)
        tile_w = min(self.tile_size, level_dims[0] - tile_x)
        tile_h = min(self.tile_size, level_dims[1] - tile_y)
        
        # Scale to level-0 coordinates
        lv0_x = tile_x * downsample
        lv0_y = tile_y * downsample
        lv0_w = tile_w * downsample
        lv0_h = tile_h * downsample
        
        return QRectF(lv0_x, lv0_y, lv0_w, lv0_h)
    
    def get_visible_tiles(self, viewport_lv0: QRectF, level: int) -> List[Tuple[int, int, int]]:
        """Calculate which tiles are visible in a viewport.
        
        Parameters
        ----------
        viewport_lv0 : QRectF
            Viewport rectangle in level-0 coordinates.
        level : int
            DeepZoom level to get tiles for.
            
        Returns
        -------
        List[Tuple[int, int, int]]
            List of (level, col, row) tuples for visible tiles.
        """
        downsample = self.get_level_downsample(level)
        level_tiles = self.get_level_tiles(level)
        
        # Convert viewport to level coordinates
        level_left = viewport_lv0.left() / downsample
        level_top = viewport_lv0.top() / downsample
        level_right = viewport_lv0.right() / downsample
        level_bottom = viewport_lv0.bottom() / downsample
        
        # Calculate tile range
        col_start = max(0, int(level_left / self.tile_size))
        col_end = min(level_tiles[0], int(level_right / self.tile_size) + 1)
        row_start = max(0, int(level_top / self.tile_size))
        row_end = min(level_tiles[1], int(level_bottom / self.tile_size) + 1)
        
        tiles = []
        for col in range(col_start, col_end):
            for row in range(row_start, row_end):
                tiles.append((level, col, row))
        
        return tiles
    
    def get_prefetch_tiles(self, viewport_lv0: QRectF, level: int) -> List[Tuple[int, int, int]]:
        """Get tiles just outside the viewport for prefetching.
        
        Returns tiles in a one-tile-wide ring around the visible area.
        These tiles are likely to become visible soon when panning.
        
        Parameters
        ----------
        viewport_lv0 : QRectF
            Viewport rectangle in level-0 coordinates.
        level : int
            DeepZoom level to get tiles for.
            
        Returns
        -------
        List[Tuple[int, int, int]]
            List of (level, col, row) tuples for prefetch tiles.
        """
        downsample = self.get_level_downsample(level)
        
        # Calculate tile size in level-0 coordinates
        tile_size_lv0 = self.tile_size * downsample
        
        # Expand viewport by one tile in each direction
        expanded_viewport = viewport_lv0.adjusted(
            -tile_size_lv0, -tile_size_lv0,
            tile_size_lv0, tile_size_lv0
        )
        
        # Clamp to slide bounds
        slide_rect = QRectF(0, 0, self.slide_dimensions[0], self.slide_dimensions[1])
        expanded_viewport = expanded_viewport.intersected(slide_rect)
        
        # Get all tiles in expanded area
        all_tiles = set(self.get_visible_tiles(expanded_viewport, level))
        
        # Get visible tiles
        visible_tiles = set(self.get_visible_tiles(viewport_lv0, level))
        
        # Return only the tiles outside the visible area
        prefetch_tiles = list(all_tiles - visible_tiles)
        
        return prefetch_tiles
    
    def _calculate_tile_priority(self, level: int, col: int, row: int, 
                                  viewport_center: QPointF) -> float:
        """Calculate priority for a tile based on distance from viewport center.
        
        Lower values indicate higher priority (closer to center).
        
        Parameters
        ----------
        level : int
            DeepZoom level of the tile.
        col : int
            Column index of the tile.
        row : int
            Row index of the tile.
        viewport_center : QPointF
            Center point of the viewport in level-0 coordinates.
            
        Returns
        -------
        float
            Squared distance from tile center to viewport center.
        """
        tile_bounds = self.get_tile_bounds_lv0(level, col, row)
        tile_center = tile_bounds.center()
        
        dx = tile_center.x() - viewport_center.x()
        dy = tile_center.y() - viewport_center.y()
        
        # Return squared distance (no sqrt needed for comparison)
        return dx * dx + dy * dy
    
    def request_tiles_prioritized(self, tiles: List[Tuple[int, int, int]], 
                                   viewport_center: QPointF,
                                   priority_offset: float = 0.0) -> None:
        """Request tiles with priority based on distance from viewport center.
        
        Tiles closer to the viewport center are loaded first. This improves
        perceived performance by ensuring the most visible content loads first.
        
        Parameters
        ----------
        tiles : List[Tuple[int, int, int]]
            List of (level, col, row) tuples to load.
        viewport_center : QPointF
            Center point of the viewport in level-0 coordinates.
        priority_offset : float
            Additional priority offset (higher = lower priority). Used to
            deprioritize prefetch tiles relative to visible tiles.
        """
        tiles_to_load = []
        
        for level, col, row in tiles:
            key = (level, col, row)
            
            # Check cache first
            cached = self.cache.get(key)
            if cached is not None:
                # Emit immediately from cache
                self.tile_loaded.emit(level, col, row, cached)
            else:
                # Need to load this tile
                with self._pending_lock:
                    if key not in self._pending_requests:
                        self._pending_requests.add(key)
                        # Calculate priority
                        priority = self._calculate_tile_priority(level, col, row, viewport_center)
                        priority += priority_offset
                        tiles_to_load.append((priority, level, col, row))
        
        # Queue uncached tiles for loading with priority
        if tiles_to_load:
            self.loading_started.emit()
            for priority, level, col, row in tiles_to_load:
                request = TileRequest(level, col, row, self._generation)
                prioritized = PrioritizedTileRequest(priority=priority, request=request)
                self._request_queue.put(prioritized)
    
    def request_tiles(self, tiles: List[Tuple[int, int, int]]) -> None:
        """Request tiles to be loaded.
        
        Tiles already in cache are returned immediately via tile_loaded signal.
        Tiles not in cache are queued for background loading.
        
        Parameters
        ----------
        tiles : List[Tuple[int, int, int]]
            List of (level, col, row) tuples to load.
        """
        tiles_to_load = []
        
        for level, col, row in tiles:
            key = (level, col, row)
            
            # Check cache first
            cached = self.cache.get(key)
            if cached is not None:
                # Emit immediately from cache
                self.tile_loaded.emit(level, col, row, cached)
            else:
                # Need to load this tile
                with self._pending_lock:
                    if key not in self._pending_requests:
                        self._pending_requests.add(key)
                        tiles_to_load.append(key)
        
        # Queue uncached tiles for loading
        if tiles_to_load:
            self.loading_started.emit()
            for level, col, row in tiles_to_load:
                request = TileRequest(level, col, row, self._generation)
                self._request_queue.put(request)
    
    def cancel_pending(self) -> None:
        """Cancel all pending tile requests.
        
        This increments the generation counter so that any in-flight
        tiles will be ignored when they arrive.
        """
        self._generation += 1
        with self._pending_lock:
            self._pending_requests.clear()
        
        # Clear the queue
        while not self._request_queue.empty():
            try:
                self._request_queue.get_nowait()
            except Empty:
                break
    
    def _on_tile_loaded(self, level: int, col: int, row: int, 
                        generation: int, pixmap: QPixmap) -> None:
        """Handle a tile loaded by the worker."""
        # Ignore stale tiles
        if generation != self._generation:
            return
        
        key = (level, col, row)
        
        # Add to cache
        self.cache.put(key, pixmap)
        
        # Remove from pending
        with self._pending_lock:
            self._pending_requests.discard(key)
            pending_count = len(self._pending_requests)
        
        # Emit signal
        self.tile_loaded.emit(level, col, row, pixmap)
        
        # Check if all done
        if pending_count == 0:
            self.loading_finished.emit()
    
    def _on_tile_error(self, level: int, col: int, row: int, error: str) -> None:
        """Handle a tile loading error."""
        key = (level, col, row)
        with self._pending_lock:
            self._pending_requests.discard(key)
        print(f"Tile load error [{level}, {col}, {row}]: {error}")
    
    def clear_cache(self) -> None:
        """Clear all cached tiles."""
        self.cache.clear()
    
    def close(self) -> None:
        """Close the tile manager and release resources."""
        # Stop worker thread
        if self._worker:
            self._worker.stop()
            # Worker will exit naturally when _running becomes False
            # No poison pill needed - queue.get(timeout=0.1) will timeout
            # and the while loop will check _running and exit
        
        if self._thread:
            self._thread.quit()
            self._thread.wait(1000)
        
        # Clear cache
        self.cache.clear()
        
        # Close slide
        if self.slide:
            self.slide.close()
            self.slide = None
        
        print("TileManager closed")
