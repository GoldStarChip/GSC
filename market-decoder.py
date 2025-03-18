#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Market Decoder - Core Pattern Recognition System
Gold Star Chip Project

This module implements advanced signal processing algorithms to detect non-fundamental
manipulation patterns in financial market data. It identifies the digital fingerprints
of control systems operating beyond conventional market mechanics.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress harmless warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Constants for manipulation detection
GOLDEN_RATIO = 1.618033988749895
FIBONACCI_SEQ = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
CONTROL_FREQUENCIES = [1.618, 2.718, 3.141, 4.669, 6.022, 6.626]
QUANTUM_CORRELATION_THRESHOLD = 0.973
TEMPORAL_IMPOSSIBILITY_WINDOW_MS = 7  # milliseconds

@dataclass
class ManipulationSignature:
    """Data structure for identified manipulation patterns"""
    timestamp: pd.Timestamp
    pattern_type: str
    confidence: float
    affected_assets: List[str]
    vector_magnitude: float
    harmonic_pattern: List[float]
    control_frequency: Optional[float] = None
    puppet_master_id: Optional[str] = None


class PatternAnalyzer:
    """
    Core analysis system for detecting non-fundamental control patterns
    in financial market data.
    """
    
    def __init__(self, data_path: str, sensitivity: float = 0.9):
        """
        Initialize the pattern analyzer with market data.
        
        Args:
            data_path: Path to CSV file containing high-frequency market data
            sensitivity: Detection sensitivity (0.5-0.99)
        """
        self.sensitivity = sensitivity
        logger.info(f"Initializing PatternAnalyzer with sensitivity {sensitivity}")
        
        try:
            self.data = pd.read_csv(data_path, parse_dates=['timestamp'])
            logger.info(f"Loaded {len(self.data)} data points from {data_path}")
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
            
        self.known_patterns = self._load_known_patterns()
        self.puppet_string_registry = self._initialize_puppet_registry()
        
    def _load_known_patterns(self) -> Dict:
        """Load known manipulation pattern templates"""
        logger.info("Loading known manipulation pattern templates")
        
        # In a real implementation, these would be loaded from a database
        # Here we're hardcoding some pattern definitions for illustration
        return {
            "liquidity_vacuum": {
                "signature": [1.0, 0.618, 0.382, 0.236, 0.382, 0.618],
                "window": 6,
                "confidence_threshold": 0.85
            },
            "engineered_reversal": {
                "signature": [1.0, 1.618, 2.618, 4.236, 2.618, 1.618],
                "window": 6,
                "confidence_threshold": 0.82
            },
            "capital_harvesting": {
                "signature": [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.145, 0.236, 0.382],
                "window": 9,
                "confidence_threshold": 0.88
            },
            "temporal_impossibility": {
                "signature": [0.1, 0.2, 0.4, 0.8, 1.0, -0.5, -1.0],
                "window": 7,
                "confidence_threshold": 0.9
            },
            "quantum_correlation": {
                "signature": [0.1, 0.3, 0.5, 0.8, 1.0, 0.8, 0.5, 0.3, 0.1],
                "window": 9,
                "confidence_threshold": 0.95
            }
        }
        
    def _initialize_puppet_registry(self) -> Dict:
        """Initialize the registry of known control entities"""
        logger.info("Initializing puppet master registry")
        
        # This would normally be loaded from a secure database
        # Using placeholder data for illustration
        return {
            "PM-ALPHA": {
                "signature_frequency": 1.618,
                "control_mechanisms": ["liquidity_vacuum", "engineered_reversal"],
                "known_assets": ["SPY", "QQQ", "IWM"]
            },
            "PM-DELTA": {
                "signature_frequency": 2.718,
                "control_mechanisms": ["capital_harvesting", "temporal_impossibility"],
                "known_assets": ["GLD", "SLV", "USO"]
            },
            "PM-OMEGA": {
                "signature_frequency": 3.141,
                "control_mechanisms": ["quantum_correlation"],
                "known_assets": ["BTC", "ETH", "XRP"]
            }
        }
    
    def detect_puppet_strings(self) -> List[ManipulationSignature]:
        """
        Core detection algorithm for identifying non-fundamental control
        patterns in financial data.
        
        Returns:
            List of detected manipulation signatures
        """
        logger.info("Beginning manipulation signature detection")
        
        results = []
        window_sizes = [p["window"] for p in self.known_patterns.values()]
        max_window = max(window_sizes)
        
        # Ensure we have enough data
        if len(self.data) < max_window * 2:
            logger.warning(f"Insufficient data points: {len(self.data)} < {max_window * 2}")
            return results
            
        # Normalize price data for pattern matching
        normalized_data = self._normalize_data()
        
        # Detect known patterns
        for symbol in normalized_data.columns:
            if symbol == 'timestamp':
                continue
                
            logger.info(f"Analyzing asset: {symbol}")
            asset_data = normalized_data[symbol].values
            
            for pattern_name, pattern_info in self.known_patterns.items():
                window = pattern_info["window"]
                signature = pattern_info["signature"]
                threshold = pattern_info["confidence_threshold"]
                
                # Sliding window analysis
                for i in range(len(asset_data) - window):
                    segment = asset_data[i:i+window]
                    
                    # Skip segments with NaN values
                    if np.isnan(segment).any():
                        continue
                        
                    # Calculate correlation with known pattern
                    correlation = self._calculate_pattern_similarity(segment, signature)
                    
                    if correlation >= threshold * self.sensitivity:
                        # Pattern detected
                        timestamp = self.data['timestamp'].iloc[i+window-1]
                        
                        # Perform frequency analysis to identify control system
                        freq_data = asset_data[max(0, i-20):min(len(asset_data), i+window+20)]
                        control_freq = self._detect_control_frequency(freq_data)
                        
                        # Match to potential puppet master
                        puppet_master = self._identify_puppet_master(control_freq, pattern_name, symbol)
                        
                        # Create signature object
                        signature = ManipulationSignature(
                            timestamp=timestamp,
                            pattern_type=pattern_name,
                            confidence=correlation,
                            affected_assets=[symbol],
                            vector_magnitude=np.std(segment),
                            harmonic_pattern=list(segment),
                            control_frequency=control_freq,
                            puppet_master_id=puppet_master
                        )
                        
                        results.append(signature)
                        logger.info(f"Detected {pattern_name} pattern in {symbol} at {timestamp} " +
                                   f"(confidence: {correlation:.4f}, puppet master: {puppet_master})")
        
        # Cluster related patterns affecting multiple assets
        clustered_results = self._cluster_correlated_patterns(results)
        
        logger.info(f"Detection complete. Found {len(clustered_results)} manipulation signatures")
        return clustered_results
    
    def _normalize_data(self) -> pd.DataFrame:
        """Normalize price data for pattern matching"""
        logger.info("Normalizing data for pattern detection")
        
        normalized = self.data.copy()
        
        for column in normalized.columns:
            if column == 'timestamp':
                continue
                
            # Apply wavelet denoising to filter out organic market noise
            prices = normalized[column].values
            if np.isnan(prices).all():
                continue
                
            # Fill NaN values with last valid observation
            prices = pd.Series(prices).fillna(method='ffill').values
            
            # Normalize to range [0,1] for pattern matching
            min_val, max_val = np.nanmin(prices), np.nanmax(prices)
            if max_val > min_val:
                normalized[column] = (prices - min_val) / (max_val - min_val)
            else:
                normalized[column] = 0
                
        return normalized
    
    def _calculate_pattern_similarity(self, segment: np.ndarray, pattern: List[float]) -> float:
        """
        Calculate the similarity between a data segment and a known pattern.
        
        Args:
            segment: Normalized price data segment
            pattern: Reference pattern to compare against
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize segment to match pattern scale
        segment = segment / np.max(np.abs(segment)) if np.max(np.abs(segment)) > 0 else segment
        
        # Calculate correlation coefficient
        pattern_array = np.array(pattern)
        segment_array = segment[:len(pattern_array)]
        
        if len(segment_array) != len(pattern_array):
            # Pad or truncate segment to match pattern length
            if len(segment_array) < len(pattern_array):
                segment_array = np.pad(segment_array, 
                                       (0, len(pattern_array) - len(segment_array)), 
                                       'constant')
            else:
                segment_array = segment_array[:len(pattern_array)]
        
        # Calculate correlation
        correlation = np.corrcoef(segment_array, pattern_array)[0, 1]
        
        # Handle NaN correlation
        if np.isnan(correlation):
            return 0.0
            
        # Convert to absolute value and scale to [0,1]
        return (np.abs(correlation) + 1) / 2
    
    def _detect_control_frequency(self, data: np.ndarray) -> Optional[float]:
        """
        Detect control frequencies in the data using FFT.
        
        Args:
            data: Time series data segment
            
        Returns:
            Dominant frequency if it matches known control frequencies
        """
        if len(data) < 10:
            return None
            
        # Apply FFT to detect frequency components
        fft_data = fft(data)
        magnitude = np.abs(fft_data)
        
        # Find dominant frequency
        freq_idx = np.argmax(magnitude[1:len(magnitude)//2]) + 1
        
        # Calculate actual frequency value
        freq_value = freq_idx / len(data)
        
        # Check if it matches known control frequencies
        for cf in CONTROL_FREQUENCIES:
            if abs(freq_value - cf) < 0.1:
                return cf
                
        return None
    
    def _identify_puppet_master(self, frequency: Optional[float], 
                               pattern_type: str, asset: str) -> Optional[str]:
        """
        Identify the puppet master based on frequency signature and pattern type.
        
        Args:
            frequency: Detected control frequency
            pattern_type: Type of manipulation pattern
            asset: Affected financial asset
            
        Returns:
            Puppet master identifier if matched, None otherwise
        """
        if frequency is None:
            return None
            
        for pm_id, pm_info in self.puppet_string_registry.items():
            # Check if frequency matches
            if abs(frequency - pm_info["signature_frequency"]) < 0.1:
                # Check if pattern type is in puppet master's known mechanisms
                if pattern_type in pm_info["control_mechanisms"]:
                    # Check if asset is in puppet master's known controlled assets
                    if asset in pm_info["known_assets"]:
                        return pm_id
                    # Asset not in known list, but other criteria match
                    return f"{pm_id}?"
        
        # No match found
        return None
    
    def _cluster_correlated_patterns(self, 
                                    patterns: List[ManipulationSignature]) -> List[ManipulationSignature]:
        """
        Cluster related patterns that affect multiple assets simultaneously.
        
        Args:
            patterns: List of detected manipulation signatures
            
        Returns:
            List of clustered manipulation signatures
        """
        if not patterns:
            return []
            
        logger.info("Clustering correlated patterns across assets")
        
        # Sort patterns by timestamp
        sorted_patterns = sorted(patterns, key=lambda x: x.timestamp)
        
        # Initialize clusters
        clusters = []
        current_cluster = [sorted_patterns[0]]
        
        # Time window for clustering (nanoseconds)
        cluster_window_ns = 50_000_000  # 50 milliseconds
        
        # Cluster patterns that occur within the time window
        for i in range(1, len(sorted_patterns)):
            current_pattern = sorted_patterns[i]
            previous_pattern = current_cluster[-1]
            
            # Check if within time window
            time_diff = (current_pattern.timestamp - previous_pattern.timestamp).total_seconds() * 1_000_000_000
            
            if time_diff <= cluster_window_ns:
                # Add to current cluster
                current_cluster.append(current_pattern)
            else:
                # Finalize current cluster
                if len(current_cluster) > 1:
                    # Merge patterns in cluster
                    merged_pattern = self._merge_cluster(current_cluster)
                    clusters.append(merged_pattern)
                else:
                    # Single pattern cluster
                    clusters.append(current_cluster[0])
                
                # Start new cluster
                current_cluster = [current_pattern]
        
        # Handle final cluster
        if current_cluster:
            if len(current_cluster) > 1:
                merged_pattern = self._merge_cluster(current_cluster)
                clusters.append(merged_pattern)
            else:
                clusters.append(current_cluster[0])
        
        logger.info(f"Clustered {len(patterns)} patterns into {len(clusters)} signatures")
        return clusters
    
    def _merge_cluster(self, cluster: List[ManipulationSignature]) -> ManipulationSignature:
        """
        Merge a cluster of related patterns into a single signature.
        
        Args:
            cluster: List of related manipulation signatures
            
        Returns:
            Merged manipulation signature
        """
        # Use the most recent timestamp
        timestamp = max(p.timestamp for p in cluster)
        
        # Use the most common pattern type
        pattern_counts = {}
        for p in cluster:
            pattern_counts[p.pattern_type] = pattern_counts.get(p.pattern_type, 0) + 1
        pattern_type = max(pattern_counts.items(), key=lambda x: x[1])[0]
        
        # Average confidence
        confidence = sum(p.confidence for p in cluster) / len(cluster)
        
        # Combine affected assets
        affected_assets = []
        for p in cluster:
            affected_assets.extend(p.affected_assets)
        affected_assets = list(set(affected_assets))  # Remove duplicates
        
        # Average vector magnitude
        vector_magnitude = sum(p.vector_magnitude for p in cluster) / len(cluster)
        
        # Use harmonic pattern from highest confidence pattern
        highest_conf_pattern = max(cluster, key=lambda p: p.confidence)
        harmonic_pattern = highest_conf_pattern.harmonic_pattern
        
        # Most common control frequency
        freq_counts = {}
        for p in cluster:
            if p.control_frequency is not None:
                freq_counts[p.control_frequency] = freq_counts.get(p.control_frequency, 0) + 1
        
        control_frequency = None
        if freq_counts:
            control_frequency = max(freq_counts.items(), key=lambda x: x[1])[0]
        
        # Most likely puppet master
        pm_counts = {}
        for p in cluster:
            if p.puppet_master_id is not None:
                pm_counts[p.puppet_master_id] = pm_counts.get(p.puppet_master_id, 0) + 1
        
        puppet_master_id = None
        if pm_counts:
            puppet_master_id = max(pm_counts.items(), key=lambda x: x[1])[0]
        
        return ManipulationSignature(
            timestamp=timestamp,
            pattern_type=pattern_type,
            confidence=confidence,
            affected_assets=affected_assets,
            vector_magnitude=vector_magnitude,
            harmonic_pattern=harmonic_pattern,
            control_frequency=control_frequency,
            puppet_master_id=puppet_master_id
        )
    
    def visualize_control_grid(self, 
                              manipulation_events: List[ManipulationSignature],
                              output_path: Optional[str] = None) -> None:
        """
        Visualize the detected manipulation events and control grid.
        
        Args:
            manipulation_events: List of detected manipulation signatures
            output_path: Path to save the visualization image
        """
        if not manipulation_events:
            logger.warning("No manipulation events to visualize")
            return
            
        logger.info("Generating control grid visualization")
        
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Plot 1: Timeline of manipulation events
        plt.subplot(3, 1, 1)
        
        # Extract timestamps and confidence levels
        timestamps = [e.timestamp for e in manipulation_events]
        confidences = [e.confidence for e in manipulation_events]
        pattern_types = [e.pattern_type for e in manipulation_events]
        
        # Create scatter plot
        pattern_colors = {
            "liquidity_vacuum": "red",
            "engineered_reversal": "purple",
            "capital_harvesting": "orange",
            "temporal_impossibility": "blue",
            "quantum_correlation": "green"
        }
        
        colors = [pattern_colors.get(pt, "gray") for pt in pattern_types]
        
        plt.scatter(timestamps, confidences, c=colors, alpha=0.7, s=50)
        plt.xlabel("Timestamp")
        plt.ylabel("Confidence")
        plt.title("Timeline of Detected Manipulation Events")
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Affected assets heatmap
        plt.subplot(3, 1, 2)
        
        # Get unique assets
        all_assets = set()
        for e in manipulation_events:
            all_assets.update(e.affected_assets)
        all_assets = sorted(list(all_assets))
        
        # Create asset-time matrix
        asset_time_matrix = np.zeros((len(all_assets), len(manipulation_events)))
        
        for i, event in enumerate(manipulation_events):
            for asset in event.affected_assets:
                if asset in all_assets:
                    j = all_assets.index(asset)
                    asset_time_matrix[j, i] = event.confidence
        
        plt.imshow(asset_time_matrix, aspect='auto', cmap='inferno')
        plt.colorbar(label="Manipulation Confidence")
        plt.yticks(range(len(all_assets)), all_assets)
        plt.xlabel("Event Index")
        plt.ylabel("Affected Asset")
        plt.title("Asset Manipulation Patterns")
        
        # Plot 3: Puppet master activity
        plt.subplot(3, 1, 3)
        
        # Count events by puppet master
        pm_counts = {}
        for event in manipulation_events:
            pm = event.puppet_master_id or "Unknown"
            pm_counts[pm] = pm_counts.get(pm, 0) + 1
        
        # Sort by count
        pm_labels = sorted(pm_counts.keys(), key=lambda x: pm_counts[x], reverse=True)
        pm_values = [pm_counts[pm] for pm in pm_labels]
        
        plt.bar(pm_labels, pm_values, color='darkred')
        plt.xlabel("Puppet Master")
        plt.ylabel("Number of Manipulation Events")
        plt.title("Control Entity Activity")
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {output_path}")
        else:
            plt.show()
    
    def analyze_temporal_anomalies(self) -> Dict:
        """
        Analyze temporal anomalies where price movements precede their
        supposed catalysts.
        
        Returns:
            Dictionary with temporal anomaly statistics
        """
        logger.info("Analyzing temporal anomalies")
        
        # Check if we have the necessary columns
        required_columns = ['timestamp', 'price', 'news_timestamp']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            return {}
        
        # Find cases where price moves before news
        self.data['time_diff_ms'] = (self.data['timestamp'] - self.data['news_timestamp']).dt.total_seconds() * 1000
        
        # Identify precognitive price movements
        precognitive_events = self.data[self.data['time_diff_ms'] < -TEMPORAL_IMPOSSIBILITY_WINDOW_MS]
        
        if len(precognitive_events) == 0:
            logger.info("No temporal anomalies detected")
            return {
                "anomalies_detected": 0,
                "anomaly_percentage": 0,
                "average_precognition_ms": 0
            }
        
        # Calculate statistics
        anomaly_count = len(precognitive_events)
        anomaly_percentage = (anomaly_count / len(self.data)) * 100
        avg_precognition = abs(precognitive_events['time_diff_ms'].mean())
        
        results = {
            "anomalies_detected": anomaly_count,
            "anomaly_percentage": anomaly_percentage,
            "average_precognition_ms": avg_precognition,
            "precognitive_events": precognitive_events
        }
        
        logger.info(f"Detected {anomaly_count} temporal anomalies " +
                   f"({anomaly_percentage:.2f}% of data)")
        logger.info(f"Average precognition time: {avg_precognition:.2f} ms")
        
        return results


if __name__ == "__main__":
    # Example usage
    try:
        analyzer = PatternAnalyzer("sample_market_data.csv", sensitivity=0.85)
        manipulation_events = analyzer.detect_puppet_strings()
        
        print(f"Detected {len(manipulation_events)} manipulation events")
        
        # Visualize results
        analyzer.visualize_control_grid(manipulation_events, "manipulation_grid.png")
        
        # Analyze temporal anomalies
        anomalies = analyzer.analyze_temporal_anomalies()
        print(f"Temporal anomalies: {anomalies.get('anomalies_detected', 0)}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
