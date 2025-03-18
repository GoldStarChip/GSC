#!/usr/bin/env python3
"""
Manipulation Harmonic Detector
Gold Star Chip Project

Identifies Golden Ratio proportions in engineered price movements
to detect mathematically precise market manipulation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HarmonicDetector")

# Constants
GOLDEN_RATIO = 1.618033988749895
FIBONACCI_LEVELS = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618, 4.236]
TOLERANCE = 0.0021  # Precision tolerance
MIN_SWING_POINTS = 5  # Minimum swing points for a valid pattern

class HarmonicDetector:
    """Detects harmonic manipulation patterns in price data"""
    
    def __init__(self, sensitivity=0.9):
        """Initialize the detector with sensitivity level"""
        self.sensitivity = sensitivity
        self.tolerance = TOLERANCE / sensitivity
        self.detected_patterns = []
        logger.info(f"Initialized Harmonic Detector with sensitivity {sensitivity}")
        
    def analyze(self, price_data, symbol=None):
        """
        Analyze price data for harmonic patterns
        
        Args:
            price_data: DataFrame or array of price data
            symbol: Optional symbol name for identification
            
        Returns:
            List of detected harmonic patterns
        """
        logger.info(f"Analyzing {'symbol ' + symbol if symbol else 'price data'}")
        
        if isinstance(price_data, pd.DataFrame):
            # Extract price from DataFrame
            if 'close' in price_data.columns:
                prices = price_data['close'].values
            else:
                prices = price_data.iloc[:, 0].values
            
            timestamps = price_data.index if isinstance(price_data.index, pd.DatetimeIndex) else None
        else:
            # Assume numpy array or list
            prices = np.array(price_data)
            timestamps = None
        
        # Find swing points (local minima and maxima)
        swing_points = self._find_swing_points(prices)
        
        if len(swing_points) < MIN_SWING_POINTS:
            logger.info(f"Insufficient swing points detected: {len(swing_points)}")
            return []
        
        # Analyze swing points for harmonic patterns
        self.detected_patterns = self._analyze_harmonics(
            prices, swing_points, timestamps, symbol
        )
        
        pattern_count = len(self.detected_patterns)
        logger.info(f"Detection complete. Found {pattern_count} harmonic patterns")
        
        return self.detected_patterns
    
    def export_results(self, output_path):
        """Export detected patterns to JSON file"""
        if not self.detected_patterns:
            logger.warning("No patterns to export")
            return False
            
        try:
            with open(output_path, 'w') as f:
                json.dump({
                    'metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'sensitivity': self.sensitivity,
                        'tolerance': self.tolerance,
                        'pattern_count': len(self.detected_patterns)
                    },
                    'patterns': self.detected_patterns
                }, f, indent=2, default=str)
                
            logger.info(f"Exported {len(self.detected_patterns)} patterns to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Export failed: {str(e)}")
            return False
    
    def plot_patterns(self, price_data, output_path=None):
        """Plot detected patterns"""
        if not self.detected_patterns:
            logger.warning("No patterns to plot")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Plot price data
        if isinstance(price_data, pd.DataFrame):
            if 'close' in price_data.columns:
                plt.plot(price_data['close'].values, color='blue', alpha=0.5)
            else:
                plt.plot(price_data.iloc[:, 0].values, color='blue', alpha=0.5)
        else:
            plt.plot(np.array(price_data), color='blue', alpha=0.5)
        
        # Plot patterns
        colors = ['red', 'green', 'purple', 'orange', 'cyan']
        
        for i, pattern in enumerate(self.detected_patterns):
            color = colors[i % len(colors)]
            points = pattern['points']
            
            # Plot points
            x_values = [p['index'] for p in points]
            y_values = [p['price'] for p in points]
            
            plt.plot(x_values, y_values, 'o-', color=color, linewidth=2, 
                    label=f"{pattern['type']} ({pattern['confidence']:.2f})")
            
            # Annotate Fibonacci ratios
            for j in range(1, len(points)-1):
                plt.annotate(
                    f"{points[j]['ratio']:.3f}",
                    (x_values[j], y_values[j]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    color=color
                )
        
        plt.title("Detected Harmonic Manipulation Patterns")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {output_path}")
        else:
            plt.show()
    
    def _find_swing_points(self, prices, window=5):
        """
        Find swing points (local minima and maxima)
        
        Args:
            prices: Array of price values
            window: Window size for finding local extrema
            
        Returns:
            List of swing point indices
        """
        swing_points = []
        
        for i in range(window, len(prices) - window):
            # Local maximum
            if all(prices[i] > prices[i-j] for j in range(1, window+1)) and \
               all(prices[i] > prices[i+j] for j in range(1, window+1)):
                swing_points.append(i)
            
            # Local minimum
            elif all(prices[i] < prices[i-j] for j in range(1, window+1)) and \
                 all(prices[i] < prices[i+j] for j in range(1, window+1)):
                swing_points.append(i)
        
        logger.info(f"Found {len(swing_points)} swing points")
        return swing_points
    
    def _analyze_harmonics(self, prices, swing_points, timestamps, symbol):
        """
        Analyze swing points for harmonic patterns
        
        Args:
            prices: Array of price values
            swing_points: Indices of swing points
            timestamps: Optional array of timestamps
            symbol: Optional symbol name
            
        Returns:
            List of detected patterns
        """
        patterns = []
        
        # Need at least 5 points for a valid pattern
        if len(swing_points) < MIN_SWING_POINTS:
            return patterns
        
        # Check all possible 5-point combinations
        for i in range(len(swing_points) - 4):
            points = [swing_points[i+j] for j in range(5)]
            
            # Calculate price values at these points
            price_values = [prices[idx] for idx in points]
            
            # Calculate retracement ratios
            ratios = self._calculate_ratios(price_values)
            
            # Check if ratios match known harmonic patterns
            pattern_match = self._match_harmonic_pattern(ratios)
            
            if pattern_match:
                # Pattern detected
                pattern_type, confidence = pattern_match
                
                # Format pattern data
                pattern_data = {
                    'type': pattern_type,
                    'confidence': confidence,
                    'symbol': symbol,
                    'points': []
                }
                
                # Add point data
                for j, idx in enumerate(points):
                    point_data = {
                        'index': int(idx),
                        'price': float(prices[idx]),
                    }
                    
                    # Add ratio for internal points
                    if j > 0 and j < 4:
                        point_data['ratio'] = ratios[j-1]
                    
                    # Add timestamp if available
                    if timestamps is not None:
                        point_data['timestamp'] = timestamps[idx]
                        
                    pattern_data['points'].append(point_data)
                
                patterns.append(pattern_data)
                logger.info(f"Detected {pattern_type} pattern with confidence {confidence:.4f}")
        
        return patterns
    
    def _calculate_ratios(self, prices):
        """
        Calculate retracement ratios between 5 price points
        
        Args:
            prices: List of 5 price values
            
        Returns:
            List of 3 ratios (B, C, D points)
        """
        # Extract points
        p0, p1, p2, p3, p4 = prices
        
        # Calculate ratios
        # XA = p1 - p0
        # AB = p2 - p1
        # BC = p3 - p2
        # CD = p4 - p3
        
        # B point (retracement of XA)
        if p1 != p0:
            ratio_b = abs((p2 - p1) / (p1 - p0))
        else:
            ratio_b = 0
            
        # C point (retracement of AB)
        if p2 != p1:
            ratio_c = abs((p3 - p2) / (p2 - p1))
        else:
            ratio_c = 0
            
        # D point (retracement of XA)
        if p1 != p0:
            ratio_d = abs((p4 - p3) / (p1 - p0))
        else:
            ratio_d = 0
            
        return [ratio_b, ratio_c, ratio_d]
    
    def _match_harmonic_pattern(self, ratios):
        """
        Match calculated ratios to known harmonic patterns
        
        Args:
            ratios: List of 3 ratios
            
        Returns:
            Tuple of (pattern_type, confidence) or None if no match
        """
        patterns = {
            "Gartley": [
                [0.618, 0.382, 0.786],
                self.tolerance
            ],
            "Butterfly": [
                [0.786, 0.382, 1.618],
                self.tolerance
            ],
            "Bat": [
                [0.382, 0.382, 1.618],
                self.tolerance
            ],
            "Crab": [
                [0.382, 0.618, 2.618],
                self.tolerance
            ],
            "GoldenGartley": [
                [0.618, 0.618, 0.618],
                self.tolerance
            ]
        }
        
        best_match = None
        best_confidence = 0
        
        for pattern_name, (target_ratios, max_deviation) in patterns.items():
            # Calculate how closely the ratios match the pattern
            deviations = [abs(ratios[i] - target_ratios[i]) for i in range(3)]
            max_dev = max(deviations)
            
            # Convert deviation to confidence score
            if max_dev <= max_deviation:
                confidence = 1.0 - (max_dev / max_deviation)
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = pattern_name
        
        if best_match and best_confidence >= 0.7:
            return (best_match, best_confidence)
        
        return None


def main():
    parser = argparse.ArgumentParser(description="Detect harmonic manipulation patterns in market data")
    parser.add_argument("input_file", help="Path to CSV file with price data")
    parser.add_argument("-o", "--output", help="Path to save results JSON", default="harmonic_patterns.json")
    parser.add_argument("-p", "--plot", help="Path to save plot image")
    parser.add_argument("-s", "--sensitivity", type=float, default=0.9, help="Detection sensitivity (0.5-0.99)")
    
    args = parser.parse_args()
    
    try:
        # Load data
        data = pd.read_csv(args.input_file)
        
        # Extract symbol from filename
        symbol = args.input_file.split('/')[-1].split('.')[0]
        
        # Create detector
        detector = HarmonicDetector(sensitivity=args.sensitivity)
        
        # Analyze data
        patterns = detector.analyze(data, symbol)
        
        # Export results
        if patterns:
            detector.export_results(args.output)
            
            # Generate plot if requested
            if args.plot:
                detector.plot_patterns(data, args.plot)
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
