#!/usr/bin/env python3
"""
Temporal Impossibility Scanner
Gold Star Chip Project

Identifies price movements that precede their supposed catalysts,
revealing the non-causal nature of modern markets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
import json
import logging
from typing import List, Dict, Tuple, Optional, Union
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TemporalScanner")

# Constants
IMPOSSIBILITY_THRESHOLD_MS = 7  # Price cannot react faster than 7ms to news
PROBABILITY_THRESHOLD = 0.9973  # 3-sigma statistical significance
MIN_PRICE_MOVE_PCT = 0.05  # Minimum price movement to be considered significant


class TemporalImpossibilityScanner:
    """Scans for temporal impossibilities in market data"""
    
    def __init__(self, sensitivity: float = 0.9):
        """
        Initialize the scanner
        
        Args:
            sensitivity: Detection sensitivity (0-1)
        """
        self.sensitivity = sensitivity
        self.threshold_ms = IMPOSSIBILITY_THRESHOLD_MS / sensitivity
        self.anomalies = []
        self.catalog = {}
        
        logger.info(f"Initialized Temporal Impossibility Scanner with {self.threshold_ms}ms threshold")
    
    def scan(self, price_data: pd.DataFrame, news_data: pd.DataFrame, 
             symbol: Optional[str] = None) -> List[Dict]:
        """
        Scan for temporal impossibilities between price and news data
        
        Args:
            price_data: DataFrame with price timestamps and values
            news_data: DataFrame with news timestamps and headlines
            symbol: Optional asset symbol
            
        Returns:
            List of detected temporal anomalies
        """
        if symbol:
            logger.info(f"Scanning {symbol} for temporal impossibilities")
        else:
            logger.info("Scanning for temporal impossibilities")
        
        # Validate data
        if not self._validate_data(price_data, news_data):
            return []
            
        # Reset anomalies
        self.anomalies = []
        
        # Extract timestamps and ensure they're datetime objects
        price_data = self._ensure_datetime_index(price_data)
        news_data = self._ensure_datetime_index(news_data)
        
        # Scan for price movements that precede news
        self._scan_price_preceding_news(price_data, news_data, symbol)
        
        # Calculate statistical significance
        self._calculate_statistical_significance()
        
        # Log results
        anomaly_count = len(self.anomalies)
        logger.info(f"Detected {anomaly_count} temporal impossibilities")
        
        return self.anomalies
    
    def catalog_anomalies(self) -> Dict:
        """
        Create catalog of anomalies by type, symbol, etc.
        
        Returns:
            Dictionary with categorized anomalies
        """
        catalog = {
            "bySymbol": {},
            "byType": {},
            "byTimeframe": {
                "microsecond": [],
                "millisecond": [],
                "second": [],
                "minute": []
            },
            "bySignificance": {
                "high": [],
                "medium": [],
                "low": []
            }
        }
        
        for anomaly in self.anomalies:
            # By symbol
            symbol = anomaly.get("symbol", "unknown")
            if symbol not in catalog["bySymbol"]:
                catalog["bySymbol"][symbol] = []
            catalog["bySymbol"][symbol].append(anomaly)
            
            # By type
            anomaly_type = anomaly.get("type", "unknown")
            if anomaly_type not in catalog["byType"]:
                catalog["byType"][anomaly_type] = []
            catalog["byType"][anomaly_type].append(anomaly)
            
            # By timeframe
            precognition_ms = anomaly.get("precognitionMs", 0)
            if precognition_ms < 1:
                timeframe = "microsecond"
            elif precognition_ms < 1000:
                timeframe = "millisecond"
            elif precognition_ms < 60000:
                timeframe = "second"
            else:
                timeframe = "minute"
            catalog["byTimeframe"][timeframe].append(anomaly)
            
            # By significance
            significance = anomaly.get("significance", 0)
            if significance > 0.95:
                catalog["bySignificance"]["high"].append(anomaly)
            elif significance > 0.8:
                catalog["bySignificance"]["medium"].append(anomaly)
            else:
                catalog["bySignificance"]["low"].append(anomaly)
        
        self.catalog = catalog
        return catalog
    
    def export_results(self, output_path: str) -> bool:
        """
        Export anomalies to JSON file
        
        Args:
            output_path: Path to save results
            
        Returns:
            Success status
        """
        if not self.anomalies:
            logger.warning("No anomalies to export")
            return False
            
        try:
            # Ensure catalog is created
            if not self.catalog:
                self.catalog_anomalies()
                
            # Build results object
            results = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "sensitivity": self.sensitivity,
                    "threshold_ms": self.threshold_ms,
                    "anomaly_count": len(self.anomalies)
                },
                "anomalies": self.anomalies,
                "catalog": self.catalog
            }
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
                
            logger.info(f"Exported {len(self.anomalies)} anomalies to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Export failed: {str(e)}")
            return False
    
    def plot_anomalies(self, price_data: pd.DataFrame, output_path: Optional[str] = None):
        """
        Plot detected anomalies
        
        Args:
            price_data: Price data DataFrame
            output_path: Optional path to save plot
        """
        if not self.anomalies:
            logger.warning("No anomalies to plot")
            return
            
        # Ensure datetime index
        price_data = self._ensure_datetime_index(price_data)
        
        # Filter to significant anomalies
        significant_anomalies = [a for a in self.anomalies 
                                 if a.get("significance", 0) > 0.8]
        
        if not significant_anomalies:
            logger.warning("No significant anomalies to plot")
            return
            
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot price data
        if 'close' in price_data.columns:
            plt.plot(price_data.index, price_data['close'], color='blue', alpha=0.5)
        else:
            plt.plot(price_data.index, price_data.iloc[:, 0], color='blue', alpha=0.5)
            
        # Plot anomalies
        for anomaly in significant_anomalies:
            price_time = anomaly.get("priceTimestamp")
            news_time = anomaly.get("newsTimestamp")
            
            if not price_time or not news_time:
                continue
                
            # Convert to datetime if needed
            if isinstance(price_time, str):
                price_time = datetime.fromisoformat(price_time.replace('Z', '+00:00'))
            if isinstance(news_time, str):
                news_time = datetime.fromisoformat(news_time.replace('Z', '+00:00'))
                
            # Find price at this point
            try:
                if 'close' in price_data.columns:
                    price = price_data.loc[price_data.index == price_time, 'close'].values[0]
                else:
                    price = price_data.loc[price_data.index == price_time].iloc[0, 0]
            except (IndexError, KeyError):
                # Find closest timestamp
                closest_idx = price_data.index.get_indexer([price_time], method='nearest')[0]
                if closest_idx >= 0:
                    if 'close' in price_data.columns:
                        price = price_data.iloc[closest_idx]['close']
                    else:
                        price = price_data.iloc[closest_idx, 0]
                else:
                    continue
                    
            # Plot price point
            plt.scatter(price_time, price, color='red', s=50, zorder=5)
            
            # Draw arrow from news time to price time
            arrow_props = dict(arrowstyle='->', linewidth=2, color='red', alpha=0.7)
            plt.annotate('', xy=(price_time, price), xytext=(news_time, price),
                        arrowprops=arrow_props)
            
            # Add label
            precognition_ms = anomaly.get("precognitionMs", 0)
            plt.annotate(f"{precognition_ms:.2f}ms", 
                        xy=(price_time, price),
                        xytext=(5, 10),
                        textcoords="offset points",
                        color='red',
                        fontweight='bold')
        
        plt.title("Detected Temporal Impossibilities")
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {output_path}")
        else:
            plt.show()
    
    def _validate_data(self, price_data: pd.DataFrame, news_data: pd.DataFrame) -> bool:
        """
        Validate input data format
        
        Args:
            price_data: Price data DataFrame
            news_data: News data DataFrame
            
        Returns:
            True if data is valid
        """
        # Check if DataFrames are empty
        if price_data.empty or news_data.empty:
            logger.error("Empty DataFrame provided")
            return False
            
        # Check if timestamps are present
        if not isinstance(price_data.index, pd.DatetimeIndex) and not any(
            col.lower() in ['timestamp', 'time', 'date'] for col in price_data.columns
        ):
            logger.error("Price data missing timestamp column or index")
            return False
            
        if not isinstance(news_data.index, pd.DatetimeIndex) and not any(
            col.lower() in ['timestamp', 'time', 'date'] for col in news_data.columns
        ):
            logger.error("News data missing timestamp column or index")
            return False
            
        # Check if price data contains price values
        if price_data.shape[1] == 0:
            logger.error("Price data contains no columns")
            return False
            
        # Check if news data contains headlines
        if not any(col.lower() in ['headline', 'title', 'news'] for col in news_data.columns):
            logger.warning("News data might be missing headline/title column")
            
        return True
    
    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure DataFrame has a datetime index
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with datetime index
        """
        # If already has datetime index, return as is
        if isinstance(df.index, pd.DatetimeIndex):
            return df
            
        # Try to find timestamp column
        timestamp_cols = [col for col in df.columns 
                         if col.lower() in ['timestamp', 'time', 'date']]
        
        if timestamp_cols:
            # Set first matching column as index
            df = df.copy()
            df['_timestamp'] = pd.to_datetime(df[timestamp_cols[0]])
            return df.set_index('_timestamp')
        
        # If no timestamp column found, use first column
        logger.warning("No timestamp column found, using first column")
        df = df.copy()
        df['_timestamp'] = pd.to_datetime(df.iloc[:, 0])
        return df.set_index('_timestamp')
    
    def _scan_price_preceding_news(self, price_data: pd.DataFrame, 
                                  news_data: pd.DataFrame,
                                  symbol: Optional[str] = None):
        """
        Scan for price movements that precede news events
        
        Args:
            price_data: Price data with datetime index
            news_data: News data with datetime index
            symbol: Optional asset symbol
        """
        # Extract price series
        if 'close' in price_data.columns:
            prices = price_data['close']
        else:
            prices = price_data.iloc[:, 0]
            
        # Extract news headlines
        if 'headline' in news_data.columns:
            headlines = news_data['headline']
        elif 'title' in news_data.columns:
            headlines = news_data['title']
        elif 'news' in news_data.columns:
            headlines = news_data['news']
        else:
            headlines = news_data.iloc[:, 1] if news_data.shape[1] > 1 else None
            
        # For each news event, check if price moved before it
        for news_time, headline in zip(news_data.index, headlines if headlines is not None else [None] * len(news_data)):
            # Find significant price movements within a window before and after the news
            window_before = news_time - timedelta(seconds=1)
            window_after = news_time + timedelta(seconds=1)
            
            # Get price data in window
            window_mask = (prices.index >= window_before) & (prices.index <= window_after)
            window_prices = prices[window_mask]
            
            if len(window_prices) < 2:
                continue
                
            # Find significant price movements
            price_changes = window_prices.pct_change().abs()
            significant_moves = price_changes[price_changes > MIN_PRICE_MOVE_PCT]
            
            for move_time, move_pct in significant_moves.items():
                # Skip moves that happen after the news (these are normal)
                time_diff_ms = (news_time - move_time).total_seconds() * 1000
                
                # If price moved before news
                if 0 < time_diff_ms < self.threshold_ms * 10:  # Check larger window for cataloging
                    # Calculate significance based on how close to the threshold
                    impossibility_significance = 1.0
                    if time_diff_ms > self.threshold_ms:
                        # Still record but with lower significance if outside threshold
                        impossibility_significance = self.threshold_ms / time_diff_ms
                        
                    # Calculate price movement significance
                    movement_significance = min(move_pct / (MIN_PRICE_MOVE_PCT * 2), 1.0)
                    
                    # Overall significance
                    significance = impossibility_significance * movement_significance
                    
                    # Skip low significance anomalies
                    if significance < 0.5:
                        continue
                        
                    # Determine anomaly type
                    if time_diff_ms <= self.threshold_ms:
                        anomaly_type = "impossibility"
                    elif time_diff_ms <= self.threshold_ms * 3:
                        anomaly_type = "high_suspicion"
                    else:
                        anomaly_type = "suspicious"
                    
                    # Record anomaly
                    anomaly = {
                        "symbol": symbol,
                        "type": anomaly_type,
                        "newsTimestamp": news_time,
                        "priceTimestamp": move_time,
                        "precognitionMs": time_diff_ms,
                        "priceMovePct": float(move_pct),
                        "significance": float(significance),
                        "headline": headline if headline is not None else ""
                    }
                    
                    self.anomalies.append(anomaly)
    
    def _calculate_statistical_significance(self):
        """Calculate statistical significance of detected anomalies"""
        if not self.anomalies:
            return
            
        # Calculate Z-score for each anomaly based on precognition time
        precognition_times = [a.get("precognitionMs", 0) for a in self.anomalies]
        
        if not precognition_times:
            return
            
        mean_time = np.mean(precognition_times)
        std_time = np.std(precognition_times) if len(precognition_times) > 1 else 1.0
        
        if std_time == 0:
            std_time = 1.0  # Avoid division by zero
            
        for i, anomaly in enumerate(self.anomalies):
            precognition_ms = anomaly.get("precognitionMs", 0)
            
            # Calculate Z-score (distance from mean in standard deviations)
            z_score = abs(precognition_ms - mean_time) / std_time
            
            # Calculate p-value (probability of seeing this by chance)
            # Using simplified approximation
            p_value = 1 - (1 - 2 * (1 - 0.5 * (1 + np.erf(z_score / np.sqrt(2)))))
            
            # Add to anomaly
            self.anomalies[i]["zScore"] = float(z_score)
            self.anomalies[i]["pValue"] = float(min(p_value, 1.0))
            
            # Update significance based on statistical measures
            stat_significance = 1 - min(p_value, 1.0)
            self.anomalies[i]["significance"] = float(
                (self.anomalies[i].get("significance", 0) + stat_significance) / 2
            )


def main():
    """Main function when script is run directly"""
    parser = argparse.ArgumentParser(description="Scan for temporal impossibilities in market data")
    parser.add_argument("price_file", help="Path to CSV file with price data")
    parser.add_argument("news_file", help="Path to CSV file with news data")
    parser.add_argument("-o", "--output", help="Path to save results JSON", default="temporal_anomalies.json")
    parser.add_argument("-p", "--plot", help="Path to save plot image")
    parser.add_argument("-s", "--sensitivity", type=float, default=0.9, help="Detection sensitivity (0.5-0.99)")
    parser.add_argument("--symbol", help="Asset symbol")
    
    args = parser.parse_args()
    
    try:
        # Load data
        price_data = pd.read_csv(args.price_file)
        news_data = pd.read_csv(args.news_file)
        
        # Extract symbol from filename if not provided
        symbol = args.symbol
        if not symbol:
            symbol = os.path.basename(args.price_file).split('.')[0]
        
        # Create scanner
        scanner = TemporalImpossibilityScanner(sensitivity=args.sensitivity)
        
        # Scan for anomalies
        anomalies = scanner.scan(price_data, news_data, symbol)
        
        # Export results
        if anomalies:
            scanner.export_results(args.output)
            
            # Generate plot if requested
            if args.plot:
                scanner.plot_anomalies(price_data, args.plot)
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
