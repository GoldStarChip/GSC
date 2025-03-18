#!/usr/bin/env ruby
# encoding: utf-8

# Quantum Correlation Analyzer
# Gold Star Chip Project
#
# Identifies mathematically impossible synchronicities across diverse asset classes
# to reveal coordinated market manipulation occurring at quantum speeds.

require 'csv'
require 'json'
require 'matrix'
require 'date'

# Constants
CORRELATION_THRESHOLD = 0.973
IMPOSSIBLE_SYNC_WINDOW_MS = 7
GOLDEN_RATIO = 1.618033988749895
CONTROL_FREQUENCIES = [1.618, 2.718, 3.141, 4.669, 6.022, 6.626]

class QuantumCorrelationAnalyzer
  attr_reader :correlations, :synchronicities, :quantum_nodes

  def initialize(config = {})
    @config = {
      sensitivity: 0.9,
      min_correlation: CORRELATION_THRESHOLD,
      max_time_diff_ms: IMPOSSIBLE_SYNC_WINDOW_MS,
      debug: false
    }.merge(config)
    
    @correlations = {}
    @synchronicities = []
    @quantum_nodes = []
    @markets_analyzed = 0
  end
  
  def analyze(data_path)
    puts "Analyzing quantum correlations in #{data_path}"
    
    # Load data from CSV
    data = load_data(data_path)
    return false if data.nil? || data.empty?
    
    # Group data by timestamp
    time_series = group_by_timestamp(data)
    
    # Analyze for quantum correlations
    puts "Analyzing #{time_series.size} timestamped data points across #{data.first.size - 1} markets"
    @markets_analyzed = data.first.size - 1
    
    # Find impossible synchronicities
    @synchronicities = detect_synchronicities(time_series)
    puts "Detected #{@synchronicities.size} impossible synchronicities"
    
    # Calculate correlation matrix
    @correlations = calculate_correlations(data)
    puts "Analyzed #{@correlations.size} market pairs for quantum correlations"
    
    # Identify quantum nodes
    @quantum_nodes = identify_quantum_nodes(@synchronicities, @correlations)
    puts "Identified #{@quantum_nodes.size} quantum control nodes"
    
    return {
      correlations: @correlations,
      synchronicities: @synchronicities,
      quantum_nodes: @quantum_nodes
    }
  end
  
  def export_results(output_path)
    results = {
      meta: {
        timestamp: Time.now.iso8601,
        markets_analyzed: @markets_analyzed,
        correlation_threshold: @config[:min_correlation],
        time_window_ms: @config[:max_time_diff_ms]
      },
      quantum_correlations: @correlations,
      impossible_synchronicities: @synchronicities,
      quantum_nodes: @quantum_nodes
    }
    
    File.open(output_path, 'w') do |file|
      file.write(JSON.pretty_generate(results))
    end
    
    puts "Results exported to #{output_path}"
  end
  
  private
  
  def load_data(path)
    begin
      # Read CSV file
      data = CSV.read(path, headers: true)
      return data
    rescue => e
      puts "Error loading data: #{e.message}"
      return nil
    end
  end
  
  def group_by_timestamp(data)
    time_series = {}
    
    data.each do |row|
      timestamp = row['timestamp']
      next unless timestamp
      
      time_series[timestamp] ||= {}
      
      # Add each market's data to this timestamp
      row.each do |key, value|
        next if key == 'timestamp' || value.nil?
        time_series[timestamp][key] = value.to_f
      end
    end
    
    time_series
  end
  
  def detect_synchronicities(time_series)
    synchronicities = []
    timestamps = time_series.keys.sort
    
    # Look for coordinated moves across 3+ uncorrelated assets
    timestamps.each_with_index do |ts, i|
      # Skip if we're at the last timestamp
      next if i >= timestamps.size - 1
      
      current_data = time_series[ts]
      next_ts = timestamps[i+1]
      next_data = time_series[next_ts]
      
      # Calculate time difference in milliseconds
      begin
        current_time = DateTime.parse(ts).to_time
        next_time = DateTime.parse(next_ts).to_time
        time_diff_ms = (next_time - current_time) * 1000
      rescue
        # Skip if we can't parse timestamps
        next
      end
      
      # Look for synchronized movements within impossible timeframe
      if time_diff_ms <= @config[:max_time_diff_ms]
        # Find assets that moved in synchronized fashion
        synced_assets = []
        
        current_data.each do |asset, value|
          next unless next_data.key?(asset)
          
          # Calculate percent change
          pct_change = (next_data[asset] - value) / value
          
          # Assets with significant moves
          if pct_change.abs >= 0.001  # 0.1% move
            synced_assets << {
              asset: asset,
              pct_change: pct_change,
              start_value: value,
              end_value: next_data[asset]
            }
          end
        end
        
        # Check if we have 3+ assets moving together
        if synced_assets.size >= 3
          # Check if these assets are from different sectors/correlations
          if assets_from_different_sectors?(synced_assets.map { |a| a[:asset] })
            synchronicities << {
              start_time: ts,
              end_time: next_ts,
              time_diff_ms: time_diff_ms,
              synced_assets: synced_assets,
              signature: calculate_sync_signature(synced_assets),
              confidence: calculate_sync_confidence(synced_assets, time_diff_ms)
            }
          end
        end
      end
    end
    
    synchronicities
  end
  
  def assets_from_different_sectors?(assets)
    # This is a simplified check - in reality you'd check
    # if these assets have historically been uncorrelated
    # For this example, we'll just check if the symbols look different
    
    # Extract first 2 characters of each asset as "sector"
    sectors = assets.map { |a| a[0..1] }.uniq
    
    # If we have at least 2 different "sectors", return true
    sectors.size >= 2
  end
  
  def calculate_sync_signature(synced_assets)
    # Create a signature based on relative movement directions
    directions = synced_assets.map { |a| a[:pct_change] > 0 ? 1 : -1 }
    directions.join("")
  end
  
  def calculate_sync_confidence(synced_assets, time_diff_ms)
    # More assets + shorter time = higher confidence
    asset_factor = [synced_assets.size / 3.0, 1.0].min
    time_factor = [(IMPOSSIBLE_SYNC_WINDOW_MS - time_diff_ms) / IMPOSSIBLE_SYNC_WINDOW_MS, 0.1].max
    
    # Higher confidence for larger percentage moves
    avg_move = synced_assets.map { |a| a[:pct_change].abs }.sum / synced_assets.size
    move_factor = [avg_move * 100, 1.0].min
    
    confidence = (asset_factor * 0.4) + (time_factor * 0.4) + (move_factor * 0.2)
    [confidence, 1.0].min * @config[:sensitivity]
  end
  
  def calculate_correlations(data)
    correlations = {}
    
    # Extract asset columns (excluding timestamp)
    assets = data.headers.reject { |h| h == 'timestamp' }
    
    # Calculate correlations between each asset pair
    assets.combination(2).each do |a1, a2|
      # Extract values for both assets, removing nil values
      values1 = []
      values2 = []
      
      data.each do |row|
        v1 = row[a1]
        v2 = row[a2]
        
        next if v1.nil? || v2.nil?
        
        values1 << v1.to_f
        values2 << v2.to_f
      end
      
      # Skip if insufficient data
      next if values1.size < 10
      
      # Calculate correlation coefficient
      correlation = pearson_correlation(values1, values2)
      
      # Store if correlation exceeds threshold
      if correlation.abs >= @config[:min_correlation]
        correlations["#{a1}:#{a2}"] = {
          asset1: a1,
          asset2: a2,
          correlation: correlation,
          data_points: values1.size,
          quantum_signature: correlation >= CORRELATION_THRESHOLD
        }
      end
    end
    
    correlations
  end
  
  def pearson_correlation(x, y)
    n = [x.size, y.size].min
    return 0 if n < 2
    
    # Convert to statistical vectors
    x = x[0...n]
    y = y[0...n]
    
    # Calculate means
    mean_x = x.sum / n.to_f
    mean_y = y.sum / n.to_f
    
    # Calculate correlation
    numerator = 0
    sum_sq_x = 0
    sum_sq_y = 0
    
    n.times do |i|
      numerator += (x[i] - mean_x) * (y[i] - mean_y)
      sum_sq_x += (x[i] - mean_x) ** 2
      sum_sq_y += (y[i] - mean_y) ** 2
    end
    
    denominator = Math.sqrt(sum_sq_x * sum_sq_y)
    
    denominator.zero? ? 0 : numerator / denominator
  end
  
  def identify_quantum_nodes(synchronicities, correlations)
    quantum_nodes = []
    
    # Group synchronicities by signature
    signatures = {}
    synchronicities.each do |sync|
      signatures[sync[:signature]] ||= []
      signatures[sync[:signature]] << sync
    end
    
    # Identify recurring patterns as quantum nodes
    signatures.each do |signature, syncs|
      next if syncs.size < 3  # Need at least 3 occurrences
      
      # Calculate average confidence
      avg_confidence = syncs.map { |s| s[:confidence] }.sum / syncs.size
      
      # Extract assets involved
      assets = syncs.flat_map { |s| s[:synced_assets].map { |a| a[:asset] } }.uniq
      
      # Create quantum node
      quantum_nodes << {
        signature: signature,
        occurrences: syncs.size,
        confidence: avg_confidence,
        assets: assets,
        timestamps: syncs.map { |s| s[:start_time] },
        correlation_strength: calculate_node_correlation_strength(assets, correlations)
      }
    end
    
    quantum_nodes
  end
  
  def calculate_node_correlation_strength(assets, correlations)
    # Calculate how strongly interconnected these assets are
    pairs = 0
    correlated_pairs = 0
    
    assets.combination(2).each do |a1, a2|
      key = "#{a1}:#{a2}"
      alt_key = "#{a2}:#{a1}"
      
      pairs += 1
      if correlations.key?(key) || correlations.key?(alt_key)
        correlated_pairs += 1
      end
    end
    
    pairs.zero? ? 0 : correlated_pairs.to_f / pairs
  end
end

# Example usage
if __FILE__ == $0
  analyzer = QuantumCorrelationAnalyzer.new(sensitivity: 0.85)
  results = analyzer.analyze("market_data.csv")
  analyzer.export_results("quantum_correlations.json")
end
