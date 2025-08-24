// SEP Trading System - Identity Inspector
// Detailed analysis of individual quantum identities and their evolution

import React, { useState, useEffect, useMemo } from 'react';
import { useManifold } from '../context/ManifoldContext';
import { 
  Search, 
  Eye, 
  Clock, 
  Zap, 
  Target, 
  GitBranch, 
  Database,
  Activity,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  XCircle,
  ArrowRight,
  Copy,
  ExternalLink
} from 'lucide-react';

const IdentityInspector = () => {
  const {
    processedIdentities,
    manifoldBands,
    selectedIdentity,
    setSelectedIdentity,
    getIdentityByKey,
    getIdentityHistory,
    canDeriveBackwards,
    getDeterministicPath
  } = useManifold();


  const [searchTerm, setSearchTerm] = useState('');
  const [selectedKey, setSelectedKey] = useState('');
  const [viewMode, setViewMode] = useState('overview'); // overview, timeline, backwards, comparison
  const [inspectorData, setInspectorData] = useState(null);

  // Available identities for inspection
  const availableIdentities = useMemo(() => {
    const identities = [];
    
    processedIdentities.forEach((timeline, instrument) => {
      timeline.forEach(point => {
        identities.push({
          instrument,
          valkey_key: point.valkey_key,
          timestamp: point.timestamp,
          entropy: point.entropy,
          stability: point.stability,
          coherence: point.coherence,
          state: point.state,
          price: point.price,
          age_seconds: (Date.now() - point.timestamp) / 1000,
          band: point.entropy > 0.7 ? 'hot' : point.entropy > 0.3 ? 'warm' : 'cold'
        });
      });
    });

    // Sort by recency
    return identities.sort((a, b) => b.timestamp - a.timestamp);
  }, [processedIdentities]);

  // Filtered identities based on search
  const filteredIdentities = useMemo(() => {
    if (!searchTerm) return availableIdentities;
    
    const term = searchTerm.toLowerCase();
    return availableIdentities.filter(identity =>
      identity.instrument.toLowerCase().includes(term) ||
      identity.valkey_key.toLowerCase().includes(term) ||
      identity.state.toLowerCase().includes(term) ||
      identity.band.toLowerCase().includes(term)
    );
  }, [availableIdentities, searchTerm]);

  // Load detailed identity data
  useEffect(() => {
    if (!selectedKey) {
      setInspectorData(null);
      return;
    }

    const identity = getIdentityByKey(selectedKey);
    if (!identity) return;

    const timeline = getIdentityHistory(identity.instrument);
    const currentIndex = timeline.findIndex(p => p.valkey_key === selectedKey);
    
    if (currentIndex === -1) return;

    const currentPoint = timeline[currentIndex];
    const previousPoints = timeline.slice(0, currentIndex);
    const futurePoints = timeline.slice(currentIndex + 1);

    // Calculate evolution metrics
    const evolutionMetrics = {
      entropy_trajectory: previousPoints.map(p => p.entropy),
      stability_trajectory: previousPoints.map(p => p.stability),
      coherence_trajectory: previousPoints.map(p => p.coherence),
      state_changes: [],
      convergence_events: [],
      divergence_events: []
    };

    // Detect state changes
    for (let i = 1; i < timeline.length; i++) {
      if (timeline[i].state !== timeline[i-1].state) {
        evolutionMetrics.state_changes.push({
          from: timeline[i-1].state,
          to: timeline[i].state,
          timestamp: timeline[i].timestamp,
          entropy_at_change: timeline[i].entropy
        });
      }
    }

    // Detect convergence/divergence events
    previousPoints.forEach((point, idx) => {
      if (idx === 0) return;
      
      const prev = previousPoints[idx - 1];
      const stabilityDelta = point.stability - prev.stability;
      const coherenceDelta = point.coherence - prev.coherence;
      
      if (stabilityDelta > 0.1 && coherenceDelta > 0.1) {
        evolutionMetrics.convergence_events.push({
          timestamp: point.timestamp,
          stability_gain: stabilityDelta,
          coherence_gain: coherenceDelta,
          strength: stabilityDelta + coherenceDelta
        });
      } else if (stabilityDelta < -0.1 || coherenceDelta < -0.1) {
        evolutionMetrics.divergence_events.push({
          timestamp: point.timestamp,
          stability_loss: Math.abs(stabilityDelta),
          coherence_loss: Math.abs(coherenceDelta),
          severity: Math.abs(stabilityDelta) + Math.abs(coherenceDelta)
        });
      }
    });

    // Backwards computation analysis
    const backwardsAnalysis = {
      can_derive: canDeriveBackwards(selectedKey),
      deterministic_path: getDeterministicPath(identity.instrument),
      causality_strength: currentPoint.stability * currentPoint.coherence,
      temporal_lock: currentPoint.entropy < 0.15 && currentPoint.stability > 0.8,
      prediction_confidence: Math.min(currentPoint.stability + currentPoint.coherence, 1.0)
    };

    // Related identities (same instrument, similar metrics)
    const relatedIdentities = availableIdentities.filter(id => 
      id.instrument === identity.instrument &&
      id.valkey_key !== selectedKey &&
      Math.abs(id.entropy - currentPoint.entropy) < 0.1 &&
      Math.abs(id.stability - currentPoint.stability) < 0.1
    ).slice(0, 5);

    setInspectorData({
      identity,
      currentPoint,
      timeline,
      currentIndex,
      previousPoints,
      futurePoints,
      evolutionMetrics,
      backwardsAnalysis,
      relatedIdentities
    });
  }, [selectedKey, processedIdentities]);

  // Copy key to clipboard
  const copyKey = (key) => {
    navigator.clipboard.writeText(key);
  };

  // Format age display
  const formatAge = (ageSeconds) => {
    if (ageSeconds < 60) return `${Math.floor(ageSeconds)}s`;
    if (ageSeconds < 3600) return `${Math.floor(ageSeconds / 60)}m`;
    return `${Math.floor(ageSeconds / 3600)}h`;
  };

  // State color mapping
  const getStateColor = (state) => {
    switch (state) {
      case 'converged': return 'text-green-400';
      case 'divergent': return 'text-red-400';
      case 'flux': return 'text-yellow-400';
      case 'stable': return 'text-blue-400';
      default: return 'text-gray-400';
    }
  };

  // Band color mapping
  const getBandColor = (band) => {
    switch (band) {
      case 'hot': return 'bg-red-900 text-red-300';
      case 'warm': return 'bg-yellow-900 text-yellow-300';
      case 'cold': return 'bg-green-900 text-green-300';
      default: return 'bg-gray-900 text-gray-300';
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                Identity Inspector
              </h1>
              <p className="text-gray-400 mt-1">Deep analysis of quantum identity evolution and temporal manifold mapping</p>
            </div>

            <div className="flex items-center gap-4">
              <select
                value={viewMode}
                onChange={(e) => setViewMode(e.target.value)}
                className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm"
              >
                <option value="overview">Overview</option>
                <option value="timeline">Timeline</option>
                <option value="backwards">Backwards Analysis</option>
                <option value="comparison">Comparison</option>
              </select>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-12 gap-6">
          {/* Left Panel - Identity List */}
          <div className="col-span-4">
            <div className="bg-gray-900 rounded-lg p-4">
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Database className="w-5 h-5 text-cyan-400" />
                Available Identities ({filteredIdentities.length})
              </h3>

              {/* Search */}
              <div className="mb-4">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search by instrument, key, state, or band..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="w-full pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500"
                  />
                </div>
              </div>

              {/* Identity List */}
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {filteredIdentities.slice(0, 50).map((identity, idx) => (
                  <div
                    key={identity.valkey_key}
                    onClick={() => setSelectedKey(identity.valkey_key)}
                    className={`p-3 rounded-lg cursor-pointer transition-colors ${
                      selectedKey === identity.valkey_key
                        ? 'bg-cyan-900 border border-cyan-700'
                        : 'bg-gray-800 hover:bg-gray-700'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium text-sm">{identity.instrument}</span>
                      <span className={`px-2 py-1 text-xs rounded ${getBandColor(identity.band)}`}>
                        {identity.band}
                      </span>
                    </div>
                    
                    <div className="text-xs space-y-1">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Key:</span>
                        <span className="font-mono">{identity.valkey_key.slice(-8)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">State:</span>
                        <span className={getStateColor(identity.state)}>{identity.state}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Age:</span>
                        <span>{formatAge(identity.age_seconds)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Entropy:</span>
                        <span>{identity.entropy.toFixed(3)}</span>
                      </div>
                    </div>
                  </div>
                ))}

                {filteredIdentities.length === 0 && (
                  <div className="text-center text-gray-500 py-8">
                    <Eye className="w-8 h-8 mx-auto mb-2 opacity-50" />
                    <p>No identities found</p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Right Panel - Inspector Details */}
          <div className="col-span-8">
            {!inspectorData ? (
              <div className="bg-gray-900 rounded-lg p-8 text-center">
                <Eye className="w-16 h-16 mx-auto mb-4 text-gray-600" />
                <h3 className="text-xl font-semibold mb-2">Select Identity for Inspection</h3>
                <p className="text-gray-400">Choose an identity from the left panel to view detailed analysis</p>
              </div>
            ) : (
              <div className="space-y-6">
                {/* Overview Mode */}
                {viewMode === 'overview' && (
                  <>
                    {/* Identity Header */}
                    <div className="bg-gray-900 rounded-lg p-6">
                      <div className="flex items-center justify-between mb-4">
                        <div>
                          <h3 className="text-2xl font-bold flex items-center gap-3">
                            <Target className="w-6 h-6 text-cyan-400" />
                            {inspectorData.identity.instrument}
                          </h3>
                          <p className="text-gray-400 mt-1">Quantum Identity Analysis</p>
                        </div>
                        
                        <button
                          onClick={() => copyKey(selectedKey)}
                          className="flex items-center gap-2 px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors text-sm"
                        >
                          <Copy className="w-4 h-4" />
                          Copy Key
                        </button>
                      </div>

                      {/* Key Info */}
                      <div className="bg-gray-800 rounded-lg p-4">
                        <div className="grid grid-cols-3 gap-4 text-sm">
                          <div>
                            <span className="text-gray-400">Valkey Key:</span>
                            <div className="font-mono text-xs bg-gray-900 p-2 rounded mt-1">
                              {selectedKey}
                            </div>
                          </div>
                          <div>
                            <span className="text-gray-400">Timestamp:</span>
                            <div className="font-medium mt-1">
                              {new Date(inspectorData.currentPoint.timestamp).toLocaleString()}
                            </div>
                          </div>
                          <div>
                            <span className="text-gray-400">Age:</span>
                            <div className="font-medium mt-1">
                              {formatAge((Date.now() - inspectorData.currentPoint.timestamp) / 1000)}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Quantum Metrics */}
                    <div className="grid grid-cols-3 gap-4">
                      <div className="bg-gray-900 rounded-lg p-4">
                        <div className="flex items-center gap-2 mb-3">
                          <Zap className="w-5 h-5 text-red-400" />
                          <h4 className="font-semibold">Entropy</h4>
                        </div>
                        <div className="text-2xl font-bold text-red-400 mb-2">
                          {inspectorData.currentPoint.entropy.toFixed(4)}
                        </div>
                        <div className="text-xs text-gray-400">
                          Band: <span className={`px-2 py-1 rounded ${getBandColor(inspectorData.identity.band)}`}>
                            {inspectorData.identity.band}
                          </span>
                        </div>
                      </div>

                      <div className="bg-gray-900 rounded-lg p-4">
                        <div className="flex items-center gap-2 mb-3">
                          <Activity className="w-5 h-5 text-green-400" />
                          <h4 className="font-semibold">Stability</h4>
                        </div>
                        <div className="text-2xl font-bold text-green-400 mb-2">
                          {inspectorData.currentPoint.stability.toFixed(4)}
                        </div>
                        <div className="text-xs text-gray-400">
                          {inspectorData.currentPoint.stability > 0.8 ? 'High' : 
                           inspectorData.currentPoint.stability > 0.5 ? 'Medium' : 'Low'} stability
                        </div>
                      </div>

                      <div className="bg-gray-900 rounded-lg p-4">
                        <div className="flex items-center gap-2 mb-3">
                          <Target className="w-5 h-5 text-blue-400" />
                          <h4 className="font-semibold">Coherence</h4>
                        </div>
                        <div className="text-2xl font-bold text-blue-400 mb-2">
                          {inspectorData.currentPoint.coherence.toFixed(4)}
                        </div>
                        <div className="text-xs text-gray-400">
                          Phase alignment: {(inspectorData.currentPoint.coherence * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>

                    {/* State and Evolution */}
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-gray-900 rounded-lg p-4">
                        <h4 className="font-semibold mb-3 flex items-center gap-2">
                          <CheckCircle className="w-5 h-5 text-green-400" />
                          Current State
                        </h4>
                        <div className="space-y-2">
                          <div className="flex justify-between items-center">
                            <span className="text-gray-400">State:</span>
                            <span className={`font-medium ${getStateColor(inspectorData.currentPoint.state)}`}>
                              {inspectorData.currentPoint.state}
                            </span>
                          </div>
                          {inspectorData.currentPoint.price && (
                            <div className="flex justify-between items-center">
                              <span className="text-gray-400">Price:</span>
                              <span className="font-medium">{inspectorData.currentPoint.price.toFixed(5)}</span>
                            </div>
                          )}
                          <div className="flex justify-between items-center">
                            <span className="text-gray-400">Timeline Position:</span>
                            <span className="font-medium">
                              {inspectorData.currentIndex + 1} / {inspectorData.timeline.length}
                            </span>
                          </div>
                        </div>
                      </div>

                      <div className="bg-gray-900 rounded-lg p-4">
                        <h4 className="font-semibold mb-3 flex items-center gap-2">
                          <TrendingUp className="w-5 h-5 text-yellow-400" />
                          Evolution Metrics
                        </h4>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between items-center">
                            <span className="text-gray-400">State Changes:</span>
                            <span className="font-medium">{inspectorData.evolutionMetrics.state_changes.length}</span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-gray-400">Convergence Events:</span>
                            <span className="font-medium text-green-400">
                              {inspectorData.evolutionMetrics.convergence_events.length}
                            </span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-gray-400">Divergence Events:</span>
                            <span className="font-medium text-red-400">
                              {inspectorData.evolutionMetrics.divergence_events.length}
                            </span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-gray-400">Evolution Span:</span>
                            <span className="font-medium">
                              {inspectorData.previousPoints.length > 0 ? 
                                formatAge((inspectorData.currentPoint.timestamp - inspectorData.previousPoints[0].timestamp) / 1000) :
                                'N/A'
                              }
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Backwards Computation Analysis */}
                    <div className="bg-gray-900 rounded-lg p-4">
                      <h4 className="font-semibold mb-3 flex items-center gap-2">
                        <GitBranch className="w-5 h-5 text-purple-400" />
                        Backwards Computation Analysis
                      </h4>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-3">
                          <div className="flex items-center justify-between">
                            <span className="text-gray-400">Can Derive Backwards:</span>
                            <span className={`flex items-center gap-1 ${inspectorData.backwardsAnalysis.can_derive ? 'text-green-400' : 'text-red-400'}`}>
                              {inspectorData.backwardsAnalysis.can_derive ? <CheckCircle className="w-4 h-4" /> : <XCircle className="w-4 h-4" />}
                              {inspectorData.backwardsAnalysis.can_derive ? 'Yes' : 'No'}
                            </span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="text-gray-400">Temporal Lock:</span>
                            <span className={`flex items-center gap-1 ${inspectorData.backwardsAnalysis.temporal_lock ? 'text-green-400' : 'text-gray-400'}`}>
                              {inspectorData.backwardsAnalysis.temporal_lock ? <CheckCircle className="w-4 h-4" /> : <XCircle className="w-4 h-4" />}
                              {inspectorData.backwardsAnalysis.temporal_lock ? 'Locked' : 'Free'}
                            </span>
                          </div>
                        </div>
                        <div className="space-y-3">
                          <div className="flex justify-between items-center">
                            <span className="text-gray-400">Causality Strength:</span>
                            <span className="font-medium text-purple-400">
                              {inspectorData.backwardsAnalysis.causality_strength.toFixed(4)}
                            </span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-gray-400">Prediction Confidence:</span>
                            <span className="font-medium text-blue-400">
                              {(inspectorData.backwardsAnalysis.prediction_confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </>
                )}

                {/* Timeline Mode */}
                {viewMode === 'timeline' && (
                  <div className="bg-gray-900 rounded-lg p-6">
                    <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                      <Clock className="w-5 h-5 text-cyan-400" />
                      Identity Timeline ({inspectorData.timeline.length} points)
                    </h3>
                    
                    <div className="space-y-3 max-h-96 overflow-y-auto">
                      {inspectorData.timeline.map((point, idx) => (
                        <div
                          key={idx}
                          className={`p-3 rounded-lg border-l-4 ${
                            point.valkey_key === selectedKey 
                              ? 'bg-cyan-900 border-cyan-500' 
                              : 'bg-gray-800 border-gray-600'
                          }`}
                        >
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-sm font-medium">
                              {new Date(point.timestamp).toLocaleString()}
                            </span>
                            <span className={`px-2 py-1 text-xs rounded ${getStateColor(point.state).replace('text-', 'bg-').replace('-400', '-900')} ${getStateColor(point.state).replace('-400', '-300')}`}>
                              {point.state}
                            </span>
                          </div>
                          <div className="grid grid-cols-3 gap-4 text-xs">
                            <div>
                              <span className="text-gray-400">Entropy: </span>
                              <span className="text-red-400">{point.entropy.toFixed(3)}</span>
                            </div>
                            <div>
                              <span className="text-gray-400">Stability: </span>
                              <span className="text-green-400">{point.stability.toFixed(3)}</span>
                            </div>
                            <div>
                              <span className="text-gray-400">Coherence: </span>
                              <span className="text-blue-400">{point.coherence.toFixed(3)}</span>
                            </div>
                          </div>
                          {point.price && (
                            <div className="text-xs text-gray-400 mt-2">
                              Price: {point.price.toFixed(5)}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Related Identities */}
                {inspectorData.relatedIdentities.length > 0 && (
                  <div className="bg-gray-900 rounded-lg p-4">
                    <h4 className="font-semibold mb-3 flex items-center gap-2">
                      <GitBranch className="w-5 h-5 text-cyan-400" />
                      Related Identities ({inspectorData.relatedIdentities.length})
                    </h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {inspectorData.relatedIdentities.map((related, idx) => (
                        <div
                          key={idx}
                          onClick={() => setSelectedKey(related.valkey_key)}
                          className="p-3 bg-gray-800 hover:bg-gray-700 rounded-lg cursor-pointer transition-colors"
                        >
                          <div className="flex items-center justify-between mb-2">
                            <span className="font-mono text-xs">{related.valkey_key.slice(-8)}</span>
                            <ArrowRight className="w-4 h-4 text-gray-400" />
                          </div>
                          <div className="text-xs space-y-1">
                            <div className="flex justify-between">
                              <span className="text-gray-400">Age:</span>
                              <span>{formatAge(related.age_seconds)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-400">Entropy Δ:</span>
                              <span className={related.entropy > inspectorData.currentPoint.entropy ? 'text-red-400' : 'text-green-400'}>
                                {(related.entropy - inspectorData.currentPoint.entropy > 0 ? '+' : '')}{(related.entropy - inspectorData.currentPoint.entropy).toFixed(3)}
                              </span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default IdentityInspector;