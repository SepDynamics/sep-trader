#include "core/pattern.h"
#include <cstring>
#include <algorithm>

// Implementation of conversion functions between canonical and POD types

namespace sep::core {

void convertToPOD(const QuantumState& src, QuantumStatePOD& dst) {
    dst.coherence = src.coherence;
    dst.stability = src.stability;
    dst.entropy = src.entropy;
    dst.mutation_rate = src.mutation_rate;
    dst.evolution_rate = src.evolution_rate;
    dst.energy = src.energy;
    dst.coupling_strength = src.coupling_strength;
    dst.phase = src.phase;
    dst.generation = src.generation;
    dst.mutation_count = src.mutation_count;
    dst.access_frequency = src.access_frequency;
    dst.status = static_cast<int>(src.status);
}

void convertFromPOD(const QuantumStatePOD& src, QuantumState& dst) {
    dst.coherence = src.coherence;
    dst.stability = src.stability;
    dst.entropy = src.entropy;
    dst.mutation_rate = src.mutation_rate;
    dst.evolution_rate = src.evolution_rate;
    dst.energy = src.energy;
    dst.coupling_strength = src.coupling_strength;
    dst.phase = src.phase;
    dst.generation = src.generation;
    dst.mutation_count = src.mutation_count;
    dst.access_frequency = src.access_frequency;
    dst.status = static_cast<QuantumState::Status>(src.status);
}

void convertToPOD(const PatternRelationship& src, PatternRelationshipPOD& dst) {
    // Direct assignment - target_id is uint32_t, not string
    dst.target_id = src.target_id;
    dst.strength = src.strength;
    dst.type = static_cast<uint32_t>(src.type);
}

void convertFromPOD(const PatternRelationshipPOD& src, PatternRelationship& dst) {
    dst.target_id = src.target_id;
    dst.strength = src.strength;
    dst.type = static_cast<RelationshipType>(src.type);
}

void convertToPOD(const Pattern& src, PatternPOD& dst) {
    // Direct assignment - id is uint32_t, not string
    dst.id = src.id;
    
    // Convert scalar values directly
    dst.position = src.position;
    dst.momentum = src.momentum;
    dst.velocity = src.velocity;
    
    // Convert attributes vector to fixed array
    dst.attribute_count = std::min(static_cast<uint32_t>(src.attributes.size()), 16U);
    for (uint32_t i = 0; i < dst.attribute_count; ++i) {
        dst.attributes[i] = src.attributes[i];
    }
    
    // Convert quantum state
    convertToPOD(src.quantum_state, dst.quantum_state);
    
    // Convert relationships (limited to array size)
    dst.relationship_count = std::min(static_cast<int>(src.relationships.size()), 16);
    for (int i = 0; i < dst.relationship_count; ++i) {
        convertToPOD(src.relationships[i], dst.relationships[i]);
    }
    
    // Convert parent IDs (limited to array size)
    dst.parent_count = std::min(static_cast<uint32_t>(src.parent_ids.size()), 16U);
    for (uint32_t i = 0; i < dst.parent_count; ++i) {
        dst.parent_ids[i] = src.parent_ids[i];
    }
    
    // Convert complex amplitude to real/imaginary parts
    dst.amplitude_real = src.amplitude.real();
    dst.amplitude_imag = src.amplitude.imag();
    
    // Copy scalar fields
    dst.timestamp = src.timestamp;
    dst.last_accessed = src.last_accessed;
    dst.last_modified = src.last_modified;
    dst.last_updated = src.last_updated;
    dst.generation = src.generation;
    dst.coherence = src.coherence;
}

void convertFromPOD(const PatternPOD& src, Pattern& dst) {
    // Copy ID directly - it's uint32_t in both
    dst.id = src.id;
    
    // Convert scalar values directly
    dst.position = src.position;
    dst.momentum = src.momentum;
    dst.velocity = src.velocity;
    
    // Convert attributes array back to vector
    dst.attributes.clear();
    dst.attributes.reserve(src.attribute_count);
    for (uint32_t i = 0; i < src.attribute_count; ++i) {
        dst.attributes.push_back(src.attributes[i]);
    }
    
    // Convert quantum state
    convertFromPOD(src.quantum_state, dst.quantum_state);
    
    // Convert relationships
    dst.relationships.clear();
    dst.relationships.reserve(src.relationship_count);
    for (uint32_t i = 0; i < src.relationship_count; ++i) {
        PatternRelationship rel;
        convertFromPOD(src.relationships[i], rel);
        dst.relationships.push_back(rel);
    }
    
    // Convert parent IDs
    dst.parent_ids.clear();
    dst.parent_ids.reserve(src.parent_count);
    for (uint32_t i = 0; i < src.parent_count; ++i) {
        dst.parent_ids.push_back(src.parent_ids[i]);
    }
    
    // Convert complex amplitude from real/imaginary parts (using double, not float)
    dst.amplitude = std::complex<double>(src.amplitude_real, src.amplitude_imag);
    
    // Copy scalar fields
    dst.timestamp = src.timestamp;
    dst.last_accessed = src.last_accessed;
    dst.last_modified = src.last_modified;
    dst.last_updated = src.last_updated;
    dst.generation = src.generation;
    dst.coherence = src.coherence;
}

} // namespace sep::core
