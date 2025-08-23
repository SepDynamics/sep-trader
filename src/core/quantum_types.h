#pragma once

#include <vector>
#include <complex>
#include <cstdint>

namespace sep {
namespace quantum {

/**
 * Relationship types between patterns
 */
enum class RelationshipType {
    NONE = 0,
    CAUSAL,
    CORRELATION,
    INTERFERENCE,
    ENTANGLEMENT,
    SUPERPOSITION,
    COHERENCE
};

/**
 * Pattern relationship structure
 */
struct PatternRelationship {
    uint32_t target_id{0};
    double strength{0.0};
    RelationshipType type{RelationshipType::NONE};
    
    bool operator==(const PatternRelationship& other) const {
        return target_id == other.target_id && 
               strength == other.strength && 
               type == other.type;
    }
};

/**
 * Quantum state structure
 */
struct QuantumState {
    enum class Status {
        ACTIVE = 0,
        DORMANT,
        COLLAPSED,
        EVOLVING,
        STABLE,
        UNSTABLE
    };
    
    double coherence{0.0};
    double stability{0.0};
    double entropy{0.0};
    double mutation_rate{0.0};
    double evolution_rate{0.0};
    double energy{0.0};
    double coupling_strength{0.0};
    double phase{0.0};
    uint32_t generation{0};
    uint32_t mutation_count{0};
    uint32_t access_frequency{0};
    Status status{Status::ACTIVE};
    
    bool operator==(const QuantumState& other) const {
        return coherence == other.coherence &&
               stability == other.stability &&
               entropy == other.entropy &&
               status == other.status;
    }
};

/**
 * Pattern structure for quantum field harmonics
 */
struct Pattern {
    uint32_t id{0};
    
    // Spatial properties
    double position{0.0};
    double momentum{0.0};
    double velocity{0.0};
    std::vector<double> attributes;
    
    // Quantum properties
    QuantumState quantum_state;
    std::complex<double> amplitude{0.0, 0.0};
    
    // Relationships
    std::vector<PatternRelationship> relationships;
    std::vector<uint32_t> parent_ids;

    // Memory residency tier
    enum class MemoryTier { Hot, Warm, Cold };
    MemoryTier tier{MemoryTier::Hot};

    // Temporal properties
    uint64_t timestamp{0};
    uint64_t last_accessed{0};
    uint64_t last_modified{0};
    uint64_t last_updated{0};
    uint32_t generation{0};
    double coherence{0.0};
    
    bool operator==(const Pattern& other) const {
        return id == other.id && 
               coherence == other.coherence &&
               quantum_state == other.quantum_state;
    }
};

/**
 * POD versions for serialization/GPU transfer
 */
struct QuantumStatePOD {
    double coherence;
    double stability;
    double entropy;
    double mutation_rate;
    double evolution_rate;
    double energy;
    double coupling_strength;
    double phase;
    uint32_t generation;
    uint32_t mutation_count;
    uint32_t access_frequency;
    uint32_t status; // Cast from QuantumState::Status
};

struct PatternRelationshipPOD {
    uint32_t target_id;
    double strength;
    uint32_t type; // Cast from RelationshipType
};

struct PatternPOD {
    uint32_t id;
    
    // Spatial properties
    double position;
    double momentum;
    double velocity;
    double attributes[16]; // Fixed size for POD
    uint32_t attribute_count;
    
    // Quantum properties
    QuantumStatePOD quantum_state;
    double amplitude_real;
    double amplitude_imag;
    
    // Relationships - fixed size for POD
    PatternRelationshipPOD relationships[32];
    uint32_t relationship_count;
    uint32_t parent_ids[16];
    uint32_t parent_count;
    
    // Temporal properties
    uint64_t timestamp;
    uint64_t last_accessed;
    uint64_t last_modified;
    uint64_t last_updated;
    uint32_t generation;
    double coherence;
};

} // namespace quantum
} // namespace sep