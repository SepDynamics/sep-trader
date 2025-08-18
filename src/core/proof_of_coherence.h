#ifndef SEP_CORE_PROOF_OF_COHERENCE_H
#define SEP_CORE_PROOF_OF_COHERENCE_H

#include <string>
#include <vector>
#include <cstdint>

#include "types.h" // For common SEP types

namespace sep::core {

/**
 * @struct Proof
 * @brief Represents a verifiable proof of computation, containing key metrics
 *        and cryptographic signatures derived from SEP's quantum-inspired algorithms.
 */
struct Proof {
    std::string data_id;
    std::string node_id;
    uint64_t timestamp; // Using uint64_t to align with SEP's typical timestamp format
    float coherence;
    float stability;
    float entropy;
    float correction_ratio;
    bool collapse;
    std::string metrics_hash;
    std::string signature;
};

/**
 * @class ProofGenerator
 * @brief A class responsible for producing and verifying Proof-of-Coherence records.
 *
 * This class integrates with the existing QFH and QBSA processors to generate
 * cryptographic proofs of computation on a given data set. It is designed to be
 * instantiated by a network node.
 */
class ProofGenerator {
public:
    /**
     * @brief Constructs a ProofGenerator.
     * @param node_id The identifier of the node generating the proof.
     * @param private_key The private key of the node, used for signing proofs.
     */
    ProofGenerator(std::string node_id, std::vector<uint8_t> private_key);

    /**
     * @brief Produces a Proof-of-Coherence for a given block of data.
     * @param data_id An identifier for the data being processed.
     * @param data The raw byte data to analyze.
     * @param expectation The expected pattern for QBSA analysis.
     * @return A signed Proof object.
     */
    Proof produce_proof(const std::string& data_id, const std::vector<uint8_t>& data, const std::vector<uint32_t>& expectation);

    /**
     * @brief Verifies a given Proof against the original data.
     * @param proof The Proof object to verify.
     * @param data The original data used to generate the proof.
     * @param expectation The expected pattern for QBSA analysis.
     * @param public_key The public key of the node that signed the proof.
     * @return True if the proof is valid, false otherwise.
     */
    static bool verify_proof(const Proof& proof, const std::vector<uint8_t>& data, const std::vector<uint32_t>& expectation, const std::vector<uint8_t>& public_key);

private:
    std::string node_id_;
    std::vector<uint8_t> private_key_;

    // Internal helper for signing the proof's canonical representation.
    std::string sign_proof_message(const std::string& message);
};

} // namespace sep::core

#endif // SEP_CORE_PROOF_OF_COHERENCE_H