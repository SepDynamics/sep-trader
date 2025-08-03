/**
 * SEP DSL - Advanced AGI Pattern Analysis Language for Node.js
 * 
 * A domain-specific language for quantum coherence analysis, entropy measurement,
 * and sophisticated pattern recognition in real-time data streams.
 */

const bindings = require('bindings');
const { DSLInterpreter: NativeDSLInterpreter } = bindings('sep_dsl_native');

/**
 * Base error class for SEP DSL operations
 */
class DSLError extends Error {
    constructor(message) {
        super(message);
        this.name = 'DSLError';
    }
}

/**
 * Runtime error during DSL script execution
 */
class DSLRuntimeError extends DSLError {
    constructor(message) {
        super(message);
        this.name = 'DSLRuntimeError';
    }
}

/**
 * Error accessing DSL variables
 */
class DSLVariableError extends DSLError {
    constructor(message) {
        super(message);
        this.name = 'DSLVariableError';
    }
}

/**
 * High-level SEP DSL Interpreter
 * 
 * Provides a clean JavaScript interface to the SEP DSL engine for quantum pattern analysis.
 * 
 * @example
 * const { DSLInterpreter } = require('sep-dsl');
 * 
 * const dsl = new DSLInterpreter();
 * dsl.execute(`
 *     pattern sensor_analysis {
 *         coherence = measure_coherence("sensor_data")
 *         entropy = measure_entropy("sensor_data")
 *         print("Coherence:", coherence, "Entropy:", entropy)
 *     }
 * `);
 * 
 * const coherence = dsl.getVariable("sensor_analysis.coherence");
 * console.log(`Analysis complete: ${coherence}`);
 */
class DSLInterpreter {
    /**
     * Create a new DSL interpreter instance
     * @throws {DSLRuntimeError} If interpreter initialization fails
     */
    constructor() {
        try {
            this._interpreter = new NativeDSLInterpreter();
        } catch (error) {
            throw new DSLRuntimeError(`Failed to initialize DSL interpreter: ${error.message}`);
        }
    }

    /**
     * Execute DSL script
     * @param {string} script - DSL script content to execute
     * @throws {DSLRuntimeError} If script execution fails
     * @throws {TypeError} If script is not a string
     */
    execute(script) {
        if (typeof script !== 'string') {
            throw new TypeError('Script must be a string');
        }

        try {
            this._interpreter.execute(script);
        } catch (error) {
            throw new DSLRuntimeError(`Script execution failed: ${error.message}`);
        }
    }

    /**
     * Execute DSL script from file
     * @param {string} filepath - Path to .sep file to execute
     * @throws {DSLRuntimeError} If file reading or execution fails
     */
    async executeFile(filepath) {
        const fs = require('fs').promises;
        
        try {
            const script = await fs.readFile(filepath, 'utf-8');
            this.execute(script);
        } catch (error) {
            if (error.code === 'ENOENT') {
                throw new DSLRuntimeError(`File not found: ${filepath}`);
            }
            throw new DSLRuntimeError(`Failed to read file ${filepath}: ${error.message}`);
        }
    }

    /**
     * Get variable value from DSL context
     * @param {string} name - Variable name (supports dot notation like "pattern.variable")
     * @returns {string} Variable value as string
     * @throws {DSLVariableError} If variable not found
     * @throws {TypeError} If name is not a string
     */
    getVariable(name) {
        if (typeof name !== 'string') {
            throw new TypeError('Variable name must be a string');
        }

        try {
            return this._interpreter.getVariable(name);
        } catch (error) {
            throw new DSLVariableError(`Variable '${name}' not found: ${error.message}`);
        }
    }

    /**
     * Get all variables from a pattern as an object
     * @param {string} patternName - Name of the pattern
     * @returns {Object} Object with variable names as keys and values as strings
     * @throws {DSLVariableError} If pattern not found or variables inaccessible
     */
    getPatternResults(patternName) {
        // Common pattern variables to check
        const commonVars = ['coherence', 'entropy', 'stability', 'rupture', 'signal'];
        const results = {};
        
        for (const varName of commonVars) {
            const fullName = `${patternName}.${varName}`;
            try {
                const value = this.getVariable(fullName);
                results[varName] = value;
            } catch (error) {
                // Variable doesn't exist, skip it
                continue;
            }
        }
        
        if (Object.keys(results).length === 0) {
            throw new DSLVariableError(`No variables found in pattern '${patternName}'`);
        }
        
        return results;
    }

    /**
     * Quick coherence analysis helper
     * @param {string} dataName - Name of data to analyze
     * @returns {number} Coherence value (0.0 to 1.0)
     * @throws {DSLRuntimeError} If analysis fails
     */
    analyzeCoherence(dataName = 'sensor_data') {
        const script = `
        pattern quick_coherence {
            coherence = measure_coherence("${dataName}")
        }
        `;
        this.execute(script);
        const coherenceStr = this.getVariable('quick_coherence.coherence');
        const coherence = parseFloat(coherenceStr);
        
        if (isNaN(coherence)) {
            throw new DSLRuntimeError(`Invalid coherence value: ${coherenceStr}`);
        }
        
        return coherence;
    }

    /**
     * Quick entropy analysis helper
     * @param {string} dataName - Name of data to analyze
     * @returns {number} Entropy value (0.0 to 1.0)
     * @throws {DSLRuntimeError} If analysis fails
     */
    analyzeEntropy(dataName = 'sensor_data') {
        const script = `
        pattern quick_entropy {
            entropy = measure_entropy("${dataName}")
        }
        `;
        this.execute(script);
        const entropyStr = this.getVariable('quick_entropy.entropy');
        const entropy = parseFloat(entropyStr);
        
        if (isNaN(entropy)) {
            throw new DSLRuntimeError(`Invalid entropy value: ${entropyStr}`);
        }
        
        return entropy;
    }
}

/**
 * Convenience function for quick pattern analysis
 * @param {string} dataName - Name of data to analyze
 * @returns {Object} Object with 'coherence' and 'entropy' properties
 */
function quickAnalysis(dataName = 'sensor_data') {
    const dsl = new DSLInterpreter();
    const script = `
    pattern analysis {
        coherence = measure_coherence("${dataName}")
        entropy = measure_entropy("${dataName}")
    }
    `;
    dsl.execute(script);
    
    return {
        coherence: parseFloat(dsl.getVariable('analysis.coherence')),
        entropy: parseFloat(dsl.getVariable('analysis.entropy'))
    };
}

module.exports = {
    DSLInterpreter,
    DSLError,
    DSLRuntimeError,
    DSLVariableError,
    quickAnalysis
};
