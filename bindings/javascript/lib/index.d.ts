/**
 * SEP DSL TypeScript Definitions
 * Advanced AGI Pattern Analysis Language for Node.js
 */

/**
 * Base error class for SEP DSL operations
 */
export class DSLError extends Error {
    constructor(message: string);
}

/**
 * Runtime error during DSL script execution
 */
export class DSLRuntimeError extends DSLError {
    constructor(message: string);
}

/**
 * Error accessing DSL variables
 */
export class DSLVariableError extends DSLError {
    constructor(message: string);
}

/**
 * Pattern analysis results
 */
export interface PatternResults {
    [key: string]: string;
    coherence?: string;
    entropy?: string;
    stability?: string;
    rupture?: string;
    signal?: string;
}

/**
 * Quick analysis results
 */
export interface QuickAnalysisResults {
    coherence: number;
    entropy: number;
}

/**
 * High-level SEP DSL Interpreter
 * 
 * Provides a clean JavaScript interface to the SEP DSL engine for quantum pattern analysis.
 */
export class DSLInterpreter {
    /**
     * Create a new DSL interpreter instance
     * @throws {DSLRuntimeError} If interpreter initialization fails
     */
    constructor();

    /**
     * Execute DSL script
     * @param script - DSL script content to execute
     * @throws {DSLRuntimeError} If script execution fails
     * @throws {TypeError} If script is not a string
     */
    execute(script: string): void;

    /**
     * Execute DSL script from file
     * @param filepath - Path to .sep file to execute
     * @throws {DSLRuntimeError} If file reading or execution fails
     */
    executeFile(filepath: string): Promise<void>;

    /**
     * Get variable value from DSL context
     * @param name - Variable name (supports dot notation like "pattern.variable")
     * @returns Variable value as string
     * @throws {DSLVariableError} If variable not found
     * @throws {TypeError} If name is not a string
     */
    getVariable(name: string): string;

    /**
     * Get all variables from a pattern as an object
     * @param patternName - Name of the pattern
     * @returns Object with variable names as keys and values as strings
     * @throws {DSLVariableError} If pattern not found or variables inaccessible
     */
    getPatternResults(patternName: string): PatternResults;

    /**
     * Quick coherence analysis helper
     * @param dataName - Name of data to analyze
     * @returns Coherence value (0.0 to 1.0)
     * @throws {DSLRuntimeError} If analysis fails
     */
    analyzeCoherence(dataName?: string): number;

    /**
     * Quick entropy analysis helper
     * @param dataName - Name of data to analyze
     * @returns Entropy value (0.0 to 1.0)
     * @throws {DSLRuntimeError} If analysis fails
     */
    analyzeEntropy(dataName?: string): number;
}

/**
 * Convenience function for quick pattern analysis
 * @param dataName - Name of data to analyze
 * @returns Object with 'coherence' and 'entropy' properties
 */
export function quickAnalysis(dataName?: string): QuickAnalysisResults;
