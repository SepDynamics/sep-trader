/**
 * SEP DSL JavaScript Test Suite
 */

const { DSLInterpreter, DSLError, DSLRuntimeError, DSLVariableError, quickAnalysis } = require('../lib/index');
const fs = require('fs').promises;
const path = require('path');

describe('DSLInterpreter', () => {
    let dsl;

    beforeEach(() => {
        dsl = new DSLInterpreter();
    });

    describe('constructor', () => {
        test('should create DSL interpreter instance', () => {
            expect(dsl).toBeInstanceOf(DSLInterpreter);
        });
    });

    describe('execute', () => {
        test('should execute simple pattern', () => {
            const script = `
            pattern test_pattern {
                coherence = measure_coherence("test_data")
                entropy = measure_entropy("test_data")
                print("Test pattern executed")
            }
            `;
            
            expect(() => dsl.execute(script)).not.toThrow();
        });

        test('should throw error for invalid script type', () => {
            expect(() => dsl.execute(123)).toThrow(TypeError);
            expect(() => dsl.execute(null)).toThrow(TypeError);
            expect(() => dsl.execute(undefined)).toThrow(TypeError);
        });

        test('should throw DSLRuntimeError for invalid syntax', () => {
            const invalidScript = 'this is not valid DSL syntax';
            expect(() => dsl.execute(invalidScript)).toThrow(DSLRuntimeError);
        });

        test('should handle empty script', () => {
            expect(() => dsl.execute('')).not.toThrow();
        });
    });

    describe('getVariable', () => {
        test('should get variable from executed pattern', () => {
            const script = `
            pattern var_test {
                coherence = measure_coherence("sensor_data")
                entropy = measure_entropy("sensor_data")
            }
            `;
            dsl.execute(script);

            const coherence = dsl.getVariable('var_test.coherence');
            const entropy = dsl.getVariable('var_test.entropy');

            expect(typeof coherence).toBe('string');
            expect(typeof entropy).toBe('string');
            
            // Should be valid numbers
            expect(parseFloat(coherence)).not.toBeNaN();
            expect(parseFloat(entropy)).not.toBeNaN();
        });

        test('should throw error for invalid variable name type', () => {
            expect(() => dsl.getVariable(123)).toThrow(TypeError);
            expect(() => dsl.getVariable(null)).toThrow(TypeError);
        });

        test('should throw DSLVariableError for non-existent variable', () => {
            expect(() => dsl.getVariable('nonexistent.variable')).toThrow(DSLVariableError);
        });
    });

    describe('getPatternResults', () => {
        test('should get all pattern results as object', () => {
            const script = `
            pattern results_test {
                coherence = measure_coherence("data")
                entropy = measure_entropy("data")
            }
            `;
            dsl.execute(script);

            const results = dsl.getPatternResults('results_test');
            
            expect(typeof results).toBe('object');
            expect(results).toHaveProperty('coherence');
            expect(results).toHaveProperty('entropy');
            expect(typeof results.coherence).toBe('string');
            expect(typeof results.entropy).toBe('string');
        });

        test('should throw error for pattern with no accessible variables', () => {
            // Execute empty script
            dsl.execute('// Just a comment');
            
            expect(() => dsl.getPatternResults('nonexistent_pattern')).toThrow(DSLVariableError);
        });
    });

    describe('analyzeCoherence', () => {
        test('should return coherence as number', () => {
            const coherence = dsl.analyzeCoherence('test_data');
            
            expect(typeof coherence).toBe('number');
            expect(coherence).toBeGreaterThanOrEqual(0.0);
            expect(coherence).toBeLessThanOrEqual(1.0);
        });

        test('should use default data name', () => {
            const coherence = dsl.analyzeCoherence();
            expect(typeof coherence).toBe('number');
        });
    });

    describe('analyzeEntropy', () => {
        test('should return entropy as number', () => {
            const entropy = dsl.analyzeEntropy('test_data');
            
            expect(typeof entropy).toBe('number');
            expect(entropy).toBeGreaterThanOrEqual(0.0);
            expect(entropy).toBeLessThanOrEqual(1.0);
        });

        test('should use default data name', () => {
            const entropy = dsl.analyzeEntropy();
            expect(typeof entropy).toBe('number');
        });
    });

    describe('executeFile', () => {
        const testFile = path.join(__dirname, 'test_script.sep');

        afterEach(async () => {
            try {
                await fs.unlink(testFile);
            } catch (error) {
                // File might not exist, ignore
            }
        });

        test('should execute DSL script from file', async () => {
            const scriptContent = `
            pattern file_test {
                coherence = measure_coherence("file_data")
                print("Executed from file")
            }
            `;

            await fs.writeFile(testFile, scriptContent);
            await dsl.executeFile(testFile);

            const coherence = dsl.getVariable('file_test.coherence');
            expect(typeof coherence).toBe('string');
        });

        test('should throw error for non-existent file', async () => {
            await expect(dsl.executeFile('nonexistent.sep')).rejects.toThrow(DSLRuntimeError);
        });
    });
});

describe('quickAnalysis', () => {
    test('should return coherence and entropy', () => {
        const results = quickAnalysis('test_data');
        
        expect(typeof results).toBe('object');
        expect(results).toHaveProperty('coherence');
        expect(results).toHaveProperty('entropy');
        expect(typeof results.coherence).toBe('number');
        expect(typeof results.entropy).toBe('number');
        expect(results.coherence).toBeGreaterThanOrEqual(0.0);
        expect(results.coherence).toBeLessThanOrEqual(1.0);
        expect(results.entropy).toBeGreaterThanOrEqual(0.0);
        expect(results.entropy).toBeLessThanOrEqual(1.0);
    });

    test('should use default data name', () => {
        const results = quickAnalysis();
        expect(typeof results.coherence).toBe('number');
        expect(typeof results.entropy).toBe('number');
    });
});

describe('Error classes', () => {
    test('DSLError should be proper Error subclass', () => {
        const error = new DSLError('test message');
        expect(error).toBeInstanceOf(Error);
        expect(error).toBeInstanceOf(DSLError);
        expect(error.name).toBe('DSLError');
        expect(error.message).toBe('test message');
    });

    test('DSLRuntimeError should be proper DSLError subclass', () => {
        const error = new DSLRuntimeError('runtime error');
        expect(error).toBeInstanceOf(Error);
        expect(error).toBeInstanceOf(DSLError);
        expect(error).toBeInstanceOf(DSLRuntimeError);
        expect(error.name).toBe('DSLRuntimeError');
        expect(error.message).toBe('runtime error');
    });

    test('DSLVariableError should be proper DSLError subclass', () => {
        const error = new DSLVariableError('variable error');
        expect(error).toBeInstanceOf(Error);
        expect(error).toBeInstanceOf(DSLError);
        expect(error).toBeInstanceOf(DSLVariableError);
        expect(error.name).toBe('DSLVariableError');
        expect(error.message).toBe('variable error');
    });
});

describe('Integration tests', () => {
    test('should handle complex pattern analysis workflow', () => {
        const dsl = new DSLInterpreter();
        
        // Execute comprehensive analysis
        dsl.execute(`
            pattern comprehensive_analysis {
                coherence = measure_coherence("data_stream")
                entropy = measure_entropy("data_stream")
                bits = extract_bits("data_stream")
                rupture = qfh_analyze(bits)
            }
        `);

        // Get individual values
        const coherence = parseFloat(dsl.getVariable('comprehensive_analysis.coherence'));
        const entropy = parseFloat(dsl.getVariable('comprehensive_analysis.entropy'));

        // Get all results
        const results = dsl.getPatternResults('comprehensive_analysis');

        expect(coherence).toBeGreaterThanOrEqual(0.0);
        expect(entropy).toBeGreaterThanOrEqual(0.0);
        expect(Object.keys(results).length).toBeGreaterThan(0);
    });

    test('should handle multiple patterns in sequence', () => {
        const dsl = new DSLInterpreter();
        
        // Execute first pattern
        dsl.execute(`
            pattern first_analysis {
                coherence = measure_coherence("data1")
            }
        `);

        // Execute second pattern
        dsl.execute(`
            pattern second_analysis {
                entropy = measure_entropy("data2")
            }
        `);

        // Both should be accessible
        const coherence = dsl.getVariable('first_analysis.coherence');
        const entropy = dsl.getVariable('second_analysis.entropy');

        expect(typeof coherence).toBe('string');
        expect(typeof entropy).toBe('string');
    });
});
