# SEP DSL JavaScript SDK

Advanced AGI Pattern Analysis Language for Node.js

## Overview

SEP DSL is a domain-specific language for advanced AGI pattern analysis, providing quantum coherence analysis, entropy measurement, and sophisticated pattern recognition capabilities through an intuitive DSL syntax.

## Features

- **Real-time Quantum Analysis**: Coherence and entropy measurement
- **CUDA Acceleration**: GPU-powered pattern recognition  
- **Advanced AGI Algorithms**: Quantum field harmonics and trajectory damping
- **Production-Grade**: Mathematical validation and testing
- **Modern JavaScript**: Clean ES6+ interface with TypeScript support
- **Async/Await Support**: Promise-based file operations

## Installation

### Prerequisites

- Node.js 14.0+
- SEP DSL core library (`libsep.so`)
- C++ compiler for building native extensions

### From Source

```bash
# Clone the repository
git clone https://github.com/SepDynamics/sep-dsl.git
cd sep-dsl/bindings/javascript

# Install dependencies and build
npm install
npm run build
```

### Using npm (when available)

```bash
# Build from source - not yet published
```

## Quick Start

```javascript
const { DSLInterpreter, quickAnalysis } = require('sep-dsl');

// Create DSL interpreter
const dsl = new DSLInterpreter();

// Execute DSL script
dsl.execute(`
    pattern sensor_analysis {
        coherence = measure_coherence("sensor_data")
        entropy = measure_entropy("sensor_data") 
        print("Coherence:", coherence, "Entropy:", entropy)
    }
`);

// Get results
const coherence = dsl.getVariable("sensor_analysis.coherence");
const entropy = dsl.getVariable("sensor_analysis.entropy");

console.log(`Analysis: Coherence=${coherence}, Entropy=${entropy}`);
```

## API Reference

### DSLInterpreter

Main interface to the SEP DSL engine.

#### Constructor

```javascript
const dsl = new DSLInterpreter();
```

#### Methods

##### `execute(script: string): void`

Execute DSL script synchronously.

```javascript
dsl.execute(`
    pattern analysis {
        coherence = measure_coherence("data")
    }
`);
```

##### `executeFile(filepath: string): Promise<void>`

Execute DSL script from file asynchronously.

```javascript
await dsl.executeFile('./analysis.sep');
```

##### `getVariable(name: string): string`

Get variable value with dot notation support.

```javascript
const value = dsl.getVariable("pattern.variable");
```

##### `getPatternResults(patternName: string): Object`

Get all pattern variables as an object.

```javascript
const results = dsl.getPatternResults("my_pattern");
// { coherence: "0.85", entropy: "0.23", ... }
```

##### `analyzeCoherence(dataName?: string): number`

Quick coherence analysis helper.

```javascript
const coherence = dsl.analyzeCoherence("sensor_data");
// Returns: 0.75
```

##### `analyzeEntropy(dataName?: string): number`

Quick entropy analysis helper.

```javascript
const entropy = dsl.analyzeEntropy("sensor_data");
// Returns: 0.32
```

### Convenience Functions

##### `quickAnalysis(dataName?: string): {coherence: number, entropy: number}`

Fast coherence and entropy analysis in one call.

```javascript
const { quickAnalysis } = require('sep-dsl');

const results = quickAnalysis("my_data");
console.log(`Coherence: ${results.coherence}, Entropy: ${results.entropy}`);
```

## Examples

### Basic Pattern Analysis

```javascript
const { DSLInterpreter } = require('sep-dsl');

const dsl = new DSLInterpreter();

// Complex pattern with conditional logic
dsl.execute(`
    pattern comprehensive_analysis {
        coherence = measure_coherence("data_stream")
        entropy = measure_entropy("data_stream")
        bits = extract_bits("data_stream")
        rupture = qfh_analyze(bits)
        
        if (coherence > 0.8 && entropy < 0.3) {
            print("High-quality signal detected")
            signal = "BUY"
        } else {
            signal = "HOLD"
        }
    }
`);

// Get all results
const results = dsl.getPatternResults("comprehensive_analysis");
console.log("Analysis Results:", results);
```

### File-based Execution

```javascript
// Save DSL script to file
const fs = require('fs').promises;

await fs.writeFile('analysis.sep', `
    pattern file_analysis {
        coherence = measure_coherence("sensor_input")
        stability = measure_entropy("sensor_input")
        print("File-based analysis complete")
    }
`);

// Execute from file
const dsl = new DSLInterpreter();
await dsl.executeFile('analysis.sep');

const coherence = parseFloat(dsl.getVariable("file_analysis.coherence"));
console.log(`Coherence: ${coherence.toFixed(3)}`);
```

### Real-time Data Processing

```javascript
const { DSLInterpreter } = require('sep-dsl');

class RealTimeAnalyzer {
    constructor() {
        this.dsl = new DSLInterpreter();
    }
    
    async processDataStream(dataId) {
        // Quick coherence check
        const coherence = this.dsl.analyzeCoherence(dataId);
        
        if (coherence > 0.7) {
            // High coherence - do full analysis
            this.dsl.execute(`
                pattern detailed_analysis {
                    entropy = measure_entropy("${dataId}")
                    bits = extract_bits("${dataId}") 
                    rupture = qfh_analyze(bits)
                    optimized = manifold_optimize("pattern", 0.8, 0.9)
                }
            `);
            
            return this.dsl.getPatternResults("detailed_analysis");
        }
        
        return { coherence, status: "low_quality" };
    }
}

// Usage
const analyzer = new RealTimeAnalyzer();

setInterval(async () => {
    const results = await analyzer.processDataStream(`stream_${Date.now()}`);
    console.log("Analysis:", results);
}, 1000);
```

### Express.js Integration

```javascript
const express = require('express');
const { DSLInterpreter, quickAnalysis } = require('sep-dsl');

const app = express();
app.use(express.json());

// Endpoint for quick analysis
app.post('/analyze', (req, res) => {
    try {
        const { dataName = 'default_data' } = req.body;
        const results = quickAnalysis(dataName);
        res.json({ success: true, ...results });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Endpoint for custom DSL execution
app.post('/execute', (req, res) => {
    try {
        const { script } = req.body;
        const dsl = new DSLInterpreter();
        dsl.execute(script);
        res.json({ success: true, message: 'Script executed successfully' });
    } catch (error) {
        res.status(400).json({ error: error.message });
    }
});

app.listen(3000, () => {
    console.log('SEP DSL API server running on port 3000');
});
```

### TypeScript Usage

```typescript
import { DSLInterpreter, DSLRuntimeError, PatternResults } from 'sep-dsl';

class AnalysisService {
    private dsl: DSLInterpreter;
    
    constructor() {
        this.dsl = new DSLInterpreter();
    }
    
    analyzePattern(script: string): PatternResults {
        try {
            this.dsl.execute(script);
            return this.dsl.getPatternResults('analysis');
        } catch (error) {
            if (error instanceof DSLRuntimeError) {
                throw new Error(`Analysis failed: ${error.message}`);
            }
            throw error;
        }
    }
    
    getCoherence(dataName: string): number {
        return this.dsl.analyzeCoherence(dataName);
    }
}
```

## Error Handling

```javascript
const { DSLInterpreter, DSLRuntimeError, DSLVariableError } = require('sep-dsl');

try {
    const dsl = new DSLInterpreter();
    dsl.execute("invalid syntax here");
} catch (error) {
    if (error instanceof DSLRuntimeError) {
        console.error(`Script execution failed: ${error.message}`);
    }
}

try {
    const value = dsl.getVariable("nonexistent.variable");
} catch (error) {
    if (error instanceof DSLVariableError) {
        console.error(`Variable not found: ${error.message}`);
    }
}
```

## Testing

```bash
# Run test suite
npm test

# Run tests in watch mode
npm run test:watch

# Run linting
npm run lint
```

## Building

```bash
# Clean build
npm run clean && npm run build

# Development build
npm run build
```

## Performance Tips

1. **Reuse Interpreters**: Create one `DSLInterpreter` instance and reuse it
2. **Batch Processing**: Group multiple patterns in single script execution
3. **Quick Methods**: Use `analyzeCoherence()` and `analyzeEntropy()` for simple cases
4. **Async File Operations**: Use `executeFile()` for non-blocking file processing

## Integration Examples

### Socket.IO Real-time Analysis

```javascript
const io = require('socket.io')(server);
const { DSLInterpreter } = require('sep-dsl');

const dsl = new DSLInterpreter();

io.on('connection', (socket) => {
    socket.on('analyze_data', (data) => {
        try {
            const results = dsl.analyzeCoherence(data.stream);
            socket.emit('analysis_result', { coherence: results });
        } catch (error) {
            socket.emit('analysis_error', { error: error.message });
        }
    });
});
```

### MongoDB Integration

```javascript
const { MongoClient } = require('mongodb');
const { quickAnalysis } = require('sep-dsl');

async function analyzeStoredData() {
    const client = new MongoClient(url);
    await client.connect();
    
    const db = client.db('analytics');
    const collection = db.collection('sensor_data');
    
    const documents = await collection.find({}).toArray();
    
    for (const doc of documents) {
        const analysis = quickAnalysis(doc.data_stream);
        
        await collection.updateOne(
            { _id: doc._id },
            { $set: { analysis } }
        );
    }
    
    await client.close();
}
```

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

## License

MIT License - see [LICENSE](../../LICENSE) for details.

## Support

- Issues: https://github.com/SepDynamics/sep-dsl/issues
- Documentation: https://github.com/SepDynamics/sep-dsl#readme
- Email: contact@example.com
