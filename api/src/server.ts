/**
 * SEP DSL API Server
 * 
 * Provides REST API and WebSocket endpoints for DSL execution
 */

import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import { createServer } from 'http';
import { Server as SocketIOServer } from 'socket.io';
import swaggerUi from 'swagger-ui-express';
import YAML from 'yamljs';
import { RateLimiterMemory } from 'rate-limiter-flexible';
import Joi from 'joi';

// Mock DSL interpreter - in real implementation, use the actual bindings
class MockDSLInterpreter {
    execute(script: string): void {
        // Simulate execution
        console.log(`Executing DSL script: ${script.substring(0, 50)}...`);
    }

    getVariable(name: string): string {
        // Simulate variable retrieval
        const mockValues: { [key: string]: string } = {
            'pattern.coherence': '0.85',
            'pattern.entropy': '0.23',
            'analysis.coherence': '0.72',
            'analysis.entropy': '0.41'
        };
        return mockValues[name] || Math.random().toFixed(3);
    }

    analyzeCoherence(dataName: string): number {
        return Math.random();
    }

    analyzeEntropy(dataName: string): number {
        return Math.random();
    }
}

// Rate limiting
const rateLimiter = new RateLimiterMemory({
    points: 100, // 100 requests
    duration: 60, // per 60 seconds
});

// Validation schemas
const executeSchema = Joi.object({
    script: Joi.string().required().max(10000),
    timeout: Joi.number().optional().min(1).max(30).default(10)
});

const analyzeSchema = Joi.object({
    dataName: Joi.string().optional().default('sensor_data'),
    type: Joi.string().valid('coherence', 'entropy', 'both').default('both')
});

const variableSchema = Joi.object({
    name: Joi.string().required().pattern(/^[a-zA-Z_][a-zA-Z0-9_.]*$/)
});

const app = express();
const server = createServer(app);
const io = new SocketIOServer(server, {
    cors: {
        origin: "*",
        methods: ["GET", "POST"]
    }
});

// Middleware
app.use(helmet());
app.use(cors());
app.use(morgan('combined'));
app.use(express.json({ limit: '1mb' }));

// Rate limiting middleware
app.use(async (req, res, next) => {
    try {
        await rateLimiter.consume(req.ip);
        next();
    } catch (rateLimiterRes) {
        res.status(429).json({
            error: 'Too Many Requests',
            retryAfter: Math.round(rateLimiterRes.msBeforeNext / 1000) || 1
        });
    }
});

// Health check
app.get('/health', (req, res) => {
    res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        version: '1.0.0'
    });
});

// API Documentation
const swaggerDocument = {
    openapi: '3.0.0',
    info: {
        title: 'SEP DSL API',
        version: '1.0.0',
        description: 'REST API for SEP DSL pattern analysis'
    },
    servers: [
        {
            url: 'http://localhost:3000',
            description: 'Development server'
        }
    ],
    paths: {
        '/api/execute': {
            post: {
                summary: 'Execute DSL script',
                requestBody: {
                    required: true,
                    content: {
                        'application/json': {
                            schema: {
                                type: 'object',
                                properties: {
                                    script: { type: 'string' },
                                    timeout: { type: 'number' }
                                },
                                required: ['script']
                            }
                        }
                    }
                },
                responses: {
                    '200': {
                        description: 'Script executed successfully'
                    }
                }
            }
        },
        '/api/analyze': {
            post: {
                summary: 'Quick analysis',
                requestBody: {
                    required: true,
                    content: {
                        'application/json': {
                            schema: {
                                type: 'object',
                                properties: {
                                    dataName: { type: 'string' },
                                    type: { 
                                        type: 'string',
                                        enum: ['coherence', 'entropy', 'both']
                                    }
                                }
                            }
                        }
                    }
                },
                responses: {
                    '200': {
                        description: 'Analysis completed'
                    }
                }
            }
        }
    }
};

app.use('/docs', swaggerUi.serve, swaggerUi.setup(swaggerDocument));

// REST API Routes

// Execute DSL script
app.post('/api/execute', async (req, res) => {
    try {
        const { error, value } = executeSchema.validate(req.body);
        if (error) {
            return res.status(400).json({
                error: 'Validation failed',
                details: error.details
            });
        }

        const { script, timeout } = value;
        const dsl = new MockDSLInterpreter();
        
        // Execute with timeout
        const executionPromise = new Promise<void>((resolve, reject) => {
            try {
                dsl.execute(script);
                resolve();
            } catch (err) {
                reject(err);
            }
        });

        const timeoutPromise = new Promise<never>((_, reject) => {
            setTimeout(() => reject(new Error('Execution timeout')), timeout * 1000);
        });

        await Promise.race([executionPromise, timeoutPromise]);

        res.json({
            success: true,
            message: 'Script executed successfully',
            timestamp: new Date().toISOString()
        });

    } catch (err: any) {
        res.status(500).json({
            error: 'Execution failed',
            message: err.message
        });
    }
});

// Get variable value
app.get('/api/variable/:name', (req, res) => {
    try {
        const { error } = variableSchema.validate({ name: req.params.name });
        if (error) {
            return res.status(400).json({
                error: 'Invalid variable name',
                details: error.details
            });
        }

        const dsl = new MockDSLInterpreter();
        const value = dsl.getVariable(req.params.name);

        res.json({
            name: req.params.name,
            value,
            timestamp: new Date().toISOString()
        });

    } catch (err: any) {
        res.status(404).json({
            error: 'Variable not found',
            message: err.message
        });
    }
});

// Quick analysis
app.post('/api/analyze', (req, res) => {
    try {
        const { error, value } = analyzeSchema.validate(req.body);
        if (error) {
            return res.status(400).json({
                error: 'Validation failed',
                details: error.details
            });
        }

        const { dataName, type } = value;
        const dsl = new MockDSLInterpreter();
        
        const results: any = {
            dataName,
            timestamp: new Date().toISOString()
        };

        if (type === 'coherence' || type === 'both') {
            results.coherence = dsl.analyzeCoherence(dataName);
        }

        if (type === 'entropy' || type === 'both') {
            results.entropy = dsl.analyzeEntropy(dataName);
        }

        res.json(results);

    } catch (err: any) {
        res.status(500).json({
            error: 'Analysis failed',
            message: err.message
        });
    }
});

// Batch processing
app.post('/api/batch', async (req, res) => {
    try {
        const { scripts } = req.body;
        
        if (!Array.isArray(scripts) || scripts.length === 0) {
            return res.status(400).json({
                error: 'Scripts array is required'
            });
        }

        if (scripts.length > 10) {
            return res.status(400).json({
                error: 'Maximum 10 scripts per batch'
            });
        }

        const dsl = new MockDSLInterpreter();
        const results = [];

        for (let i = 0; i < scripts.length; i++) {
            try {
                dsl.execute(scripts[i]);
                results.push({
                    index: i,
                    success: true,
                    message: 'Executed successfully'
                });
            } catch (err: any) {
                results.push({
                    index: i,
                    success: false,
                    error: err.message
                });
            }
        }

        res.json({
            batchId: `batch_${Date.now()}`,
            results,
            timestamp: new Date().toISOString()
        });

    } catch (err: any) {
        res.status(500).json({
            error: 'Batch processing failed',
            message: err.message
        });
    }
});

// WebSocket handling
io.on('connection', (socket) => {
    console.log(`Client connected: ${socket.id}`);

    // Real-time analysis
    socket.on('analyze', (data) => {
        try {
            const dsl = new MockDSLInterpreter();
            const result = {
                coherence: dsl.analyzeCoherence(data.dataName || 'sensor_data'),
                entropy: dsl.analyzeEntropy(data.dataName || 'sensor_data'),
                timestamp: new Date().toISOString()
            };
            
            socket.emit('analysis_result', result);
        } catch (error: any) {
            socket.emit('analysis_error', { error: error.message });
        }
    });

    // Execute script
    socket.on('execute', (data) => {
        try {
            const dsl = new MockDSLInterpreter();
            dsl.execute(data.script);
            
            socket.emit('execution_result', {
                success: true,
                timestamp: new Date().toISOString()
            });
        } catch (error: any) {
            socket.emit('execution_error', { error: error.message });
        }
    });

    // Subscribe to real-time updates
    socket.on('subscribe', (data) => {
        const room = `updates_${data.dataName || 'default'}`;
        socket.join(room);
        socket.emit('subscribed', { room });
        
        // Simulate periodic updates
        const interval = setInterval(() => {
            const dsl = new MockDSLInterpreter();
            const update = {
                dataName: data.dataName || 'sensor_data',
                coherence: dsl.analyzeCoherence('live_data'),
                entropy: dsl.analyzeEntropy('live_data'),
                timestamp: new Date().toISOString()
            };
            
            io.to(room).emit('data_update', update);
        }, 5000);

        socket.on('disconnect', () => {
            clearInterval(interval);
        });
    });

    socket.on('disconnect', () => {
        console.log(`Client disconnected: ${socket.id}`);
    });
});

// Error handling
app.use((err: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
    console.error('Unhandled error:', err);
    res.status(500).json({
        error: 'Internal server error',
        message: process.env.NODE_ENV === 'development' ? err.message : 'Something went wrong'
    });
});

// 404 handler
app.use((req, res) => {
    res.status(404).json({
        error: 'Not found',
        message: `Route ${req.method} ${req.path} not found`
    });
});

const PORT = process.env.PORT || 3000;

server.listen(PORT, () => {
    console.log(`🚀 SEP DSL API Server running on port ${PORT}`);
    console.log(`📚 API Documentation: http://localhost:${PORT}/docs`);
    console.log(`🔗 WebSocket endpoint: ws://localhost:${PORT}`);
});

export { app, server, io };
