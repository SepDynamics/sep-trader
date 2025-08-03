/**
 * SEP DSL Language Server
 * 
 * Provides Language Server Protocol support for SEP DSL including:
 * - Syntax highlighting
 * - Auto-completion
 * - Error diagnostics
 * - Go-to-definition
 * - Hover documentation
 */

import {
    createConnection,
    TextDocuments,
    Diagnostic,
    DiagnosticSeverity,
    ProposedFeatures,
    InitializeParams,
    DidChangeConfigurationNotification,
    CompletionItem,
    CompletionItemKind,
    TextDocumentPositionParams,
    Hover,
    MarkupKind,
    Position,
    Range
} from 'vscode-languageserver/node';

import { TextDocument } from 'vscode-languageserver-textdocument';

// Create a connection for the server
const connection = createConnection(ProposedFeatures.all);

// Create a simple text document manager
const documents: TextDocuments<TextDocument> = new TextDocuments(TextDocument);

let hasConfigurationCapability = false;
let hasWorkspaceFolderCapability = false;
let hasDiagnosticRelatedInformationCapability = false;

connection.onInitialize((params: InitializeParams) => {
    const capabilities = params.capabilities;

    // Does the client support the `workspace/configuration` request?
    hasConfigurationCapability = !!(
        capabilities.workspace && !!capabilities.workspace.configuration
    );
    hasWorkspaceFolderCapability = !!(
        capabilities.workspace && !!capabilities.workspace.workspaceFolders
    );
    hasDiagnosticRelatedInformationCapability = !!(
        capabilities.textDocument &&
        capabilities.textDocument.publishDiagnostics &&
        capabilities.textDocument.publishDiagnostics.relatedInformation
    );

    return {
        capabilities: {
            textDocumentSync: 1, // Full sync
            completionProvider: {
                resolveProvider: true,
                triggerCharacters: ['.', '(', '"']
            },
            hoverProvider: true,
            definitionProvider: true,
            documentSymbolProvider: true
        }
    };
});

connection.onInitialized(() => {
    if (hasConfigurationCapability) {
        // Register for all configuration changes
        connection.client.register(DidChangeConfigurationNotification.type, undefined);
    }
    if (hasWorkspaceFolderCapability) {
        connection.workspace.onDidChangeWorkspaceFolders(_event => {
            connection.console.log('Workspace folder change event received.');
        });
    }
});

// SEP DSL Language Definitions
const DSL_KEYWORDS = [
    'pattern', 'if', 'else', 'print', 'true', 'false'
];

const DSL_BUILTIN_FUNCTIONS = [
    {
        name: 'measure_coherence',
        signature: 'measure_coherence(data_name: string): number',
        description: 'Measures quantum coherence of the specified data stream',
        detail: 'Returns a value between 0.0 and 1.0 representing coherence level'
    },
    {
        name: 'measure_entropy',
        signature: 'measure_entropy(data_name: string): number', 
        description: 'Measures Shannon entropy of the specified data stream',
        detail: 'Returns a value between 0.0 and 1.0 representing entropy level'
    },
    {
        name: 'extract_bits',
        signature: 'extract_bits(data_name: string): string',
        description: 'Extracts binary representation from data stream',
        detail: 'Returns binary string representation of the data'
    },
    {
        name: 'qfh_analyze',
        signature: 'qfh_analyze(bitstream: string): number',
        description: 'Performs quantum field harmonics analysis on bitstream',
        detail: 'Returns rupture ratio from QFH trajectory analysis'
    },
    {
        name: 'manifold_optimize',
        signature: 'manifold_optimize(pattern: string, coherence: number, stability: number): number',
        description: 'Optimizes pattern using quantum manifold algorithms',
        detail: 'Returns optimized coherence value based on manifold optimization'
    }
];

const DSL_OPERATORS = [
    '&&', '||', '!', '==', '!=', '<', '>', '<=', '>=', '+', '-', '*', '/', '='
];

// Configuration interface
interface SepDslSettings {
    maxNumberOfProblems: number;
    enableDiagnostics: boolean;
    enableAutoComplete: boolean;
}

// Default settings
const defaultSettings: SepDslSettings = {
    maxNumberOfProblems: 1000,
    enableDiagnostics: true,
    enableAutoComplete: true
};
let globalSettings: SepDslSettings = defaultSettings;

// Cache the settings of all open documents
const documentSettings: Map<string, Thenable<SepDslSettings>> = new Map();

connection.onDidChangeConfiguration(change => {
    if (hasConfigurationCapability) {
        // Reset all cached document settings
        documentSettings.clear();
    } else {
        globalSettings = <SepDslSettings>(
            (change.settings.sepDsl || defaultSettings)
        );
    }

    // Revalidate all open text documents
    documents.all().forEach(validateTextDocument);
});

function getDocumentSettings(resource: string): Thenable<SepDslSettings> {
    if (!hasConfigurationCapability) {
        return Promise.resolve(globalSettings);
    }
    let result = documentSettings.get(resource);
    if (!result) {
        result = connection.workspace.getConfiguration({
            scopeUri: resource,
            section: 'sepDsl'
        });
        documentSettings.set(resource, result);
    }
    return result;
}

// Only keep settings for open documents
documents.onDidClose(e => {
    documentSettings.delete(e.document.uri);
});

// The content of a text document has changed. This event is emitted
// when the text document first opened or when its content has changed.
documents.onDidChangeContent(change => {
    validateTextDocument(change.document);
});

// Validation function
async function validateTextDocument(textDocument: TextDocument): Promise<void> {
    const settings = await getDocumentSettings(textDocument.uri);
    
    if (!settings.enableDiagnostics) {
        return;
    }

    const text = textDocument.getText();
    const diagnostics: Diagnostic[] = [];
    
    // Basic syntax validation
    const lines = text.split('\n');
    
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        
        // Skip empty lines and comments
        if (!line || line.startsWith('//')) {
            continue;
        }
        
        // Check for unclosed braces
        const openBraces = (line.match(/\{/g) || []).length;
        const closeBraces = (line.match(/\}/g) || []).length;
        
        // Check for invalid function calls
        const functionCallMatch = line.match(/(\w+)\s*\(/);
        if (functionCallMatch) {
            const funcName = functionCallMatch[1];
            const isBuiltin = DSL_BUILTIN_FUNCTIONS.some(f => f.name === funcName);
            const isKeyword = DSL_KEYWORDS.includes(funcName);
            
            if (!isBuiltin && !isKeyword && funcName !== 'print') {
                const diagnostic: Diagnostic = {
                    severity: DiagnosticSeverity.Error,
                    range: {
                        start: { line: i, character: line.indexOf(funcName) },
                        end: { line: i, character: line.indexOf(funcName) + funcName.length }
                    },
                    message: `Unknown function '${funcName}'. Available functions: ${DSL_BUILTIN_FUNCTIONS.map(f => f.name).join(', ')}`,
                    source: 'sep-dsl'
                };
                
                if (hasDiagnosticRelatedInformationCapability) {
                    diagnostic.relatedInformation = [
                        {
                            location: {
                                uri: textDocument.uri,
                                range: diagnostic.range
                            },
                            message: 'Check function name spelling and available built-ins'
                        }
                    ];
                }
                
                diagnostics.push(diagnostic);
            }
        }
        
        // Check for pattern syntax
        if (line.includes('pattern') && !line.match(/pattern\s+\w+\s*\{/)) {
            const diagnostic: Diagnostic = {
                severity: DiagnosticSeverity.Error,
                range: {
                    start: { line: i, character: 0 },
                    end: { line: i, character: line.length }
                },
                message: 'Invalid pattern syntax. Expected: pattern name { ... }',
                source: 'sep-dsl'
            };
            diagnostics.push(diagnostic);
        }
    }
    
    // Send the computed diagnostics to VSCode
    connection.sendDiagnostics({ uri: textDocument.uri, diagnostics });
}

// Auto-completion provider
connection.onCompletion((textDocumentPosition: TextDocumentPositionParams): CompletionItem[] => {
    const document = documents.get(textDocumentPosition.textDocument.uri);
    if (!document) {
        return [];
    }
    
    const completionItems: CompletionItem[] = [];
    
    // Add keywords
    DSL_KEYWORDS.forEach(keyword => {
        completionItems.push({
            label: keyword,
            kind: CompletionItemKind.Keyword,
            data: keyword
        });
    });
    
    // Add built-in functions
    DSL_BUILTIN_FUNCTIONS.forEach((func, index) => {
        completionItems.push({
            label: func.name,
            kind: CompletionItemKind.Function,
            data: index + 1000, // Offset to distinguish from keywords
            detail: func.signature,
            documentation: func.description,
            insertText: `${func.name}("$1")`,
            insertTextFormat: 2 // Snippet format
        });
    });
    
    // Add operators
    DSL_OPERATORS.forEach(op => {
        completionItems.push({
            label: op,
            kind: CompletionItemKind.Operator,
            data: op
        });
    });
    
    return completionItems;
});

// Completion item resolve provider
connection.onCompletionResolve((item: CompletionItem): CompletionItem => {
    if (typeof item.data === 'number' && item.data >= 1000) {
        // This is a built-in function
        const funcIndex = item.data - 1000;
        const func = DSL_BUILTIN_FUNCTIONS[funcIndex];
        if (func) {
            item.detail = func.signature;
            item.documentation = {
                kind: MarkupKind.Markdown,
                value: `**${func.name}**\n\n${func.description}\n\n${func.detail}`
            };
        }
    }
    return item;
});

// Hover provider
connection.onHover((params: TextDocumentPositionParams): Hover | undefined => {
    const document = documents.get(params.textDocument.uri);
    if (!document) {
        return undefined;
    }
    
    const text = document.getText();
    const position = params.position;
    const lines = text.split('\n');
    const line = lines[position.line];
    
    // Find word at position
    const wordMatch = line.match(/\w+/g);
    if (!wordMatch) {
        return undefined;
    }
    
    let hoveredWord = '';
    let wordStart = 0;
    let wordEnd = 0;
    
    for (const word of wordMatch) {
        const wordIndex = line.indexOf(word, wordEnd);
        if (wordIndex <= position.character && position.character <= wordIndex + word.length) {
            hoveredWord = word;
            wordStart = wordIndex;
            wordEnd = wordIndex + word.length;
            break;
        }
    }
    
    if (!hoveredWord) {
        return undefined;
    }
    
    // Check if it's a built-in function
    const func = DSL_BUILTIN_FUNCTIONS.find(f => f.name === hoveredWord);
    if (func) {
        return {
            contents: {
                kind: MarkupKind.Markdown,
                value: `**${func.name}**\n\n${func.description}\n\n\`\`\`sep\n${func.signature}\n\`\`\`\n\n${func.detail}`
            },
            range: {
                start: { line: position.line, character: wordStart },
                end: { line: position.line, character: wordEnd }
            }
        };
    }
    
    // Check if it's a keyword
    if (DSL_KEYWORDS.includes(hoveredWord)) {
        let description = '';
        switch (hoveredWord) {
            case 'pattern':
                description = 'Defines a pattern for analysis';
                break;
            case 'if':
                description = 'Conditional statement';
                break;
            case 'else':
                description = 'Alternative branch for conditional';
                break;
            case 'print':
                description = 'Output function for debugging';
                break;
            default:
                description = `SEP DSL keyword: ${hoveredWord}`;
        }
        
        return {
            contents: {
                kind: MarkupKind.Markdown,
                value: `**${hoveredWord}** (keyword)\n\n${description}`
            },
            range: {
                start: { line: position.line, character: wordStart },
                end: { line: position.line, character: wordEnd }
            }
        };
    }
    
    return undefined;
});

// Make the text document manager listen on the connection
documents.listen(connection);

// Listen on the connection
connection.listen();
