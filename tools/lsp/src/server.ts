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
    Range,
    Location,
    Definition,
    DocumentSymbol,
    SymbolKind,
    SymbolInformation,
    PrepareRenameParams,
    RenameParams,
    WorkspaceEdit,
    TextEdit
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
            documentSymbolProvider: true,
            renameProvider: {
                prepareProvider: true
            }
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
    'pattern', 'if', 'else', 'print', 'true', 'false', 'function', 'return',
    'for', 'while', 'break', 'continue', 'import', 'export', 'async', 'await',
    'try', 'catch', 'finally', 'throw'
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

// Go-to-definition provider
connection.onDefinition((params: TextDocumentPositionParams): Definition | undefined => {
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
    
    for (const word of wordMatch) {
        const wordIndex = line.indexOf(word, wordStart);
        if (wordIndex <= position.character && position.character <= wordIndex + word.length) {
            hoveredWord = word;
            wordStart = wordIndex;
            break;
        }
        wordStart = wordIndex + word.length;
    }
    
    if (!hoveredWord) {
        return undefined;
    }
    
    // Search for pattern definitions
    const patternRegex = new RegExp(`pattern\\s+${hoveredWord}\\s*\\{`, 'g');
    const allLines = text.split('\n');
    
    for (let lineIndex = 0; lineIndex < allLines.length; lineIndex++) {
        const currentLine = allLines[lineIndex];
        const match = patternRegex.exec(currentLine);
        
        if (match) {
            const patternStart = currentLine.indexOf(hoveredWord);
            return {
                uri: params.textDocument.uri,
                range: {
                    start: { line: lineIndex, character: patternStart },
                    end: { line: lineIndex, character: patternStart + hoveredWord.length }
                }
            };
        }
    }
    
    // Search for variable assignments
    const variableRegex = new RegExp(`\\b${hoveredWord}\\s*=`, 'g');
    
    for (let lineIndex = 0; lineIndex < allLines.length; lineIndex++) {
        const currentLine = allLines[lineIndex];
        const match = variableRegex.exec(currentLine);
        
        if (match) {
            const varStart = currentLine.indexOf(hoveredWord);
            return {
                uri: params.textDocument.uri,
                range: {
                    start: { line: lineIndex, character: varStart },
                    end: { line: lineIndex, character: varStart + hoveredWord.length }
                }
            };
        }
    }
    
    // Search for function definitions (user-defined functions)
    const functionRegex = new RegExp(`function\\s+${hoveredWord}\\s*\\(`, 'g');
    
    for (let lineIndex = 0; lineIndex < allLines.length; lineIndex++) {
        const currentLine = allLines[lineIndex];
        const match = functionRegex.exec(currentLine);
        
        if (match) {
            const funcStart = currentLine.indexOf(hoveredWord);
            return {
                uri: params.textDocument.uri,
                range: {
                    start: { line: lineIndex, character: funcStart },
                    end: { line: lineIndex, character: funcStart + hoveredWord.length }
                }
            };
        }
    }
    
    return undefined;
});

// Document symbol provider
connection.onDocumentSymbol((params): DocumentSymbol[] => {
    const document = documents.get(params.textDocument.uri);
    if (!document) {
        return [];
    }
    
    const text = document.getText();
    const lines = text.split('\n');
    const symbols: DocumentSymbol[] = [];
    
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        
        // Find pattern definitions
        const patternMatch = line.match(/pattern\s+(\w+)\s*\{/);
        if (patternMatch) {
            const patternName = patternMatch[1];
            const startChar = lines[i].indexOf(patternName);
            
            // Find the end of the pattern (closing brace)
            let braceCount = 1;
            let endLine = i;
            let endChar = lines[i].length;
            
            for (let j = i + 1; j < lines.length && braceCount > 0; j++) {
                const currentLine = lines[j];
                for (let k = 0; k < currentLine.length; k++) {
                    if (currentLine[k] === '{') braceCount++;
                    if (currentLine[k] === '}') braceCount--;
                    if (braceCount === 0) {
                        endLine = j;
                        endChar = k + 1;
                        break;
                    }
                }
            }
            
            symbols.push({
                name: patternName,
                kind: SymbolKind.Class,
                range: {
                    start: { line: i, character: 0 },
                    end: { line: endLine, character: endChar }
                },
                selectionRange: {
                    start: { line: i, character: startChar },
                    end: { line: i, character: startChar + patternName.length }
                },
                children: []
            });
        }
        
        // Find function definitions
        const functionMatch = line.match(/function\s+(\w+)\s*\(/);
        if (functionMatch) {
            const funcName = functionMatch[1];
            const startChar = lines[i].indexOf(funcName);
            
            // Find the end of the function (closing brace)
            let braceCount = 0;
            let endLine = i;
            let endChar = lines[i].length;
            let foundOpenBrace = false;
            
            // Look for opening brace
            for (let j = i; j < lines.length; j++) {
                const currentLine = lines[j];
                for (let k = 0; k < currentLine.length; k++) {
                    if (currentLine[k] === '{') {
                        if (!foundOpenBrace) {
                            foundOpenBrace = true;
                            braceCount = 1;
                        } else {
                            braceCount++;
                        }
                    } else if (currentLine[k] === '}' && foundOpenBrace) {
                        braceCount--;
                        if (braceCount === 0) {
                            endLine = j;
                            endChar = k + 1;
                            break;
                        }
                    }
                }
                if (braceCount === 0 && foundOpenBrace) break;
            }
            
            symbols.push({
                name: funcName,
                kind: SymbolKind.Function,
                range: {
                    start: { line: i, character: 0 },
                    end: { line: endLine, character: endChar }
                },
                selectionRange: {
                    start: { line: i, character: startChar },
                    end: { line: i, character: startChar + funcName.length }
                },
                children: []
            });
        }
        
        // Find variable assignments within patterns/functions
        const variableMatch = line.match(/(\w+)\s*=/);
        if (variableMatch && !line.includes('pattern') && !line.includes('function')) {
            const varName = variableMatch[1];
            const startChar = lines[i].indexOf(varName);
            
            symbols.push({
                name: varName,
                kind: SymbolKind.Variable,
                range: {
                    start: { line: i, character: startChar },
                    end: { line: i, character: startChar + varName.length }
                },
                selectionRange: {
                    start: { line: i, character: startChar },
                    end: { line: i, character: startChar + varName.length }
                },
                children: []
            });
        }
    }
    
    return symbols;
});

// Prepare rename provider
connection.onPrepareRename((params: PrepareRenameParams) => {
    const document = documents.get(params.textDocument.uri);
    if (!document) {
        return null;
    }
    
    const text = document.getText();
    const position = params.position;
    const lines = text.split('\n');
    const line = lines[position.line];
    
    // Find word at position
    const wordMatch = line.match(/\w+/g);
    if (!wordMatch) {
        return null;
    }
    
    let hoveredWord = '';
    let wordStart = 0;
    
    for (const word of wordMatch) {
        const wordIndex = line.indexOf(word, wordStart);
        if (wordIndex <= position.character && position.character <= wordIndex + word.length) {
            hoveredWord = word;
            wordStart = wordIndex;
            break;
        }
        wordStart = wordIndex + word.length;
    }
    
    if (!hoveredWord) {
        return null;
    }
    
    // Check if it's a renameable symbol (pattern, variable, or user function)
    const allLines = text.split('\n');
    
    // Check if it's a pattern definition
    const patternRegex = new RegExp(`pattern\\s+${hoveredWord}\\s*\\{`, 'g');
    for (const textLine of allLines) {
        if (patternRegex.test(textLine)) {
            return {
                start: { line: position.line, character: wordStart },
                end: { line: position.line, character: wordStart + hoveredWord.length }
            };
        }
    }
    
    // Check if it's a function definition
    const functionRegex = new RegExp(`function\\s+${hoveredWord}\\s*\\(`, 'g');
    for (const textLine of allLines) {
        if (functionRegex.test(textLine)) {
            return {
                start: { line: position.line, character: wordStart },
                end: { line: position.line, character: wordStart + hoveredWord.length }
            };
        }
    }
    
    // Check if it's a variable
    const variableRegex = new RegExp(`\\b${hoveredWord}\\s*=`, 'g');
    for (const textLine of allLines) {
        if (variableRegex.test(textLine)) {
            return {
                start: { line: position.line, character: wordStart },
                end: { line: position.line, character: wordStart + hoveredWord.length }
            };
        }
    }
    
    // Don't allow renaming built-in functions or keywords
    const isBuiltin = DSL_BUILTIN_FUNCTIONS.some(f => f.name === hoveredWord);
    const isKeyword = DSL_KEYWORDS.includes(hoveredWord);
    
    if (isBuiltin || isKeyword) {
        return null;
    }
    
    return null;
});

// Rename provider
connection.onRenameRequest((params: RenameParams): WorkspaceEdit | null => {
    const document = documents.get(params.textDocument.uri);
    if (!document) {
        return null;
    }
    
    const text = document.getText();
    const position = params.position;
    const newName = params.newName;
    const lines = text.split('\n');
    const line = lines[position.line];
    
    // Find word at position
    const wordMatch = line.match(/\w+/g);
    if (!wordMatch) {
        return null;
    }
    
    let oldName = '';
    let wordStart = 0;
    
    for (const word of wordMatch) {
        const wordIndex = line.indexOf(word, wordStart);
        if (wordIndex <= position.character && position.character <= wordIndex + word.length) {
            oldName = word;
            wordStart = wordIndex;
            break;
        }
        wordStart = wordIndex + word.length;
    }
    
    if (!oldName) {
        return null;
    }
    
    // Validate new name
    if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(newName)) {
        return null; // Invalid identifier
    }
    
    // Check if new name conflicts with built-ins or keywords
    const isBuiltinConflict = DSL_BUILTIN_FUNCTIONS.some(f => f.name === newName);
    const isKeywordConflict = DSL_KEYWORDS.includes(newName);
    
    if (isBuiltinConflict || isKeywordConflict) {
        return null; // Name conflict
    }
    
    const allLines = text.split('\n');
    const edits: TextEdit[] = [];
    
    // Find all occurrences of the symbol
    for (let lineIndex = 0; lineIndex < allLines.length; lineIndex++) {
        const currentLine = allLines[lineIndex];
        
        // Use word boundary regex to find exact matches
        const regex = new RegExp(`\\b${oldName}\\b`, 'g');
        let match;
        
        while ((match = regex.exec(currentLine)) !== null) {
            // Check context to ensure it's the same symbol type
            const beforeChar = match.index > 0 ? currentLine[match.index - 1] : ' ';
            const afterChar = match.index + oldName.length < currentLine.length ? 
                currentLine[match.index + oldName.length] : ' ';
            
            // Skip if it's part of a larger identifier
            if (/\w/.test(beforeChar) || /\w/.test(afterChar)) {
                continue;
            }
            
            edits.push({
                range: {
                    start: { line: lineIndex, character: match.index },
                    end: { line: lineIndex, character: match.index + oldName.length }
                },
                newText: newName
            });
        }
    }
    
    if (edits.length === 0) {
        return null;
    }
    
    return {
        changes: {
            [params.textDocument.uri]: edits
        }
    };
});

// Helper function to find all symbol references
function findSymbolReferences(text: string, symbolName: string, symbolType: 'pattern' | 'function' | 'variable'): Location[] {
    const lines = text.split('\n');
    const references: Location[] = [];
    
    for (let lineIndex = 0; lineIndex < lines.length; lineIndex++) {
        const line = lines[lineIndex];
        
        // Different regex patterns based on symbol type
        let regex: RegExp;
        switch (symbolType) {
            case 'pattern':
                regex = new RegExp(`\\b${symbolName}\\b`, 'g');
                break;
            case 'function':
                regex = new RegExp(`\\b${symbolName}\\s*\\(`, 'g');
                break;
            case 'variable':
                regex = new RegExp(`\\b${symbolName}\\b`, 'g');
                break;
        }
        
        let match;
        while ((match = regex.exec(line)) !== null) {
            references.push({
                uri: '', // Will be filled by caller
                range: {
                    start: { line: lineIndex, character: match.index },
                    end: { line: lineIndex, character: match.index + symbolName.length }
                }
            });
        }
    }
    
    return references;
}

// Make the text document manager listen on the connection
documents.listen(connection);

// Listen on the connection
connection.listen();
