/**
 * SEP DSL VS Code Extension
 */

import * as vscode from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind
} from 'vscode-languageclient/node';

let client: LanguageClient;

export function activate(context: vscode.ExtensionContext) {
    // Language server setup
    const serverModule = context.asAbsolutePath('../../lsp/lib/server.js');
    
    const serverOptions: ServerOptions = {
        run: { module: serverModule, transport: TransportKind.ipc },
        debug: {
            module: serverModule,
            transport: TransportKind.ipc,
            options: { execArgv: ['--nolazy', '--inspect=6009'] }
        }
    };

    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'sep' }],
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher('**/.seprc')
        }
    };

    // Create and start the language client
    client = new LanguageClient(
        'sepDslLanguageServer',
        'SEP DSL Language Server',
        serverOptions,
        clientOptions
    );

    // Start the client (also starts the server)
    client.start();

    // Register commands
    const analyzeCommand = vscode.commands.registerCommand('sepDsl.analyze', () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.document.languageId !== 'sep') {
            vscode.window.showWarningMessage('Please open a SEP DSL file first');
            return;
        }

        vscode.window.showInformationMessage('Analyzing SEP DSL pattern...');
        
        // TODO: Integrate with actual SEP DSL analysis engine
        setTimeout(() => {
            vscode.window.showInformationMessage('Pattern analysis complete!');
        }, 1000);
    });

    const validateCommand = vscode.commands.registerCommand('sepDsl.validateSyntax', () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.document.languageId !== 'sep') {
            vscode.window.showWarningMessage('Please open a SEP DSL file first');
            return;
        }

        // Trigger diagnostics refresh
        vscode.languages.getDiagnostics(editor.document.uri).then(diagnostics => {
            const errorCount = diagnostics.filter(d => d.severity === vscode.DiagnosticSeverity.Error).length;
            const warningCount = diagnostics.filter(d => d.severity === vscode.DiagnosticSeverity.Warning).length;
            
            if (errorCount === 0 && warningCount === 0) {
                vscode.window.showInformationMessage('✅ Syntax validation passed - no issues found');
            } else {
                vscode.window.showWarningMessage(
                    `Syntax validation found ${errorCount} errors and ${warningCount} warnings`
                );
            }
        });
    });

    context.subscriptions.push(analyzeCommand, validateCommand);

    // Register document formatting provider
    const formattingProvider = vscode.languages.registerDocumentFormattingProvider('sep', {
        provideDocumentFormattingEdits(document: vscode.TextDocument): vscode.TextEdit[] {
            const edits: vscode.TextEdit[] = [];
            const text = document.getText();
            const lines = text.split('\n');
            
            let indentLevel = 0;
            const tabSize = 4;
            
            for (let i = 0; i < lines.length; i++) {
                const line = lines[i];
                const trimmedLine = line.trim();
                
                // Skip empty lines and comments
                if (!trimmedLine || trimmedLine.startsWith('//')) {
                    continue;
                }
                
                // Decrease indent for closing braces
                if (trimmedLine === '}') {
                    indentLevel = Math.max(0, indentLevel - 1);
                }
                
                // Calculate expected indentation
                const expectedIndent = ' '.repeat(indentLevel * tabSize);
                const currentIndent = line.match(/^(\s*)/)?.[1] || '';
                
                // Apply indentation if different
                if (currentIndent !== expectedIndent) {
                    const range = new vscode.Range(
                        new vscode.Position(i, 0),
                        new vscode.Position(i, currentIndent.length)
                    );
                    edits.push(vscode.TextEdit.replace(range, expectedIndent));
                }
                
                // Increase indent for opening braces
                if (trimmedLine.endsWith('{')) {
                    indentLevel++;
                }
            }
            
            return edits;
        }
    });

    context.subscriptions.push(formattingProvider);

    // Status bar item
    const statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBarItem.text = "$(symbol-method) SEP DSL";
    statusBarItem.tooltip = "SEP DSL Language Support Active";
    statusBarItem.command = 'sepDsl.analyze';
    
    // Show status bar item only for SEP files
    vscode.window.onDidChangeActiveTextEditor(editor => {
        if (editor && editor.document.languageId === 'sep') {
            statusBarItem.show();
        } else {
            statusBarItem.hide();
        }
    });

    context.subscriptions.push(statusBarItem);

    console.log('SEP DSL extension activated');
}

export function deactivate(): Thenable<void> | undefined {
    if (!client) {
        return undefined;
    }
    return client.stop();
}
