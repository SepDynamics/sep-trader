# SEP DSL Grammar Specification

## Formal Grammar

```bnf
Program         ::= Declaration*

Declaration     ::= StreamDecl | PatternDecl | SignalDecl

StreamDecl      ::= "stream" IDENTIFIER "from" STRING

PatternDecl     ::= "pattern" IDENTIFIER "{" 
                    "input:" IDENTIFIER
                    Statement*
                    "}"

SignalDecl      ::= "signal" IDENTIFIER "{"
                    "trigger:" Expression
                    ("confidence:" Expression)?
                    "action:" IDENTIFIER
                    "}"

Statement       ::= Assignment | ExpressionStatement

Assignment      ::= IDENTIFIER "=" Expression

ExpressionStatement ::= Expression

Expression      ::= LogicalOr

LogicalOr       ::= LogicalAnd ("||" LogicalAnd)*

LogicalAnd      ::= Equality ("&&" Equality)*

Equality        ::= Comparison (("==" | "!=") Comparison)*

Comparison      ::= Term ((">" | "<" | ">=" | "<=") Term)*

Term            ::= Factor (("+" | "-") Factor)*

Factor          ::= Unary (("*" | "/") Unary)*

Unary           ::= ("!" | "-")? Call

Call            ::= Primary ("(" ArgumentList? ")")*

Primary         ::= NUMBER | STRING | IDENTIFIER | MemberAccess
                  | "(" Expression ")"

MemberAccess    ::= IDENTIFIER "." IDENTIFIER

ArgumentList    ::= Expression ("," Expression)*
```

## Token Types

```cpp
enum class TokenType {
    // Literals
    IDENTIFIER, NUMBER, STRING,
    
    // Keywords
    PATTERN, STREAM, SIGNAL, MEMORY,
    FROM, WHEN, USING, INPUT, OUTPUT,
    
    // Punctuation
    LBRACE, RBRACE, LPAREN, RPAREN,
    SEMICOLON, COLON, COMMA, DOT,
    
    // Operators
    ASSIGN, PLUS, MINUS, MULTIPLY, DIVIDE,
    GT, LT, GE, LE, EQ, NE,
    AND, OR, NOT,
    
    // Control Flow
    IF, ELSE, WHILE, FOR,
    RETURN, BREAK, CONTINUE,
    
    // Special
    EOF_TOKEN, INVALID
};
```

## Semantic Rules

### Variable Scoping
- Patterns create isolated environments
- Pattern variables accessible via `pattern.variable` syntax
- Stream identifiers global across program
- Signal triggers evaluated in global environment with pattern access

### Type System
- Dynamic typing with runtime type checking
- Supported types: `double`, `string`, `bool`, `PatternResult`
- Automatic type coercion for arithmetic operations

### Built-in Functions
- `qfh_analyze(data)` → `double`: Quantum field harmonics analysis
- `measure_coherence(data)` → `double`: Coherence measurement  
- `measure_entropy(data)` → `double`: Entropy calculation
- `extract_bits(data)` → `string`: Binary pattern extraction
