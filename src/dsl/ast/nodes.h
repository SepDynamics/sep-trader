#pragma once
#include <string>
#include <vector>
#include <memory>
#include <variant>
#include <unordered_map>

// This is the grammar of our language.

namespace dsl::ast {

// Token types
enum class TokenType
{
    IDENTIFIER,
    NUMBER,
    STRING,
    BOOLEAN,
    PATTERN,
    STREAM,
    SIGNAL,
    MEMORY,
    FROM,
    WHEN,
    USING,
    INPUT,
    OUTPUT,
    EVOLVE,
    INHERITS,
    LBRACE,
    RBRACE,
    LPAREN,
    RPAREN,
    SEMICOLON,
    COLON,
    COMMA,
    DOT,
    ASSIGN,
    PLUS,
    MINUS,
    MULTIPLY,
    DIVIDE,
    GT,
    LT,
    GE,
    LE,
    EQ,
    NE,
    AND,
    OR,
    NOT,
    IF,
    ELSE,
    WHILE,
    FOR,
    FUNCTION,
    RETURN,
    BREAK,
    CONTINUE,
    IMPORT,
    EXPORT,
    ASYNC,
    AWAIT,
    TRY,
    CATCH,
    THROW,
    FINALLY,
    WEIGHTED_SUM,
    // Type annotation tokens
    NUMBER_TYPE,
    STRING_TYPE,
    BOOL_TYPE,
    PATTERN_TYPE,
    VOID_TYPE,
    // Array tokens
    LBRACKET,
    RBRACKET,
    EOF_TOKEN,
    INVALID
};

// Source location information
struct SourceLocation {
    size_t line = 0;
    size_t column = 0;
    
    SourceLocation() = default;
    SourceLocation(size_t l, size_t c) : line(l), column(c) {}
    
    std::string to_string() const {
        return std::to_string(line) + ":" + std::to_string(column);
    }
};

// Base class for all nodes
struct Node { 
    SourceLocation location;
    virtual ~Node() = default; 
};

// Type annotations
enum class TypeAnnotation {
    NUMBER,
    STRING,
    BOOL,
    PATTERN,
    VOID,
    ARRAY,    // Array of numbers
    INFERRED  // Type will be inferred
};

// Expressions
struct Expression : Node {};
struct Statement : Node {};

struct NumberLiteral : Expression { double value; };
struct StringLiteral : Expression { std::string value; };
struct BooleanLiteral : Expression { bool value; };
struct Identifier : Expression { std::string name; };

struct BinaryOp : Expression {
    std::unique_ptr<Expression> left;
    std::string op;
    std::unique_ptr<Expression> right;
};

struct UnaryOp : Expression {
    std::string op;
    std::unique_ptr<Expression> right;
};

struct Call : Expression {
    std::string callee;
    std::vector<std::unique_ptr<Expression>> args;
};

struct MemberAccess : Expression {
    std::unique_ptr<Expression> object;
    std::string member;
};

struct ArrayLiteral : Expression {
    std::vector<std::unique_ptr<Expression>> elements;
};

struct ArrayAccess : Expression {
    std::unique_ptr<Expression> array;
    std::unique_ptr<Expression> index;
};

struct WeightedSum : Expression {
    std::vector<std::pair<std::unique_ptr<Expression>, std::unique_ptr<Expression>>> pairs; // weight : value pairs
};

struct EvolveStatement : Statement {
    std::unique_ptr<Expression> condition;
    std::vector<std::unique_ptr<Statement>> body;
};

struct IfStatement : Statement
{
    std::unique_ptr<Expression> condition;
    std::vector<std::unique_ptr<Statement>> then_branch;
    std::vector<std::unique_ptr<Statement>> else_branch;
};

struct WhileStatement : Statement
{
    std::unique_ptr<Expression> condition;
    std::vector<std::unique_ptr<Statement>> body;
};

struct ReturnStatement : Statement
{
    std::unique_ptr<Expression> value;
};

struct FunctionDeclaration : Statement
{
    std::string name;
    std::vector<std::pair<std::string, TypeAnnotation>> parameters; // parameter name + type
    TypeAnnotation return_type = TypeAnnotation::INFERRED;
    std::vector<std::unique_ptr<Statement>> body;
};

struct ImportStatement : Statement
{
    std::string module_path;
    std::vector<std::string> imports; // empty = import all
};

struct ExportStatement : Statement
{
    std::vector<std::string> exports; // what to export from this module
};

struct AsyncFunctionDeclaration : Statement
{
    std::string name;
    std::vector<std::pair<std::string, TypeAnnotation>> parameters; // parameter name + type
    TypeAnnotation return_type = TypeAnnotation::INFERRED;
    std::vector<std::unique_ptr<Statement>> body;
};

struct AwaitExpression : Expression
{
    std::unique_ptr<Expression> expression;
};

struct TryStatement : Statement
{
    std::vector<std::unique_ptr<Statement>> try_body;
    std::vector<std::unique_ptr<Statement>> catch_body;
    std::string catch_variable; // variable name for caught exception
    std::vector<std::unique_ptr<Statement>> finally_body;
};

struct ThrowStatement : Statement
{
    std::unique_ptr<Expression> expression;
};

// Statements
struct Assignment : Statement {
    std::string name;
    TypeAnnotation type = TypeAnnotation::INFERRED;
    std::unique_ptr<Expression> value;
};

struct ExpressionStatement : Statement {
    std::unique_ptr<Expression> expression;
};

// High-Level Declarations
struct StreamDecl : Node {
    std::string name;
    std::string source;
    // We'll parse parameters into a simple map for now
    std::unordered_map<std::string, std::string> params;
};

struct MemoryDecl : Node
{
    std::string name;
    // TODO: Define memory block properties
};

struct PatternDecl : Node {
    std::string name;
    std::string parent_pattern;  // Name of the pattern being inherited from
    std::vector<std::string> inputs;
    std::vector<std::unique_ptr<Statement>> body;
};

struct SignalDecl : Node {
    std::string name;
    std::unique_ptr<Expression> trigger;
    std::unique_ptr<Expression> confidence;
    std::string action;
};

// The root of our program
struct Program : Node {
    std::vector<std::unique_ptr<StreamDecl>> streams;
    std::vector<std::unique_ptr<PatternDecl>> patterns;
    std::vector<std::unique_ptr<SignalDecl>> signals;
};

} // namespace dsl::ast
