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
    WEIGHTED_SUM,
    EOF_TOKEN,
    INVALID
};

// Base class for all nodes
struct Node { virtual ~Node() = default; };

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
    std::vector<std::string> parameters;
    std::vector<std::unique_ptr<Statement>> body;
};

// Statements
struct Assignment : Statement {
    std::string name;
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
