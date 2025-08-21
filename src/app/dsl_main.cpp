#include "core/standard_includes.h"
#include "core/result_types.h"
#include "core/types.h"

#include "util/serializer.h"
#include "core/facade.h"
#include "util/lexer.h"
#include "util/parser.h"
#include "util/interpreter.h"



void print_usage(const char* program_name) {

    std::cerr << "Usage: " << program_name << " [options] <script.sep>" << std::endl;

    std::cerr << "Options:" << std::endl;

    std::cerr << "  --save-ast <filename>  Save parsed AST to JSON file" << std::endl;

    std::cerr << "  --load-ast <filename>  Load and execute pre-parsed AST from JSON file" << std::endl;

    std::cerr << "  --help                 Show this help message" << std::endl;

}



int main(int argc, char* argv[]) {

    std::string script_file;

    std::string save_ast_file;

    std::string load_ast_file;

    bool help_requested = false;

    

    // Parse command line arguments

    for (int i = 1; i < argc; i++) {

        std::string arg = argv[i];

        

        if (arg == "--help") {

            help_requested = true;

        } else if (arg == "--save-ast") {

            if (i + 1 >= argc) {

                std::cerr << "Error: --save-ast requires a filename" << std::endl;

                return 1;

            }

            save_ast_file = argv[++i];

        } else if (arg == "--load-ast") {

            if (i + 1 >= argc) {

                std::cerr << "Error: --load-ast requires a filename" << std::endl;

                return 1;

            }

            load_ast_file = argv[++i];

        } else if (arg.substr(0, 2) == "--") {

            std::cerr << "Error: Unknown option " << arg << std::endl;

            print_usage(argv[0]);

            return 1;

        } else {

            if (!script_file.empty()) {

                std::cerr << "Error: Multiple script files specified" << std::endl;

                print_usage(argv[0]);

                return 1;

            }

            script_file = arg;

        }

    }

    

    if (help_requested) {

        print_usage(argv[0]);

        return 0;

    }

    

    // Validate arguments

    if (!load_ast_file.empty() && !script_file.empty()) {

        std::cerr << "Error: Cannot specify both script file and --load-ast" << std::endl;

        print_usage(argv[0]);

        return 1;

    }

    

    if (load_ast_file.empty() && script_file.empty()) {

        std::cerr << "Error: Must specify either a script file or --load-ast" << std::endl;

        print_usage(argv[0]);

        return 1;

    }

    

    std::unique_ptr<dsl::ast::Program> program;

    

    try {

        std::cout << "=== SEP DSL Interpreter ===" << std::endl;

        

        // Handle AST loading vs normal parsing

        if (!load_ast_file.empty()) {

            std::cout << "Loading AST from: " << load_ast_file << std::endl;

            program = dsl::ast::ASTSerializer::load_from_file(load_ast_file);

            if (!program) {

                std::cerr << "Error: Failed to load AST from file" << std::endl;

                return 1;

            }

            std::cout << "AST loaded successfully!" << std::endl;

        } else {

            // Read and parse the DSL script file

            std::ifstream file(script_file);

            if (!file.is_open()) {

                std::cerr << "Error: Could not open file " << script_file << std::endl;

                return 1;

            }

            

            std::stringstream buffer;

            buffer << file.rdbuf();

            std::string source = buffer.str();

            file.close();

            

            std::cout << "Source file: " << script_file << std::endl;

            std::cout << "Source content:" << std::endl;

            std::cout << source << std::endl;

            std::cout << "========================" << std::endl;

            

            // Create parser and parse the source

            dsl::parser::Parser parser(source);

            program = parser.parse();

            

            std::cout << "Parsing completed successfully!" << std::endl;

            

            // Save AST if requested

            if (!save_ast_file.empty()) {

                std::cout << "Saving AST to: " << save_ast_file << std::endl;

                if (dsl::ast::ASTSerializer::save_to_file(*program, save_ast_file)) {

                    std::cout << "AST saved successfully!" << std::endl;

                } else {

                    std::cerr << "Warning: Failed to save AST to file" << std::endl;

                }

            }

        }

        

        std::cout << "Program contains:" << std::endl;

        std::cout << "  - " << program->streams.size() << " stream(s)" << std::endl;

        std::cout << "  - " << program->patterns.size() << " pattern(s)" << std::endl;

        std::cout << "  - " << program->signals.size() << " signal(s)" << std::endl;

        std::cout << "========================" << std::endl;

        

        // Initialize the engine facade before creating interpreter

        auto& engine = sep::engine::EngineFacade::getInstance();

        auto init_result = engine.initialize();

        if (!init_result.isSuccess()) {

            std::cerr << "Failed to initialize engine facade" << std::endl;

            return 1;

        }

        

        // Create interpreter and execute the program

        dsl::runtime::Interpreter interpreter;

        std::cout << "Starting interpretation:" << std::endl;

        interpreter.interpret(*program);

        

        std::cout << "========================" << std::endl;

        std::cout << "Interpretation completed!" << std::endl;

        

    } catch (const std::exception& e) {

        std::cerr << "Error: " << e.what() << std::endl;

        return 1;

    }

    

    return 0;

}


