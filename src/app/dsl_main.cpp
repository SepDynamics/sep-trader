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
        // ... existing code ...
    } catch (const std::exception& e) {
        std::cerr << "[DSL Main] Fatal error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "[DSL Main] Unknown fatal error" << std::endl;
        return 1;
    }

    

    return 0;

}


