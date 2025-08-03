require 'mkmf'

# Check for the SEP library
unless have_library('sep', 'sep_create_interpreter')
  abort "libsep not found. Please build and install the SEP DSL engine first."
end

# Check for the header
unless have_header('sep/sep_c_api.h')
  abort "sep_c_api.h not found. Please install SEP development headers."
end

# Configure the extension
$CPPFLAGS += " -std=c++17"
$LIBS += " -lstdc++"

create_makefile('sep_dsl/sep_dsl')
