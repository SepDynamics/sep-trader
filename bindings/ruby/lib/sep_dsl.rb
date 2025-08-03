require 'sep_dsl/sep_dsl' # This loads the C extension

module SEP
  class Interpreter
    # The C extension defined initialize, execute, execute_file, get_variable, etc.
    # We can add more user-friendly Ruby methods here.
    
    def run_file(filepath)
      execute_file(filepath)
    end
    
    def run(script_source)
      execute(script_source)
    end
    
    # Allow hash-like access to variables
    alias :[] :get_variable
    
    # Execute a pattern and return its result
    def run_pattern(pattern_name, &block)
      if block_given?
        # If a block is given, execute it as DSL code
        script = "pattern #{pattern_name} {\n"
        script += yield
        script += "\n}"
        execute(script)
      else
        # Just return the pattern's variables
        get_variable(pattern_name)
      end
    end
    
    # Convenience method for quantum analysis
    def analyze_pattern(data)
      script = <<~SCRIPT
        pattern analysis {
          bits = extract_bits("#{data}")
          coherence = measure_coherence("#{data}")
          entropy = measure_entropy("#{data}")
        }
      SCRIPT
      
      execute(script)
      
      {
        coherence: get_variable("analysis.coherence"),
        entropy: get_variable("analysis.entropy"),
        bits: get_variable("analysis.bits")
      }
    end
  end
  
  # Convenience method for quick pattern analysis
  def self.analyze(data)
    interpreter = Interpreter.new
    interpreter.analyze_pattern(data)
  end
  
  # Quick script execution
  def self.run(script)
    interpreter = Interpreter.new
    interpreter.execute(script)
    interpreter
  end
end
