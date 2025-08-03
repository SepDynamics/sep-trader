Gem::Specification.new do |spec|
  spec.name          = "sep_dsl"
  spec.version       = "1.0.0"
  spec.authors       = ["SEP Engine Team"]
  spec.email         = ["info@sepengine.com"]

  spec.summary       = "AGI Coherence Framework DSL with CUDA acceleration"
  spec.description   = "Ruby bindings for the SEP DSL engine providing quantum pattern analysis and real-time signal generation"
  spec.homepage      = "https://github.com/scrallex/dsl"
  spec.license       = "MIT"
  
  spec.files         = Dir['lib/**/*', 'ext/**/*', 'README.md']
  spec.extensions    = ['ext/sep_dsl/extconf.rb']
  spec.require_paths = ["lib"]

  spec.add_development_dependency "bundler", "~> 2.0"
  spec.add_development_dependency "rake", "~> 13.0"
  spec.add_development_dependency "rake-compiler", "~> 1.0"
  spec.add_development_dependency "rspec", "~> 3.0"
end
