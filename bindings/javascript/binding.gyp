{
  "targets": [
    {
      "target_name": "sep_dsl_native",
      "sources": [
        "src/sep_dsl_node.cpp"
      ],
      "include_dirs": [
        "<!(node -e \"require('nan')\")",
        "../../src/c_api",
        "../../commercial_package/headers"
      ],
      "library_dirs": [
        "../../build/lib"
      ],
      "libraries": [
        "-lsep"
      ],
      "cflags_cc": [
        "-std=c++17",
        "-fexceptions",
        "-frtti"
      ],
      "conditions": [
        ["OS=='linux'", {
          "cflags_cc": [
            "-std=c++17",
            "-fPIC"
          ]
        }],
        ["OS=='mac'", {
          "xcode_settings": {
            "CLANG_CXX_LANGUAGE_STANDARD": "c++17",
            "CLANG_CXX_LIBRARY": "libc++",
            "MACOSX_DEPLOYMENT_TARGET": "10.15"
          }
        }],
        ["OS=='win'", {
          "msvs_settings": {
            "VCCLCompilerTool": {
              "ExceptionHandling": 1,
              "AdditionalOptions": ["/std:c++17"]
            }
          }
        }]
      ]
    }
  ]
}
