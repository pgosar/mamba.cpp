project('fast-mamba-inference', 'cpp',
        version : '1.0.0',
        default_options : ['warning_level=3', 'cpp_std=c++20'])

fast_mamba_inference = executable('fast_mamba_inference', 'src/mamba.cpp', install : true)

test('test', fast_mamba_inference)
