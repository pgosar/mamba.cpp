project('fast_mamba_inference', 'cpp', version: '0.1.0',
  default_options: ['cpp_std=c++20', 'warning_level=3'])

cc = meson.get_compiler('cpp')
m_dep = cc.find_library('m', required : false)

sources = files(['src/mamba.cpp'])
executable('mamba', sources)
