project('fast_mamba_inference', 'cpp', version: '0.1.0',
  default_options: ['cpp_std=c++20', 'warning_level=3']
)

vcpkg = find_program('vcpkg', required : true)
run_command(vcpkg, 'install', '--x-wait-for-lock',
                              '--x-manifest-root=' + meson.current_source_dir())

incdir = include_directories('vcpkg_installed/x64-linux/include')
sources = files(['src/main.cpp', 'src/util.cpp'])

executable('mamba', sources, include_directories: incdir)
