project('fast_mamba_inference', 'cpp', version: '0.1.0',
  default_options: ['cpp_std=c++20', 'warning_level=3']
)

# -D_GLIBCXX_USE_CXX11_ABI=0


add_project_arguments('-D_GLIBCXX_USE_CXX11_ABI=0', language : 'c')

vcpkg = find_program('vcpkg', required : true)
run_command(vcpkg, 'install', '--x-wait-for-lock',
                              '--x-manifest-root=' + meson.current_source_dir())

# TODO detect triplet and include @ x64-linux or arm64-osx

incdir = include_directories('vcpkg_installed/arm64-osx/include', 'vcpkg_installed/arm64-osx/include/torch/csrc/api/include')
sources = files(['src/main.cpp', 'src/util.cpp', 'src/SSModel.cpp', 'src/Tokenizer.cpp', 'src/Norm.cpp'])
yaml_dep = dependency('yaml-cpp', required: true)
torch_dep = dependency('libtorch', required: true)

executable('mamba', sources, include_directories: incdir, dependencies: [yaml_dep, torch_dep])
