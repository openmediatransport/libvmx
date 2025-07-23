mkdir x86
g++ -O3 -std=c++17 -fdeclspec -fPIC -target x86_64-apple-darwin -mlzcnt -mavx2 -mbmi -dynamiclib ../src/vmxcodec_x86.cpp ../src/vmxcodec_avx2.cpp ../src/vmxcodec.cpp -o x86/libvmx.dylib