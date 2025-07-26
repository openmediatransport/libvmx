mkdir arm64
g++ -O3 -std=c++17 -fdeclspec -fPIC -target arm64-apple-darwin -dynamiclib ../src/vmxcodec_arm.cpp ../src/vmxcodec.cpp -o arm64/libvmx.dylib
install_name_tool -id @rpath/libvmx.dylib arm64/libvmx.dylib
mkdir x86
g++ -O3 -std=c++17 -fdeclspec -fPIC -target x86_64-apple-darwin -mlzcnt -mavx2 -mbmi -dynamiclib ../src/vmxcodec_x86.cpp ../src/vmxcodec_avx2.cpp ../src/vmxcodec.cpp -o x86/libvmx.dylib
install_name_tool -id @rpath/libvmx.dylib x86/libvmx.dylib
lipo -create -output libvmx.dylib x86/libvmx.dylib arm64/libvmx.dylib
install_name_tool -id @rpath/libvmx.dylib libvmx.dylib