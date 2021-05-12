GXX=nvcc
FLAGS=-lineinfo

analyze_test_mean: test_mean
	cuda-memcheck ./test_mean
	nsys profile --stats=true ./test_mean
	# nvprof --analysis-metrics -f -o analysis.nvprof ./test_mean #--benchmark
	# nvvp -vm /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java

test_mean: mean.o test_mean.o
	$(GXX) -o test_mean mean.o test_mean.o

test_mean.o: mean.h
	$(GXX) -c test_mean.cu $(FLAGS)

mean.o: mean.h mean.cu
	$(GXX) -c mean.cu $(FLAGS)

clean:
	rm -rf *.o test_mean *.sqlite *.qdrep *.nvprof
