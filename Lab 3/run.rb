# compile .cu file
compile_output = %x( nvcc -o exec/release.out src/lab3.cu )
if !compile_output.empty?
    puts compile_output
    exit
end

# run with test
run_output = %x( ./exec/release.out < tests/test5 )
puts run_output
