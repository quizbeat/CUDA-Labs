# compile .cu file
compile_output = %x( nvcc -o exec/cuda.out src/lab3.cu )
if !compile_output.empty?
    puts 'Compilation error'
    exit
end

# run with test
run_output = %x( ./exec/cuda.out < tests/scan_sum_149.txt )
puts run_output
