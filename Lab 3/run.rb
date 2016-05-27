# compile .cu file
compile_output = %x( nvcc -o exec/cuda.out src/lab3.cu )
if !compile_output.empty?
    puts 'Compilation error'
    exi 
end

# run with test
run_output = %x( ./exec/cuda.out < tests/test5 )
puts run_output
