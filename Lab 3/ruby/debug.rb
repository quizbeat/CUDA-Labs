# compile .cu file
compile_output = %x( nvcc -DDEBUG -o exec/debug.out src/lab3_clean.cu )
if !compile_output.empty?
    puts compile_output
    exit
end

# run with test
run_output = %x( ./exec/debug.out < tests/sort_150.txt )
puts run_output
