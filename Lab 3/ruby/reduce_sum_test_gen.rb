n = gets.to_i
sum = 0
numbers = []
n.times do
    val = rand(10)
    sum += val
    numbers.push(val)
end
file = File.open("reduce_sum_#{sum}.txt", 'w')
file.write("#{n}\n")
numbers.each do |val|
    file.write("#{val} ")
end
file.close
