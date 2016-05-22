n = gets.to_i
max = -100000000000
numbers = []
n.times do
    val = rand(1000000)
    if val.even?
        val *= -1
    end
    if val > max
        max = val
    end
    numbers.push(val)
end
file = File.open("reduce_max_#{max}.txt", 'w')
file.write("#{n}\n")
numbers.each do |val|
    file.write("#{val} ")
end
file.close
