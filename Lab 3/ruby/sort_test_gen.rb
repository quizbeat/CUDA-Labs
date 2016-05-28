n = gets.to_i

numbers = []

n.times do
    val = rand(1000000)

    if val.even?
        val *= -1
    end

    val = val.to_f

    if rand(1321424) % 2 == 0
        val /= (rand(423) + 1)
    end

    numbers.push(val)
end

file = File.open("sort_#{n}.txt", 'w')
file.write("#{n}\n")
numbers.each do |val|
    file.write("#{val} ")
end

file.close
