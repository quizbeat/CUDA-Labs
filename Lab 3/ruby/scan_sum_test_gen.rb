n = gets.to_i
sum = n - 1
numbers = Array.new(n, 1)
file = File.open("scan_sum_#{sum}.txt", 'w')
file.write("#{n}\n")
numbers.each do |val|
    file.write("#{val} ")
end
file.close
