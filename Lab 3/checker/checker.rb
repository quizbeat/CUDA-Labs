test_amount = gets.to_i

for t in 1..test_amount

    data_size = rand(2000000)

    puts "Running test #{t} with #{data_size} elements..."

    data = []

    data_size.times do
        val = rand(1000000)
        val *= -1 if val.even?
        val = val.to_f
        val /= (rand(423) + 1) if rand(1321424) % 2 == 0
        data.push(val)
    end

    file_name = "sort_test_#{t}_size_#{data_size}.txt"
    file = File.open(file_name, 'w')
    file.write("#{data_size}\n")
    data.each do |val|
        file.write("#{val} ")
    end

    file.close

    cmd = "./release.out < #{file_name}"
    run_output = `#{cmd}`

    sorted_data = run_output.split(' ').map(&:to_f)

    if sorted_data.size != data_size
        puts " Status: Error, data size and sorted data size not equal"
        puts " Test file saved as: #{file_name}"
        exit
    end

    ok = true

    for i in 1...sorted_data.size
        if sorted_data[i] < sorted_data[i - 1]
            ok = false
            break
        end
    end

    if (ok)
        puts ' Status: OK'
    else
        puts ' Status: WA'
        puts " Test file saved as: #{file_name}"
        exit
    end

    File.delete(file_name)

end
