require 'rubystats'

gen = Rubystats::NormalDistribution.new(500000, 200000)

n = gets.to_i

test_file = File.open('test.txt', 'w')

test_file.write("#{n}\n")

step = n / 10

for i in 0...n do

    if i % step == 0
        puts "#{i / step * 10}%..."
    end

    number = rand(1e+12)

    out = number

    if number % 2 == 0
        out *= -1
    end

    if number % 3 == 0
        out *= rand
    end

    if number % 5 == 0
        out *= rand(10) * rand
    end

    # if number % 7 == 0
    #     out /= rand(100)
    # end

    number = out

    # number = gen.rng
    test_file.write("#{number}")

    if (i != (n - 1))
        test_file.write(" ")
    end

end

test_file.write("\n")

test_file.close

puts 'done.'
