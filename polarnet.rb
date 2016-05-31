N = 40
LEARNING_RATE = 0.01
BIAS = 2.0

def main
  synone = Array.new(3) { Array.new(N) { rand * 0.1 } }
  syntwo = Array.new(N) { Array.new(2) { rand * 0.1 } }

  count = 0

  loop do
    point = Complex.polar(rand, rand * 2 * Math::PI)
    input = point.polar + [BIAS]
    target = point.rectangular

    medin = []
    medout = []
    (0...N).each do |i| # each medial neuron
      medin[i] = (0..2).map { |j| synone[j][i] * input[j] }.inject(:+)
      medout[i] = activation_function(medin[i])
    end

    output = []
    error = []
    (0..1).each do |i| # each output neuron
      output[i] = (0...N).map { |j| syntwo[j][i] * medout[j] }.inject(:+)
      error[i] = target[i] - output[i]
    end

    if (count % 1000).zero?
      puts "#{count}: #{Math.sqrt(error.map { |e| e ** 2 }.inject(:+))}"
    end
    count += 1

    # adjust second layer
    (0..1).each do |i| # each output neuron
      (0...N).each do |j| # each medial neuron
        syntwo[j][i] += LEARNING_RATE * medout[j] * error[i]
      end
    end

    sigma = []
    sigmoid = []
    # derive sigmoidal signal
    (0...N).each do |i| # each medial neuron
      sigma[i] = 0
      (0..1).each do |j| # each output neuron
        sigma[i] += error[j] * syntwo[i][j]
      end
      sigmoid[i] = derivative_of_activation_function(medin[i])
    end

    # adjust first layer
    (0..2).each do |i| # each input neuron
      (0...N).each do |j| # each medial neuron
        delta = LEARNING_RATE * sigmoid[j] * sigma[j] * input[i]
        synone[i][j] += delta
      end
    end
  end
end

def activation_function(x)
 Math.tanh(x)
end

def derivative_of_activation_function(x)
  1 - (Math.tanh(x) ** 2)
end

main
