require 'cunn'
require 'optim'

-- Criterion
criterion = nn.ClassNLLCriterion();
criterion.sizeAverage = false
criterion.ignoreIndex = 255
criterion:cuda()

-- Network
net = dofile('./FCN/AlexNet_FCN_nn.lua')
net:cuda()
params, gradParams = net:getParameters();

-- Training data (conversion to CudaTensors is done online for each minibatch )
trainData, testData, numClasses = dofile('../../data/loadDataset.lua')
shuffledIndices = torch.randperm(trainData.data:size()[1], 'torch.LongTensor')
trainData.data = trainData.data:cuda()
trainData.labels = trainData.labels:cuda()

-- Allocating GPU memory
output = torch.CudaTensor(numClasses, trainData.labels[1]:size()[1], trainData.labels[1]:size()[2]):fill(0)
gradOutput = torch.CudaTensor(output:size()):fill(0)

-- Optimizer configuration
config = {
  learningRate = 0.001,
  weightDecay = 0.0016,
  momentum = 0.99
}
maxIteration = 25


-- Begin training
print("# StochasticGradient: training")

iteration = 1

while true do
  local currentError = 0

  print('Epoch: ' .. iteration)

  for t = 1,trainData:size()[1] do
    gradParams:zero();
    output:fill(0)
    gradOutput:fill(0)

    function feval(params)
      local input = trainData.data[shuffledIndices[t]];
      local target = trainData.labels[shuffledIndices[t]];

      output = net:forward(input)

      for i = 1,output:size()[2] do
        for j = 1,output:size()[3] do
          criterion.output:fill(0)
          criterion.gradInput:fill(0)

          outputSelect = output:select(2,i):select(2,j)
          gradSelect = gradOutput:select(2,i):select(2,j)

          currentError = currentError + criterion:forward(outputSelect, target[i][j])
          gradSelect[{{}}] = criterion:updateGradInput(outputSelect, target[i][j])
        end
      end

      net:backward(input, gradOutput)

      return currentError, gradParams
    end

    optim.sgd(feval, params, config)
  end

  currentError = currentError / dataset:size()
  print('Current average error: ' .. currentError)

  iteration = iteration + 1

  if maxIteration > 0 and iteration > maxIteration then
    print("# StochasticGradient: you have reached the maximum number of iterations")
    print("# training error = " .. currentError)
    break
  end
end
