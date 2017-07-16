require 'cunn'
require 'optim'

-- Criterion
criterion = nn.ClassNLLCriterion();
criterion.sizeAverage = false
criterion.ignoreIndex = 22
criterion:cuda()

-- Network
net = dofile('./FCN/AlexNet_FCN_nn.lua')
net:cuda()
params, gradParams = net:getParameters();

-- Optimizer configuration
config = {
  learningRate = 0.001,
  weightDecay = 0.0016,
  momentum = 0.99
}
maxIteration = 25

-- Allocating re-used GPU memory
numClasses = 21
inputWidth = 500
inputHeight = 500
output = torch.CudaTensor(numClasses, inputWidth, inputHeight):fill(0)
gradOutput = torch.CudaTensor(output:size()):fill(0)


-- Begin training
print("# StochasticGradient: training")

iteration = 1

while true do
  local currentError = 0

  print('Epoch: ' .. iteration)

for slice = 0,9 do
-- Loading new data slice
print('Loading data slice ' .. i)
trainData = torch.load('../../data/trainvalFinal/trainvalFinal' .. i .. '.t7')
print('Done leading data slice ' .. i .. '\n')
        
print('Shuffling indices \n')
numExamples = #trainData.data
shuffledIndices = torch.randperm(numExamples, 'torch.LongTensor')

  for t = 1,numExamples do
    gradParams:zero();
    output:fill(0)
    gradOutput:fill(0)

    function feval(params)
      local input = trainData.data[shuffledIndices[t]]:cuda();
      local target = trainData.labels[shuffledIndices[t]]:cuda();

      output = net:forward(input)

      for i = 1,output:size(2) do
        for j = 1,output:size(3) do
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

  net:clearState()
  torch.save('../../data/AlexNet/AlexNetFCN' .. iteration .. '.t7', net)

  currentError = currentError / dataset:size()
  print('Current average error: ' .. currentError)

-- Freeing GPU memory
--output = nil
--gradOutput = nil
trainData.data = nil
trainData.labels = nil
collectgarbage()

end

  iteration = iteration + 1

  

  if maxIteration > 0 and iteration > maxIteration then
    print("# StochasticGradient: you have reached the maximum number of iterations")
    print("# training error = " .. currentError)
    break
  end
end
