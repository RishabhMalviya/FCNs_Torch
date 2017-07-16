require 'image'

labelColorMap = torch.load('labelColorMap.t7')

dataLoader = {}

function dataLoader.loadDatasetRT(i)
    trainData = {data = {}, labels = {}}

-- Load training images
	local imageSet = io.open("./VOCdevkit/VOC2012/ImageSets/Segmentation/x0" .. i)
	imageNumber = 1
	if imageSet then
		for line in imageSet:lines() do
			imageFile = './VOCdevkit/VOC2012/JPEGImages/' .. line .. '.jpg'
			trainData.data[imageNumber] = image.load(imageFile)
			imageNumber = imageNumber + 1
		end
	end
  print('Done loading input images from slice ' .. i)

	-- Pad each so that they're all 500x500
	for i = 1,#trainData.data do
    width = trainData.data[i]:size(2)
    height = trainData.data[i]:size(3)

    local tempCopy = trainData.data[i]
    trainData.data[i] = torch.Tensor(3,500,500):fill(0)
    trainData.data[i][{ {}, {1,width}, {1,height} }] = tempCopy
	end
  print('Done padding input images from slice ' .. i)


-- Load label images
	local imageSet = io.open("./VOCdevkit/VOC2012/ImageSets/Segmentation/x0" .. i)
	imageNumber = 1
	if imageSet then
		for line in imageSet:lines() do
			imageFile = './VOCdevkit/VOC2012/SegmentationClass/' .. line .. '.png'
			trainData.labels[imageNumber] = image.load(imageFile)
			imageNumber = imageNumber + 1
        end
    end
  print('Done loading label images from slice ' .. i)

  -- Convert label images to GT tensors and pad
  for i = 1,#trainData.labels do
    width = trainData.labels[i]:size(2)
    height = trainData.labels[i]:size(3)

    local tempCopy = trainData.labels[i]
    tempCopy = tempCopy*10
    tempCopy = tempCopy + 1
    tempCopy = tempCopy:type('torch.IntTensor')

    trainData.labels[i] = torch.Tensor(1,500,500):fill(22):type('torch.ByteTensor') -- 22 is the ignoreIndex

    for x = 1,width do
      for y = 1,height do
        trainData.labels[i][1][x][y] = labelColorMap[tempCopy[1][x][y]][tempCopy[2][x][y]][tempCopy[3][x][y]]
      end
    end
  end
  print('Done converting and padding label images from slice ' .. i .. '\n')

    return trainData
    
end


