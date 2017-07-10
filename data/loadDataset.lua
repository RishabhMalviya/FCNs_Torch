require 'image'


trainData = {data = {}, labels = {}}

-- Find the trainSet file
local trainImages = io.open("./VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt")
-- Load each image listed in the train.txt file
if trainImages then
    for line in trainImages:lines() do
        imageFile = './VOCdevkit/VOC2012/JPEGImages/' .. line .. '.jpg'
        trainData.data[imageNumber] = image.load(imageFile)
        imageNumber = imageNumber + 1
    end
end  
-- Pad each so that they're all 500x500
for i = 1,#trainData.data do
    tempImage = trainData.data[i]
    trainData.data[i] = torch.Tensor(3,500,500):fill(0)
    trainData.data[i][{ {}, {1,tempImage:size()[2]}, {1,tempImage:size()[3]} }] = tempImage
end


-- Parse the labelColoMap.txt and load it


-- Load each labels image listed in the train.txt file
  -- Locate the image
  -- Load it
  -- Parse it with the labelColorMap to get a one-channel (class) image
  -- Convert each one-channel pixel into a one-hot vector
