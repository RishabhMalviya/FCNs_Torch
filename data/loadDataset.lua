require 'image'


trainData = {data = {}, labels = {}}
-- Find the trainSet file
local trainImages = io.open("./VOCdevkit/VOC2012/ImageSets/Segmentation/codeTesting.txt")
-- Load each image listed in the train.txt file
imageNumber = 1
if trainImages then
    for line in trainImages:lines() do
        imageFile = './VOCdevkit/VOC2012/JPEGImages/' .. line .. '.jpg'
        trainData.data[imageNumber] = image.load(imageFile)
        imageNumber = imageNumber + 1
    end
end
-- Pad each so that they're all 500x500
for i = 1,#trainData.data do
    local tempImage = trainData.data[i]
    trainData.data[i] = torch.Tensor(3,500,500):fill(0)
    trainData.data[i][{ {}, {1,tempImage:size()[2]}, {1,tempImage:size()[3]} }] = tempImage
end


--Load the labelColorMap
labelColorMap = torch.load('./labelColorMap.t7')


-- Load each labels image listed in the train.txt file
-- Locate the labelled images, load them
local trainImages = io.open("./VOCdevkit/VOC2012/ImageSets/Segmentation/codeTesting.txt")
imageNumber = 1
if trainImages then
    for line in trainImages:lines() do
        imageFile = './VOCdevkit/VOC2012/SegmentationClass/' .. line .. '.png'
        trainData.labels[imageNumber] = image.load(imageFile)
        imageNumber = imageNumber + 1
    end
end
-- Parse it with the labelColorMap to get a one-channel (class) image
for t = 1,#trainData.labels do
  local tIm = trainData.labels[t]
  trainData.labels[t] = torch.Tensor(1,tIm:size()[2],tIm:size()[3]):fill(-1)
  for x = 1,tIm:size()[2] do
    for y = 1,tIm:size()[3] do
      for c = 1,22 do
        if((math.abs(labelColorMap.color[c][1]-tIm[1][x][y])<0.0001) and (math.abs(labelColorMap.color[c][2]-tIm[2][x][y])<0.0001) and (math.abs(labelColorMap.color[c][3]-tIm[3][x][y])<0.0001)) then
          trainData.labels[t][1][x][y] = labelColorMap.label[c]
        end
      end
    end
  end
end
-- Pad them with the ignoreIndex so that it's all 500x500
for i = 1,#trainData.labels do
    local tempImage = trainData.labels[i]
    trainData.labels[i] = torch.Tensor(1,500,500):fill(255) --ignoreIndex
    trainData.labels[i][{ {}, {1,tempImage:size()[2]}, {1,tempImage:size()[3]} }] = tempImage
end


torch.save('./trainvalData.t7',trainData)
