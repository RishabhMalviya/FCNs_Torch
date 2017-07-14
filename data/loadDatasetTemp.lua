require 'image'


for i = 0,9 do

	trainData = {data = {}, labels = {}}

-- load training images
	local imageSet = io.open("./VOCdevkit/VOC2012/ImageSets/Segmentation/x0" .. i)

	imageNumber = 1
	if imageSet then
	    for line in imageSet:lines() do
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


-- load label images
	local imageSet = io.open("./VOCdevkit/VOC2012/ImageSets/Segmentation/x0" .. i)

	imageNumber = 1
	if imageSet then
	    for line in imageSet:lines() do
		imageFile = './VOCdevkit/VOC2012/SegmentationClass/' .. line .. '.png'
		trainData.labels[imageNumber] = image.load(imageFile)
		imageNumber = imageNumber + 1
	    end
	end

	torch.save('./trainvalData/trainvalDataTemp' .. i .. '.t7',trainData)

	print('Split ' .. i .. 'done.')

end
