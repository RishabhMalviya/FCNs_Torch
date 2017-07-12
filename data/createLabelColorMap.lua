require 'torch'

labelColorMap = {label=torch.Tensor(22), color=torch.Tensor(22,3)}
-- Parse the labelColoMap.txt and load it
local Rmap = io.open('./labelColorMap_R.txt')
lineNumber = 1
for line in Rmap:lines() do
    if(lineNumber<21) then
        labelColorMap.label[lineNumber] = lineNumber
        labelColorMap.color[lineNumber][1] = tonumber(line)
    end
    if(lineNumber==255) then
        labelColorMap.label[22] = lineNumber
        labelColorMap.color[22][1] = tonumber(line)
    end
    lineNumber = lineNumber+1
end

local Gmap = io.open('./labelColorMap_G.txt')
lineNumber = 1
for line in Gmap:lines() do
    if(lineNumber<21) then
        labelColorMap.color[lineNumber][2] = tonumber(line)
    end
    if(lineNumber==255) then
        labelColorMap.color[22][2] = tonumber(line)
    end
    lineNumber = lineNumber+1
end

local Bmap = io.open('./labelColorMap_B.txt')
lineNumber = 1
for line in Bmap:lines() do
    if(lineNumber<21) then
        labelColorMap.color[lineNumber][3] = tonumber(line)
    end
    if(lineNumber==255) then
        labelColorMap.color[22][3] = tonumber(line)
    end
    lineNumber = lineNumber+1
end
-- background label
labelColorMap.color[21] = torch.Tensor(3):fill(0)
labelColorMap.label[21] = 21

torch.save('./labelColorMap.t7', labelColorMap)
