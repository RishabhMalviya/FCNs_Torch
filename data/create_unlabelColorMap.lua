require 'torch'

unLabelColorMap = torch.Tensor(22,3)

local Rmap = io.open('./labelColorMap_R.txt')
lineNumber = 1
for line in Rmap:lines() do
  if(lineNumber<21) then
    unLabelColorMap[lineNumber][1] = tonumber(line)
  end
  if(lineNumber==21) then
    unLabelColorMap[21][1] = 0  -- backgroundIndex
  end
  if(lineNumber==255) then
    unLabelColorMap[22][1] = tonumber(line) -- ignoreIndex
  end
  lineNumber = lineNumber+1
end

local Gmap = io.open('./labelColorMap_G.txt')
lineNumber = 1
for line in Gmap:lines() do
  if(lineNumber<21) then
    unLabelColorMap[lineNumber][2] = tonumber(line)
  end
  if(lineNumber==21) then
    unLabelColorMap[21][2] = 0  -- backgroundIndex
  end
  if(lineNumber==255) then
    unLabelColorMap[22][2] = tonumber(line) -- ignoreIndex
  end
  lineNumber = lineNumber+1
end

local Bmap = io.open('./labelColorMap_B.txt')
lineNumber = 1
for line in Bmap:lines() do
  if(lineNumber<21) then
    unLabelColorMap[lineNumber][3] = tonumber(line)
  end
  if(lineNumber==21) then
    unLabelColorMap[21][3] = 0  -- backgroundIndex
  end
  if(lineNumber==255) then
    unLabelColorMap[22][3] = tonumber(line) -- ignoreIndex
  end
  lineNumber = lineNumber+1
end


torch.save('./unLabelColorMap.t7', unLabelColorMap)
