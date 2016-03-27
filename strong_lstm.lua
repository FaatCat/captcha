require 'rnn'
require 'MultiCrossEntropyCriterion'
--require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nn'
data = require 'data'
dir = 'data/'
--X,Y = data.storeXY(dir,120,240,'captchaImage.')
print('loadingXY')
X,Y = data.loadXY(dir)
--Y = Y:cuda()
local sample_data = 505

print(Y[sample_data])

model = torch.load('lstm_dec.t7')
ct = torch.load('ct.t7')
tnet = torch.load('tnet.t7')

model:evaluate()
model:cuda()


local batch = X[{{sample_data}}]:cuda()
--require 'image'
--batch = image.load('data/captchaImage.1.png'):cuda()
--print(batch:size())
--torch.reshape(batch,10)
--batch = tnet:forward(batch)
out = model:forward(batch)

--print(out)
--torch.save('out.t7',out)
--local tmp, maxoutput = out:max(2)

local Y = nil
local N = out[1]:size(1)
local C = out[1]:size(2)

for k=1,#out do
   --confidence, index = out[i]:max(2)
   Y = Y and torch.cat(Y,out[k]:reshape(N,1,C),2) or out[k]:reshape(N,1,C)
end

confidence, index = Y:max(3)

print(index)
