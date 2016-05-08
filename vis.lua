require 'image'
require 'nn'
require 'cudnn'
require 'rnn'

print('loading model..')
model = torch.load(arg[1])
print('loaded model')

cnn = model:get(1)

print('CNN: ')
print(cnn)

layer = cnn:get(tonumber(arg[2]))
image.save('vis/filter'..arg[2]..'.jpg', layer.weight:view(3,tonumber(arg[3])*tonumber(arg[4]),3))
