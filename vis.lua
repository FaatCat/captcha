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
--multiplied = tonumber(arg[3]) * tonumber(arg[4])
filters = layer.weight:view(64,1, 3, 3)

image.save('vis/filter'..arg[2]..'.jpg', image.toDisplayTensor{input=filters, padding=3, nrow=math.sqrt(filters:size(1))})
--image.save('vis/filter'..arg[2]..'.jpg', filters)
