import caffe
from caffe import layers as L #pseudo module using __getattr__ magic to generate protobuf messages
from caffe import params as P #pseudo module using __getattr__ magic to generate protobuf messages
 
 
def setLayers(leveldb, batch_size, deploy=False):
	#it is tricky to produce the deploy prototxt file, as the data input is not from a layer, so we have to creat a workaround
	#producing training and testing prototxt files is pretty straight forward
	n = caffe.NetSpec()
	
	if deploy==False:
		n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LEVELDB, source=leveldb,\
		ntop=2)
	# produce data definition for deploy net
	else:
		input="data"
		dim1=1
		dim2=1
		dim3=64
		dim4=32
	#make an empty "data" layer so the next layer accepting input will be able to take the correct blob name "data",
	#we will later have to remove this layer from the serialization string, since this is just a placeholder
	n.data=L.Layer()
	 
	n.conv2 = L.Convolution(n.data, kernel = 3, stride = 2, num_output=32, weight_filler=dict(type='xavier'))
	n.pool1 = L.Pooling(n.conv2, kernel = 3, stride_h=2, stride_w=1, pool=P.Pooling.MAX)
	 
	n.drop2=L.Dropout(n.pool1,dropout_ratio=0.1)
	n.ip1=L.InnerProduct(n.drop2, num_output=196, weight_filler=dict(type='xavier'))
	 
	n.relu1 = L.ReLU(n.ip1, in_place=True)
	n.ip4 = L.InnerProduct(n.relu1, num_output=12, weight_filler=dict(type='xavier'))
	 
	#n.loss layer is only in training and testing nets, but not in deploy net.
	if deploy==False:
		n.loss = L.SoftmaxWithLoss(n.ip4, n.label)
		return str(n.to_proto())
	#for generating the deploy net
	else:
		#generate the input information header string
		deploy_str='input: {}\ninput_dim: {}\ninput_dim: {}\ninput_dim: {}\ninput_dim: {}'.format('"'+input+'"', dim1, dim2, dim3, dim4)
		#assemble the input header with the net layers string.  remove the first placeholder layer from the net string.
		return deploy_str+'\n'+'layer {'+'layer {'.join(str(n.to_proto()).split('layer {')[2:])
	 
#write the net prototxt files out
with open('trainNet.prototxt', 'w') as f:
	print ('wrting train')
	f.write(setLayers('databases/train_subj1_leveldb', 100))
 
with open('testNet.prototxt', 'w') as f:
	print ('wrting test')
	f.write(setLayers('databases/test_subj1_leveldb', 100))
 
with open('deploy.prototxt', 'w') as f:
	f.write(str(setLayers('', 0, deploy=True)))