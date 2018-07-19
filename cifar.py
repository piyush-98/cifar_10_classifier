
def labels_names():
    ls=[]
    with open('batches.meta', mode='rb') as file:    ##file containing the names of the classes to be classified eg. cats, dogs
        batch=pickle.load(file,encoding='bytes')
        for key in batch:
            print(key)
        ls=(batch[b'label_names'])
        names=[x.decode('utf-8') for x in ls]         ##func to return the names of tthe class classified from ine hoit encodeed labels
        print(names)
        return names


# In[ ]:



def convert_images(raw_images):                       ##function to convert THE IMAGES into 4d arrays of size(batch,32,32,3)
  
    import numpy as np                                ##32 is the image height and width and 3 indicates the rgb color channels if the image
    channels=3
    img_size=32
    raw_float = np.array(raw_images, dtype=float) / 255.0
    f_images=raw_float.reshape(-1,channels,img_size,img_size)       ##the fumc is ,made for the training datset
    f_images=f_images.transpose([0,2,3,1])
    return f_images   


# In[ ]:


def one_hot(raw_labels):                                ##func to one hot encode the 10 labels of the datset
    import numpy as np
    raw_labels=np.array(raw_labels,np.int32)
    n_values = np.max(raw_labels) + 1
    onehotDash=np.eye(n_values)[raw_labels]
    onehot=np.reshape(onehotDash,(len(raw_labels),10))##we reshape from a 3D matrix as built by default using eye function to a 2D array
    #print(onehot.shape)
    return onehot


# In[ ]:


def test_data():                             
    import pickle
    import numpy as np
    with open('test_batch', mode='rb') as file:                 ##function to convert the testing images in 4d arrays (batch,32,32,3)
        batch=pickle.load(file,encoding='bytes')
        for key in batch:
            print(key)
        raw_test_images=batch[b'data']
        test_images=convert_images(raw_test_images)             
        #print(test_images.shape)
        raw_test_labels=batch[b'labels']
        test_labels=one_hot(raw_test_labels)
        #print(test_labels.shape)
        return test_images,test_labels


# In[ ]:


def load_data():                                      ##THIS FUNCTION is to load data from the 5 unzipped files
  
    import pickle
    import numpy as np                                 ##the func makes array of (50000,3072) from the datafiles each containing 10000 images
    images=np.zeros(shape=[50000,3072])
    labels=np.zeros(shape=[50000,])
    beg=0
    for i in range(5):

        with open('data_batch_'+ str(i+1), mode='rb') as file:   ##opening a respective file
            batch=pickle.load(file,encoding='bytes')
            features=batch[b'data']
            raw_labels=batch[b'labels']                         ##batch here is a python dict with b'data' as keys this key has the trainig 
        num_image=len(features)
        last=beg+num_image
        images[beg:last, :]=features
        labels[beg:last]=raw_labels                              
        beg=last

      #print(images.shape)   
      #print(np.unique(labels))
    train_labels=one_hot(labels)                                ##calling the one hot func to encode the labels
      #print(train_labels)
    train_images=convert_images(images)                          ##calling the func to convert the iamges in to 4d arrays
      #print(train_images.shape)
    return train_images,train_labels
  


# In[ ]:


import tensorflow as tf                                  
x=tf.placeholder(tf.float32,shape=[None,32,32,3])         
y_=tf.placeholder(tf.float32,shape=[None,10])


# In[ ]:


def weight_init(shape):                                  
    weight=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(weight)


# In[ ]:


def bias_init(shape):
    bias=tf.constant(0.2,shape=shape)
    return tf.Variable(bias)


# In[ ]:


def conv_d(x,w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')         ##to apply convolution input taken as images and weights
                                                                            ##padding is taken as same to rertain the size


# In[ ]:


def max_pool(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# In[37]:


w_conv1=weight_init([5,5,3,16])
b_conv1=bias_init([16])
f_conv=(conv_d(x,w_conv1) + b_conv1)
f_pool=max_pool(f_conv)                                  ##convolution layer number 1 with 16 filters


w_conv2=weight_init([5,5,16,32])
b_conv2=bias_init([32])
s_conv=tf.nn.relu(conv_d(f_pool,w_conv2) + b_conv2)     ##convolution layer number 2 with 32 filters and 16 channels
s_pool=max_pool(s_conv)


w_conv3=weight_init([5,5,32,64])
b_conv3=bias_init([64])                                 ##convolution layer number 3 with 64 filters and 32 channels
t_conv=tf.nn.relu(conv_d(s_pool,w_conv3) + b_conv3)
t_pool=max_pool(t_conv)


weight_flat=weight_init([4*4*64,200])                    ##flattening the last conv layer to apply neural networks the 1st hidden layer contains 200 neurons
bias_flat=bias_init([200])
t_pool_flat=tf.reshape(t_pool,[-1,4*4*64])              ##bias and weights initialisation

keep_prob=tf.placeholder(tf.float32)                    ## a placeholder to regulate drop out  
scale1 = tf.Variable(tf.ones([200]))
shift1 = tf.Variable(tf.zeros([200]))
f1=(tf.matmul(t_pool_flat,weight_flat)+bias_flat)       ## f1 is the 1st hidden layer 
mean1, var1 = tf.nn.moments(f1, [0])                    ##tf.nn.moments returns the mean and variance of the layer 
h_fc1_normed = tf.nn.batch_normalization(f1,mean1,var1,shift1,scale1,1e-4)               ## to apply batch normalization
h_fc1=tf.nn.relu(h_fc1_normed)                            ##activation relu applied
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)                 


weight_flat2=weight_init([200,200])
bias_flat2=bias_init([200])
scale2 = tf.Variable(tf.ones([200]))
shift2 = tf.Variable(tf.zeros([200]))                               ##2nd hidden layer with 200 neurons
f2=(tf.matmul(h_fc1_drop,weight_flat2)+bias_flat2)
mean2, var2 = tf.nn.moments(f2, [0])
h_fc2_normed = tf.nn.batch_normalization(f2,mean2,var2,shift2,scale2,1e-4)
h_fc2=tf.nn.relu(h_fc2_normed)
h_fc2_drop=tf.nn.dropout(h_fc2,keep_prob)


weight_flat3=weight_init([200,200])
bias_flat3=bias_init([200])
scale3 = tf.Variable(tf.ones([200]))                               ##3rd  hidden layer with 200 neurons
shift3 = tf.Variable(tf.zeros([200]))
f3=(tf.matmul(h_fc2_drop,weight_flat3)+bias_flat3)
mean3,var3 = tf.nn.moments(f3, [0])
h_fc3_normed = tf.nn.batch_normalization(f3,mean3,var3,shift3,scale3,1e-4)    
h_fc3=tf.nn.relu(h_fc3_normed)
h_fc3_drop=tf.nn.dropout(h_fc2,keep_prob)


weight_flat4=weight_init([200,10])
bias_flat4=bias_init([10])
y_conv=(tf.matmul(h_fc3_drop,weight_flat4)+bias_flat4)              ##the output layer with 10 neurons 


cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))  ## cross entropy as the losss function
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)      ##Adam optimiser used with 0.001 stepsize a hyperparameter for regularisation 
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))    



with tf.Session() as sess:                           
    sess.run(tf.global_variables_initializer()) n
    (train_images,labels)=load_data()                  
    from sklearn.model_selection import train_test_split
    x_train,x_vali=train_test_split(train_images,train_size=0.8,test_size=0.2,shuffle=False)#divide data in a 80:20 ratio for training and validation
    y_train,y_vali=train_test_split(labels,train_size=0.8,test_size=0.2,shuffle=False)
    for j in range(20):   ##2o epochs                        

        for i in range(1000): ##1000 iterations per epoch
 

            batch_labels=y_train[i*40:(i+1)*40,:]       ##40 batchsize as the training dataset size is 40,000

            batch_images=x_train[i*40:(i+1)*40,:]
            train_step.run(feed_dict={x:batch_images , y_:batch_labels , keep_prob: 0.5}) ## running the trainstep variable keep prob kept as 0.5 to avoid overfittong in training phase

        train_accuracy = accuracy.eval(feed_dict={x:x_vali, y_:y_vali , keep_prob: 1.0})   ##keep prob kept 1 for validating (no drop out)
        print("epoch %d, training accuracy %g"%(j, train_accuracy))
        print(sess.run(cross_entropy,feed_dict={x:x_vali,y_:y_vali, keep_prob: 1.0}))    ##vali accuracy                                      
    test_images,test_labels=test_data()
    print("test accuracy %g"%accuracy.eval(feed_dict={                                    ##testing accuracy
        x:test_images , y_: test_labels, keep_prob: 1.0}))


