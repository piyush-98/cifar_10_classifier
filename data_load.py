from google.colab import files

uploaded=files.upload()
for fn in uploaded.keys():
  print('user uploaded file "{name}" with length {length} bytes'.format(name=fn,length=len(uploaded[fn])))
##code to upload dataset files from the local directory to colab in .zip extension 

get_ipython().system('ls')


# In[14]:


import zipfile
zip_ref = zipfile.ZipFile('data_batch_5.zip', 'r')  ##code to unzip all the files
zip_ref.extractall()
zip_ref.close()
get_ipython().system('ls')
