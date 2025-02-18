**Example command:**

python cddd_encoder_torch.py **-i** test2.smi **-o** test.csv **--weights_dir** '/media/drives/drive1/robin/cddd/default_model/weights' **--batch_size** 32

Input = one SMILES string per row
Output = csv starting with SMILES, then all 512 features

To create the weights_dir, first execute the following code in the original CDDD environment in the cddd/default_model directory:

```# In original TensorFlow environment:
import tensorflow as tf
import numpy as np
import os

# Load the model
sess = tf.Session()
saver = tf.train.import_meta_graph('model.ckpt.meta')
saver.restore(sess, 'model.ckpt')
graph = sess.graph

# Get all variables
variables = tf.trainable_variables()

# Save each variable as a separate NumPy file
os.mkdir('weights')
for var in variables:
    value = sess.run(var)
    np.save(f'weights/{var.name.replace("/", "_").replace(":", "_")}.npy', value)```
