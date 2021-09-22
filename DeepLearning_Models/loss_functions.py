import tensorflow as tf

def BCE_loss_2D(gray, pred):
    pred = pred + 1e-8
    gray = tf.image.convert_image_dtype(gray, tf.float32)
    gray = gray + 1e-8
    loss = - (gray * tf.math.log(pred) + (1-gray) * tf.math.log((1-pred)))
    loss = tf.math.reduce_mean(loss)
    return loss
print('--BCE_loss_2D(gray, pred)')