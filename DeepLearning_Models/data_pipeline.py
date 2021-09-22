import tensorflow as tf

def create_dataset(batch_size):
    def read_img(path):
        rgb = tf.strings.join([path,'rgb.jpg'],'/')
        rgb = tf.io.read_file(rgb)
        rgb = tf.io.decode_jpeg(rgb,channels=3)
        gray = tf.strings.join([path,'gray.jpg'],'/')
        gray = tf.io.read_file(gray)
        gray = tf.io.decode_jpeg(gray,channels=1)
        gray = tf.squeeze(gray)
        return rgb,gray
    ds = tf.data.Dataset.list_files('data/train/*')
    ds = ds.map(read_img).padded_batch(batch_size,([None,None,3],[None,None]))
    return ds

print('--create_dataset(batch_size)')